from typing import Dict
import json
import re
from ..config import settings

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


class VerificationAgent:
    """
    Verifica con LLM si el borrador está soportado por los fragmentos (cuando existan)
    y su relevancia respecto a la pregunta. Devuelve flags y notas.
    """

    def __init__(self):
        self.client = None
        if _HAS_OPENAI and settings.OPENAI_API_KEY:
            try:
                self.client = OpenAI(settings.OPENAI_API_KEY)
            except Exception:
                self.client = None

    def run(self, question: str, draft: str) -> Dict:
        if not draft.strip():
            return {
                "supported": "NO",
                "relevant": "NO",
                "notes": "No hubo evidencia suficiente para construir respuesta."
            }

        # 1) LLM: pedimos salida JSON estricta
        if self.client:
            system = "Eres un verificador. Responde SOLO JSON con las claves: supported, relevant, notes."
            user = (
                "Evalúa este borrador de respuesta.\n\n"
                f"Pregunta: {question}\n\n"
                f"Respuesta borrador:\n{draft}\n\n"
                'Devuelve JSON con: {"supported": "YES|NO", "relevant": "YES|NO", "notes": "texto breve"}'
            )
            try:
                rsp = self.client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=120,
                )
                raw = (rsp.choices[0].message.content or "").strip()
                # intenta parsear JSON (tolera Markdown)
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start:end+1])
                    sup = str(parsed.get("supported", "NO")).upper()
                    rel = str(parsed.get("relevant", "NO")).upper()
                    notes = str(parsed.get("notes", "")).strip()
                    return {"supported": "YES" if sup.startswith("Y") else "NO",
                            "relevant": "YES" if rel.startswith("Y") else "NO",
                            "notes": notes}
            except Exception:
                pass

        # 2) Fallback simple
        def _toks(s: str):
            return re.findall(r"[a-záéíóúñ0-9]+", s.lower())

        # quita la pregunta del draft si aparece (eco)
        draft_wo_q = re.sub(re.escape(question), "", draft, flags=re.IGNORECASE)

        q_tokens = set(_toks(question)) - {"que","qué","hay","hasta","el","la","los","las","de","del","y","o","u","a","en","un","una"}
        d_tokens = set(_toks(draft_wo_q))

        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        relevant = "YES" if overlap >= 0.20 else "NO"

        supported = "YES" if len(draft_wo_q.strip()) >= 30 else "NO"

        notes = ""
        if relevant == "NO":
            notes = "La respuesta no aborda la pregunta."
        elif supported == "NO":
            notes = "La respuesta es demasiado vaga o corta."

        return {"supported": supported, "relevant": relevant, "notes": notes}