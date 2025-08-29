from typing import List
import os, re
from ..retriever.builder import Document
from ..config import settings

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from ..retriever.builder import _Embedder  # usa el mismo backend del builder
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False


_SP_STOP = {
    "el","la","los","las","de","del","y","o","u","a","en","un","una","que","qué",
    "cual","cuál","cuales","cuáles","hasta","hay","al","por","para","con","sobre",
    "esto","eso","este","esta","esas","estos","estas","lo","su","sus","mi","mis",
}
_AI_TERMS = {"gpt","chatgpt","openai","modelo","modelos","version","versión","versiones","ia","llm","api","oai"}

def _tokens(s: str):
    return re.findall(r"[a-záéíóúñ0-9]+", s.lower())

class RelevanceChecker:
    """Clasificador con prioridad: LLM -> Embeddings -> Heurística por keywords (con filtros de IA)."""

    def __init__(self):
        self.client = None
        if _HAS_OPENAI and settings.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception:
                self.client = None
        self.embedder = _Embedder() if _HAS_EMB else None

        # descriptores semánticos (centroides ligeros)
        self._gym_desc = (
            "gimnasio, gym, membresía, planes, precios, máquinas, pesas, clases, "
            "spinning, cross training, yoga, entrenador, sedes, horarios"
        )
        self._cli_desc = (
            "cliente, clientes, segmentación, retención, churn, contacto, WhatsApp, email, "
            "preferencias, marketing, comunicación"
        )

        if self.embedder and self.embedder.available():
            self._emb_gym = self.embedder.embed([self._gym_desc])[0]
            self._emb_cli = self.embedder.embed([self._cli_desc])[0]
        else:
            self._emb_gym = None
            self._emb_cli = None

    def classify(self, question: str, docs: List[Document]) -> str:
        # --- Filtro duro: términos de IA => UNKNOWN
        q_toks = [t for t in _tokens(question) if t not in _SP_STOP]
        if any(t in _AI_TERMS for t in q_toks):
            return "UNKNOWN"

        # --- 1) LLM (si está disponible)
        if self.client:
            hints = "\n".join([f"- {d.title}: {d.text[:140]}" for d in docs[:3]])
            prompt = f"""
Clasifica la consulta SOLO en: GYM, CLIENTS o UNKNOWN.
Etiqueta UNKNOWN si la consulta no trata de gimnasios o clientes, o si se refiere a IA/LLMs/GPT/OpenAI.

Consulta: {question}
Pistas (opcionales):
{hints if hints else "- (sin pistas)"}

Responde SOLO una palabra: GYM, CLIENTS o UNKNOWN.
""".strip()
            try:
                rsp = self.client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", settings.MODEL_NAME),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5,
                )
                label = (rsp.choices[0].message.content or "").strip().upper()
                if label.startswith("GYM"):
                    return "GYM"
                if label.startswith("CLIENT"):
                    return "CLIENTS"
                return "UNKNOWN"
            except Exception:
                pass

        # --- 2) Embeddings (si hay)
        if self.embedder and self.embedder.available() and self._emb_gym is not None:
            qv = self.embedder.embed([question])[0]
            sim_g = float(self._emb_gym @ qv)
            sim_c = float(self._emb_cli @ qv)
            # umbral más alto = preferimos UNKNOWN ante duda
            if max(sim_g, sim_c) < 0.40:
                return "UNKNOWN"
            # si está muy parejo, también UNKNOWN
            if abs(sim_g - sim_c) < 0.05:
                return "UNKNOWN"
            return "GYM" if sim_g > sim_c else "CLIENTS"

        # --- 3) Heurística con cobertura de keywords (sin stopwords)
        g_cover = sum(1 for t in q_toks if t in settings.GYM_KEYWORDS)
        c_cover = sum(1 for t in q_toks if t in settings.CLIENTS_KEYWORDS)

        if g_cover == 0 and c_cover == 0:
            return "UNKNOWN"
        if g_cover >= c_cover * 1.3:  # ligero margen
            return "GYM"
        if c_cover >= g_cover * 1.3:
            return "CLIENTS"
        return "UNKNOWN"
