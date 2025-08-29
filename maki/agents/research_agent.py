from typing import List, Dict
from ..retriever.builder import Document
import os

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


class ResearchAgent:
    """Síntesis con LLM si disponible; si no, unión legible de snippets (sin eco de la pregunta)."""

    def __init__(self):
        self.client = None
        if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                self.client = None

    def run(self, question: str, docs: List[Document]) -> Dict:
        if not docs:
            return {"answer": "", "evidence": ""}

        snippets = [f"- {d.title}: {d.text}" for d in docs]

        # 1) LLM
        if self.client:
            system = (
                "Eres un analista que responde de forma breve y precisa en el idioma que se mantenga la conversación, "
                "basándote SOLO en los fragmentos provistos."
            )
            content = (
                f"Pregunta: {question}\n\n"
                f"Fragmentos:\n{chr(10).join(snippets)}\n\n"
                "Redacta una respuesta concisa (3 a 6 líneas) basada SOLO en los fragmentos. "
                "No repitas literalmente la pregunta."
            )
            try:
                rsp = self.client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": content}],
                    temperature=0.2,
                    max_tokens=300,
                )
                answer = (rsp.choices[0].message.content or "").strip()
                return {"answer": answer, "evidence": "\n".join(snippets)}
            except Exception:
                pass

        # 2) Fallback local (sin eco de la pregunta)
        synthesis = _simple_synthesis(snippets)
        return {"answer": synthesis, "evidence": "\n".join(snippets)}


def _simple_synthesis(snippets: list) -> str:
    body = " ".join(snippets)
    return f"A partir de los datos disponibles: {body}"
