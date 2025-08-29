from typing import Callable, Optional
from .utils import logger
from .utils import GYM_DOCS, CLIENTS_DOCS
from .retriever.builder import make_docs, SimpleKeywordRetriever
from .agents import Workflow
from .config.settings import settings
import uuid
from .utils import start_run

from typing import Callable, Optional
from .utils.logging import logger
from .config.settings import settings

from typing import Callable, Optional
from .utils.logging import logger
from .config.settings import settings

def _openai_fallback_callable() -> Optional[Callable[[str], str]]:
    """
    Devuelve una función de fallback con OpenAI si hay API key; si no, None.
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("Fallback OpenAI desactivado: OPENAI_API_KEY no está configurada.")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        model = settings.MODEL_NAME or "gpt-4o-mini"
        max_toks = int(getattr(settings, "MAX_OUTPUT_TOKENS", 400))

        def _call(q: str) -> str:
            try:
                prompt = (
                    "No encontré datos relevantes en el set local (gyms/clientes). "
                    "Responde con conocimiento general en español, de forma breve y útil.\n\n"
                    f"Pregunta: {q}"
                )
                rsp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Eres un asistente experto y conciso."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=max_toks,
                )
                return (rsp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.exception(f"Error en completion fallback: {e}")
                return ""

        logger.info(f"Fallback OpenAI activo con modelo: {model}")
        return _call

    except Exception as e:
        logger.warning(f"No se activó fallback OpenAI: {e}")
        return None


class Maki:
    """Clase principal importable desde backend, script o UI."""
    def __init__(self):
        gym_docs = make_docs(GYM_DOCS)
        cli_docs = make_docs(CLIENTS_DOCS)
        self.gym_retriever = SimpleKeywordRetriever(gym_docs)
        self.clients_retriever = SimpleKeywordRetriever(cli_docs)
        self.workflow = Workflow(
            self.gym_retriever, self.clients_retriever, completion_fallback=_openai_fallback_callable()
        )
        self.session_id = str(uuid.uuid4())

    def ask(self, message: str) -> dict:
        """Retorna dict con final_answer, domain y trace."""
        start_run(session_id=self.session_id)
        return self.workflow.run(message)
