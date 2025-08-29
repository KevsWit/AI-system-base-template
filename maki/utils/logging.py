import os
import json
import uuid
import logging
import contextvars
from typing import Any, Dict, List, Tuple, Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from ..config import settings

# --------------------------
# Config por variables de entorno
# --------------------------
LOG_LEVEL = settings.LOG_LEVEL
LOG_DIR = settings.LOG_DIR
LOG_FILE = settings.LOG_FILE
LOG_JSON = settings.LOG_JSON
LOG_MAX_MB = settings.LOG_MAX_MB
LOG_BACKUPS = settings.LOG_BACKUPS

# --------------------------
# Contexto por ejecución
# --------------------------
_ctx_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("maki_run_id", default=None)
_ctx_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("maki_session_id", default=None)

def start_run(run_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Inicializa contexto de logging para una ejecución."""
    _ctx_run_id.set(run_id or str(uuid.uuid4()))
    if session_id is not None:
        _ctx_session_id.set(session_id)

def get_run_id() -> str:
    rid = _ctx_run_id.get()
    if rid is None:
        rid = str(uuid.uuid4())
        _ctx_run_id.set(rid)
    return rid

def get_session_id() -> Optional[str]:
    return _ctx_session_id.get()

# --------------------------
# Nivel TRACE (más fino que DEBUG)
# --------------------------
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)
    logging.Logger.trace = trace  # type: ignore

# --------------------------
# Formatters
# --------------------------
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None) or get_run_id(),
            "session_id": getattr(record, "session_id", None) or get_session_id(),
        }
        # Campos estructurados comunes para trazas de agentes
        for k in ("event", "node", "domain", "stage"):
            v = getattr(record, k, None)
            if v is not None:
                base[k] = v
        # Merge de extras si vienen como dict
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)

class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        run_id = getattr(record, "run_id", None) or get_run_id()
        session_id = getattr(record, "session_id", None) or get_session_id()
        prefix = f"run={run_id}"
        if session_id:
            prefix += f" session={session_id}"
        # opcionales
        node = getattr(record, "node", None)
        event = getattr(record, "event", None)
        domain = getattr(record, "domain", None)
        stage = getattr(record, "stage", None)
        tags = " ".join(
            f"{k}={v}" for k, v in (("node", node), ("event", event), ("domain", domain), ("stage", stage)) if v
        )
        if tags:
            prefix += f" {tags}"
        record.msg = f"{prefix} | {record.getMessage()}"
        return super().format(record)

# --------------------------
# Logger factory
# --------------------------
def _make_logger() -> logging.Logger:
    logger = logging.getLogger("maki")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(
        JSONFormatter() if LOG_JSON else TextFormatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(ch)

    # Archivo rotativo
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(LOG_DIR / LOG_FILE, maxBytes=LOG_MAX_MB * 1024 * 1024, backupCount=LOG_BACKUPS, encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(JSONFormatter() if LOG_JSON else TextFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    logger.propagate = False
    return logger

logger = _make_logger()

# --------------------------
# Helpers de traza de agentes
# --------------------------
def event(
    message: str = "",
    *,
    node: Optional[str] = None,
    event: Optional[str] = None,
    domain: Optional[str] = None,
    stage: Optional[str] = None,
    level: int = TRACE_LEVEL_NUM,
    **extra_fields: Any,
) -> None:
    """
    Log estructurado de un evento de agente.
    Uso: event("draft generado", node="research.draft", stage="research", domain="GYM", tokens=123)
    """
    logger.log(
        level,
        message,
        extra={
            "run_id": get_run_id(),
            "session_id": get_session_id(),
            "node": node,
            "event": event,
            "domain": domain,
            "stage": stage,
            "extra_fields": extra_fields or None,
        },
    )

def append_and_log(state: Dict[str, Any], key: str, value: str, **fields: Any) -> Dict[str, Any]:
    """
    Añade (key, value) al state['trace'] y lo escribe al log estructurado.
    Devuelve el mismo state para usarlo inline.
    """
    trace: List[Tuple[str, str]] = state.get("trace", [])
    trace.append((key, value))
    state["trace"] = trace
    event(value, node=key, event="trace", **fields)
    return state

def persist_agent_trace(state: Dict[str, Any], **fields: Any) -> None:
    """
    Vuelca todo el state['trace'] al log (útil al final de Workflow.run()).
    No modifica el estado.
    """
    trace: List[Tuple[str, str]] = state.get("trace", [])
    if not trace:
        return
    # Guardamos un bloque resumido y además cada ítem para búsquedas finas
    logger.info(
        f"Persistiendo trace con {len(trace)} eventos",
        extra={
            "run_id": get_run_id(),
            "session_id": get_session_id(),
            "event": "trace.persist",
            "extra_fields": fields or None,
        },
    )
    for key, val in trace:
        logger.log(
            TRACE_LEVEL_NUM,
            str(val),
            extra={
                "run_id": get_run_id(),
                "session_id": get_session_id(),
                "node": key,
                "event": "trace.item",
                "extra_fields": fields or None,
            },
        )
