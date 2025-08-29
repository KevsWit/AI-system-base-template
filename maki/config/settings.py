import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv  # asegúrate de tener python-dotenv instalado

# Carga .env si existe (no rompe si no está)
load_dotenv()

# Importa las constantes con alias para evitar sombra de nombres
from .constants import (
    MAX_OUTPUT_TOKENS as _MAX_OUTPUT_TOKENS,
    GYM_KEYWORDS as _GYM_KEYWORDS,
    CLIENTS_KEYWORDS as _CLIENTS_KEYWORDS,
    IRRELEVANT_MSG as _IRRELEVANT_MSG,
    WELCOME_TOOL_TEXT as _WELCOME_TOOL_TEXT,
)

@dataclass
class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", _MAX_OUTPUT_TOKENS))

    # Palabras clave (usar default_factory para mutables)
    GYM_KEYWORDS: set[str] = field(default_factory=lambda: set(_GYM_KEYWORDS))
    CLIENTS_KEYWORDS: set[str] = field(default_factory=lambda: set(_CLIENTS_KEYWORDS))

    # Mensajes
    IRRELEVANT_MSG: str = _IRRELEVANT_MSG
    WELCOME_TOOL_TEXT: str = _WELCOME_TOOL_TEXT

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
    LOG_FILE: str = os.getenv("LOG_FILE", "maki.log")
    LOG_JSON: bool = os.getenv("LOG_JSON", "0").lower() in ("1", "true", "yes")
    LOG_MAX_MB: int = int(os.getenv("LOG_ROTATE_MB", "10"))
    LOG_BACKUPS: int = int(os.getenv("LOG_ROTATE_BACKUPS", "7"))

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

settings = Settings()
