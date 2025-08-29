from ..config import settings

def welcome_tool() -> str:
    """Tool de ejemplo que abre el flujo con un saludo/estado."""
    return settings.WELCOME_TOOL_TEXT
