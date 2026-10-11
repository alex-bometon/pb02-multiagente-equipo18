"""Carga GEMINI_API_KEY desde .env o prompt interactivo."""

import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

_configured = False  # evita cargar .env / pedir la key más de una vez
_ROOT = Path(__file__).resolve().parent


def configurar_gemini_api_key() -> None:
    """Lee GEMINI_API_KEY del .env (o la pide por consola la primera vez)."""
    global _configured
    if _configured:
        return

    load_dotenv(_ROOT / ".env")

    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass(
            "Pega aquí tu GEMINI_API_KEY (input oculto): "
        )

    print(
        "GEMINI_API_KEY configurada:",
        "sí" if os.getenv("GEMINI_API_KEY") else "no",
    )
    _configured = True
