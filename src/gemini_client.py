# Se encarga únicamente de la conexión con la API de gemini

import os

from dotenv import load_dotenv
from google import genai

from config import BASE_DIR


def crear_cliente_gemini() -> genai.Client:
    """
    Crea y devuelve un cliente de Gemini API.

    La clave se obtiene del archivo .env situado
    en la raíz del proyecto.

    Raises:
        ValueError:
            Si GEMINI_API_KEY no está configurada.
    """

    ruta_env = BASE_DIR / ".env"

    load_dotenv(dotenv_path=ruta_env)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "No se ha encontrado GEMINI_API_KEY. "
            "Configúrala en el archivo .env."
        )

    return genai.Client(api_key=api_key)