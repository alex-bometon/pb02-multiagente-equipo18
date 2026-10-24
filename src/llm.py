
"""Inicialización del modelo Gemini para los agentes LangGraph.

El RAG mantiene su cliente nativo independiente.
Este módulo solo proporciona el modelo de LangChain.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    BASE_DIR,
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    SUPERVISOR_TEMPERATURE,
    GENERATION_MAX_RETRIES,
)


def get_llm(
    *,
    supervisor: bool = False,
) -> ChatGoogleGenerativeAI:
    """
    Crea el modelo Gemini utilizado por el sistema multiagente.

    Args:
        supervisor: Si es True, utiliza la temperatura
            configurada para el supervisor. En caso contrario,
            utiliza la temperatura de los especialistas.

    Returns:
        Instancia de ChatGoogleGenerativeAI.

    La clave GEMINI_API_KEY se carga desde .env y la
    biblioteca LangChain la utiliza automáticamente.
    """

    load_dotenv(dotenv_path=BASE_DIR / ".env")

    temperatura = (
        SUPERVISOR_TEMPERATURE
        if supervisor
        else AGENT_TEMPERATURE
    )

    return ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        temperature=temperatura,
        max_retries=GENERATION_MAX_RETRIES,
    )
