from typing import Any

from langchain_core.tools import tool

from src.rag import responder


@tool
def consultar_rag(pregunta: str) -> dict[str, Any]:
    """
    Consulta información sobre gestión de residuos en Madrid
    mediante el sistema RAG local de ReciclaTIA.
    """
    return responder(pregunta)