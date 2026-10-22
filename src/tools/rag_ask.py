"""Adaptador fino al RAG completo y ya existente de ReciclaTIA."""

from typing import Any


def rag_ask(consulta: str) -> dict[str, Any]:
    """Consulta el RAG local de residuos de Madrid sin repetir su lógica."""
    if not isinstance(consulta, str):
        raise ValueError("consulta debe ser una cadena de texto")
    # Import tardío: no se carga ChromaDB ni se requiere una API key al importar tools.
    from src.rag import responder

    resultado = responder(consulta)
    return {
        "origen": "rag_local",
        "respuesta": resultado["respuesta"],
        "abstencion": resultado["abstencion"],
        "motivo": resultado["motivo"],
        "fuentes": resultado["fuentes"],
        "metricas": resultado["metricas"],
    }
