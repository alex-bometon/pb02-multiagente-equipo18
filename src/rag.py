"""Orquesta el RAG de ReciclaTIA sin duplicar lógica de dominio.

Coordina la recuperación en ChromaDB y la generación fundamentada
con Gemini. No gestiona la interfaz, herramientas ni agentes LangGraph.
"""

from time import perf_counter
from typing import Any

from config import GENERATION_MODEL, TOP_K
from src.generate import generar_respuesta
from src.retrieve import buscar_chunks_relevantes


def _validar_top_k(top_k: int) -> int:
    """Impide pedir más chunks de los que procesa la generación."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k debe ser un número entero.")

    if not 1 <= top_k <= TOP_K:
        raise ValueError(
            f"top_k debe estar entre 1 y {TOP_K} para este RAG."
        )

    return top_k


def responder(
    pregunta: str,
    *,
    top_k: int = TOP_K,
    query_embedding: list[float] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Devuelve una respuesta RAG, sus fuentes y datos de ejecución.

    Args:
        pregunta: Pregunta del usuario, validada por el retrieval.
        top_k: Máximo de chunks recuperados, entre 1 y TOP_K.
        query_embedding: Vector opcional ya calculado. Permite evaluar
            el RAG sin repetir una llamada a la API de embeddings.
        client: Cliente Gemini opcional para generación. Si se inyecta,
            su ciclo de vida sigue siendo responsabilidad del llamador.

    Returns:
        Diccionario con pregunta, respuesta, abstencion, motivo,
        fuentes, chunks y metricas. Las fuentes proceden de la
        validación de citas realizada en src.generate.

    Raises:
        ValueError: Entrada o parámetros inválidos.
        FileNotFoundError: No existe el índice de ChromaDB.
        RuntimeError: Fallo técnico de retrieval o generación.

    Notes:
        Los errores técnicos se propagan y no se convierten en
        abstenciones. No se invoca el generador de embeddings cuando
        se proporciona query_embedding.
    """
    top_k = _validar_top_k(top_k)
    inicio = perf_counter()

    chunks = buscar_chunks_relevantes(
        pregunta,
        top_k=top_k,
        query_embedding=query_embedding,
    )
    fin_retrieval = perf_counter()

    resultado = generar_respuesta(pregunta, chunks, client=client)
    fin_generacion = perf_counter()

    return {
        "pregunta": pregunta.strip(),
        "respuesta": resultado["respuesta"],
        "abstencion": resultado["abstencion"],
        "motivo": resultado["motivo"],
        "fuentes": resultado["fuentes"],
        "chunks": chunks,
        "metricas": {
            "top_k": top_k,
            "chunks_recuperados": len(chunks),
            "modelo_configurado": GENERATION_MODEL,
            "retrieval_ms": round(
                (fin_retrieval - inicio) * 1_000, 2
            ),
            "generacion_ms": round(
                (fin_generacion - fin_retrieval) * 1_000, 2
            ),
            "total_ms": round(
                (fin_generacion - inicio) * 1_000, 2
            ),
        },
    }
