"""Validación compartida de preguntas entre capas del RAG."""

from config import MAX_PREGUNTA_CARACTERES


def validar_pregunta(pregunta: str) -> str:
    """Exige texto no vacío y respeta el límite global de longitud."""
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta debe ser un texto no vacío.")

    limpia = pregunta.strip()
    if len(limpia) > MAX_PREGUNTA_CARACTERES:
        raise ValueError(
            "La pregunta no puede superar "
            f"{MAX_PREGUNTA_CARACTERES} caracteres."
        )
    return limpia
