
"""Herramienta multimodal para describir residuos en imágenes."""

from pathlib import Path

from google.genai import types
from langchain_core.tools import tool
from PIL import Image

from config import BASE_DIR, GENERATION_MODEL
from src.gemini_client import crear_cliente_gemini


@tool
def describir_imagen(ruta_imagen: str) -> str:
    """
    Identifica y describe residuos visibles en una imagen.

    Solo admite imágenes JPEG o PNG almacenadas en assets/.
    No determina el contenedor de reciclaje.
    """

    # =========================================================
    # VALIDACIÓN DE LA RUTA
    # =========================================================

    if not isinstance(ruta_imagen, str) or not ruta_imagen.strip():
        raise ValueError("Debes proporcionar una ruta de imagen.")

    directorio_permitido = (BASE_DIR / "assets").resolve()
    ruta = (BASE_DIR / ruta_imagen).resolve()

    if not ruta.is_relative_to(directorio_permitido):
        raise ValueError("La imagen debe estar dentro de assets/.")

    if not ruta.is_file():
        raise FileNotFoundError("No se encuentra la imagen indicada.")

    if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Solo se admiten imágenes JPEG o PNG.")

    if ruta.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("La imagen no puede superar los 5 MB.")

    # =========================================================
    # VALIDACIÓN DE LA IMAGEN
    # =========================================================

    with Image.open(ruta) as imagen:
        if imagen.format not in {"JPEG", "PNG"}:
            raise ValueError("El contenido no es JPEG ni PNG.")

        if imagen.width * imagen.height > 16_000_000:
            raise ValueError("La imagen supera los 16 megapíxeles.")

        mime_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
        }[imagen.format]

        imagen.verify()

    # =========================================================
    # ANÁLISIS MULTIMODAL CON GEMINI
    # =========================================================

    cliente = crear_cliente_gemini()

    try:
        resultado = cliente.models.generate_content(
            model=GENERATION_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=ruta.read_bytes(),
                    mime_type=mime_type,
                ),
                (
                    "Describe en español los residuos u objetos "
                    "visibles en la imagen. Identifica sus materiales "
                    "solo cuando sean reconocibles. "
                    "Si existe incertidumbre, indícalo. "
                    "No determines el contenedor de reciclaje "
                    "ni inventes características que no sean visibles."
                ),
            ],
        )
    finally:
        cliente.close()

    descripcion = (resultado.text or "").strip()

    if not descripcion:
        raise RuntimeError("Gemini no ha devuelto una descripción.")

    return descripcion
