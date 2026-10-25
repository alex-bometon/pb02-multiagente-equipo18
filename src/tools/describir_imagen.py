"""Herramienta multimodal para describir residuos en imágenes."""


from google.genai import types
from langchain_core.tools import tool
from PIL import Image

from config import (
    BASE_DIR,
    ASSETS_DIR,
    IMAGE_MODEL,
    IMAGE_MAX_BYTES,
    IMAGE_MAX_PIXELS,
    IMAGE_ALLOWED_EXTENSIONS,
    IMAGE_ALLOWED_FORMATS,
)
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

    directorio_permitido = ASSETS_DIR.resolve()
    ruta = (BASE_DIR / ruta_imagen).resolve()

    if not ruta.is_relative_to(directorio_permitido):
        raise ValueError("La imagen debe estar dentro de assets/.")

    if not ruta.is_file():
        raise FileNotFoundError("No se encuentra la imagen indicada.")

    if ruta.suffix.lower() not in IMAGE_ALLOWED_EXTENSIONS:
        raise ValueError("Solo se admiten imágenes JPEG o PNG.")

    if ruta.stat().st_size > IMAGE_MAX_BYTES:
        raise ValueError("La imagen no puede superar los 5 MB.")

    # =========================================================
    # VALIDACIÓN DE LA IMAGEN
    # =========================================================

    with Image.open(ruta) as imagen:
        if imagen.format not in IMAGE_ALLOWED_FORMATS:
            raise ValueError("El contenido no es JPEG ni PNG.")

        if imagen.width * imagen.height > IMAGE_MAX_PIXELS:
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
            model=IMAGE_MODEL,
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
