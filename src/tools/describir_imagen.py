"""Multimodal: acceso solo a una imagen demo de reciclaje autorizada."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

import config

TIPOS = {"JPEG": "image/jpeg", "PNG": "image/png"}
EXTENSIONES = {"JPEG": ".jpg", "PNG": ".png"}


def _leer_asset(nombre_imagen: str) -> tuple[bytes, str]:
    if not isinstance(nombre_imagen, str) or nombre_imagen not in config.TOOLS_IMAGENES_AUTORIZADAS:
        raise ValueError("Imagen no autorizada")
    if Path(nombre_imagen).name != nombre_imagen:
        raise ValueError("No se admiten rutas")
    raiz = config.TOOLS_ASSETS_DIR.resolve()
    archivo = config.TOOLS_ASSETS_DIR / nombre_imagen
    if archivo.is_symlink() or not archivo.is_file() or archivo.resolve().parent != raiz:
        raise ValueError("Imagen inexistente o ruta no permitida")
    tamano = archivo.stat().st_size
    if tamano <= 0 or tamano > config.TOOLS_IMAGEN_MAX_BYTES:
        raise ValueError("La imagen supera el tamaño permitido o está vacía")
    try:
        with Image.open(archivo) as imagen:
            formato = imagen.format
            if (formato not in TIPOS or
                archivo.suffix.lower() != EXTENSIONES[formato] or
                imagen.width * imagen.height > config.TOOLS_IMAGEN_MAX_PIXELES):
                raise ValueError("Formato o dimensiones no permitidos")
            imagen.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError("Imagen inválida") from e
    return archivo.read_bytes(), TIPOS[formato]


def describir_imagen(nombre_imagen: str = "imagen_residuos_demo.jpg") -> dict[str, Any]:
    """Describe objetos de una imagen demo de residuos, sin deducir su contenedor."""
    datos, mime = _leer_asset(nombre_imagen)
    # Reutiliza la configuración de credenciales y el modelo del RAG.
    from google.genai import types
    from src.gemini_client import crear_cliente_gemini

    cliente = crear_cliente_gemini()
    respuesta = cliente.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=[
            types.Part.from_bytes(data=datos, mime_type=mime),
            types.Part.from_text(text=(
                "Describe en 2-3 frases en español solo los objetos y residuos "
                "observables. Indica incertidumbre cuando proceda. "
                "No afirmes a qué contenedor pertenecen sin consultar las fuentes."
            )),
        ],
    )
    descripcion = (getattr(respuesta, "text", None) or "").strip()
    if not descripcion:
        raise RuntimeError("Gemini no devolvió una descripción")
    return {
        "descripcion": descripcion,
        "imagen": nombre_imagen,
        "origen": "gemini_multimodal",
    }
