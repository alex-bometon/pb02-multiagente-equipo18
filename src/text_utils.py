"""Normalización de texto geográfico para recuperación y orquestación.

No transforma el texto original mostrado al usuario ni los metadatos
almacenados en ChromaDB.
"""

import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """Normaliza tildes, mayúsculas y espacios solo para detección."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return " ".join(texto.split())


def normalizar_distrito(nombre: str) -> str:
    """Unifica tildes, mayúsculas y separadores de los distritos."""
    normalizado = normalizar_texto(nombre)
    return re.sub(r"[\s-]+", " ", normalizado).strip()
