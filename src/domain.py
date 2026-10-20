"""Reglas geográficas conservadoras para el dominio de Madrid.

La lista representa excepciones reconocidas, no todos los municipios.
No es un geocodificador ni interpreta cualquier topónimo como ciudad.
"""

import re

from src.text_utils import normalizar_texto


CIUDADES_FUERA_DE_MADRID = (
    "barcelona",
    "bilbao",
    "sevilla",
    "valencia",
    "zaragoza",
    "malaga",
    "valladolid",
    "toledo",
    "alicante",
    "murcia",
)


def _es_otra_ciudad(pregunta: str) -> bool:
    """Detecta referencias explícitas inequívocas a otras ciudades."""
    normalizada = normalizar_texto(pregunta)
    ciudades = "|".join(CIUDADES_FUERA_DE_MADRID)
    return bool(re.search(
        rf"\b(?:en|para|ciudad\s+de)\s+({ciudades})\b",
        normalizada,
    ))
