"""Filtro textual del ámbito Madrid ciudad, sin geocodificación.

Reconoce referencias explícitas a localidades por su posición en la frase;
los distritos y las direcciones de Madrid no se consideran ciudades externas.
No pretende identificar todas las entidades geográficas del mundo.
"""

import re
from typing import Callable

from src.text_utils import normalizar_distrito, normalizar_texto

# Divisiones administrativas de Madrid ciudad, no ciudades prohibidas.
# Permiten distinguir «Salamanca» (distrito) de la ciudad homónima cuando
# la consulta no dice explícitamente «ciudad de Salamanca».
DISTRITOS_MADRID = frozenset({
    "arganzuela", "barajas", "carabanchel", "centro", "chamartin",
    "chamberi", "ciudad lineal", "fuencarral el pardo", "hortaleza",
    "latina", "moncloa aravaca", "moratalaz", "puente de vallecas",
    "retiro", "salamanca", "san blas canillejas", "tetuan", "usera",
    "vicalvaro", "villa de vallecas", "villaverde",
})

_CALLE = re.compile(
    r"^(?:la\s+|el\s+)?(?:calle|avda|avenida|plaza|paseo|ronda|camino)\b"
)
_DESTINO = re.compile(
    r"^(?:el\s+|la\s+|un\s+|una\s+)?"
    r"(?:contenedor|punto\s+limpio|reciclaje|residuo|papel|vidrio|"
    r"plastico|horario|distrito|barrio|casa|hogar|cocina|"
    r"papelera|bolsa|cubo|botella|fraccion|"
    r"supermercado|mercado)\b"
)
CIUDADES_FUERA_DE_MADRID = re.compile(
    r"\b(?:en|ciudad\s+de|municipio\s+de|localidad\s+de|"
    r"distrito\s+de)\s+"
    r"(?P<lugar>[a-zñ][a-zñ\s-]{1,60}?)"
    r"(?=\s*(?:[?.,;!]|$|\b(?:el|los|para|con|del)\s+(?:lunes|"
    r"martes|miercoles|jueves|viernes|sabado|domingo)\b))"
)
_PATRON_PARA = re.compile(
    r"\bpara\s+(?P<lugar>[a-zñ][a-zñ\s-]{1,60}?)"
    r"(?=\s*[?.,;!]|$)"
)
_PATRON_MOVIL = re.compile(
    r"\bpuntos?\s+limpios?\s+movil(?:es)?\s+"
    r"(?:de|del|para)\s+(?P<lugar>[a-zñ][a-zñ\s-]{1,60}?)"
    r"(?=\s*[?.,;!]|$)"
)


def _extraer_lugar(pregunta: str) -> tuple[str | None, bool]:
    """Extrae un topónimo explícito y si se pidió una ciudad concreta."""
    texto = normalizar_texto(pregunta)
    especifica_ciudad = bool(re.search(
        r"\b(?:ciudad|municipio|localidad)\s+de\b", texto
    ))
    coincidencias = list(CIUDADES_FUERA_DE_MADRID.finditer(texto))
    if not coincidencias:
        movil = _PATRON_MOVIL.search(texto)
        coincidencias = [movil] if movil else []
    if not coincidencias:
        para = _PATRON_PARA.search(texto)
        coincidencias = [para] if para else []
    for coincidencia in reversed(coincidencias):
        lugar = coincidencia.group("lugar").strip()
        lugar = re.sub(r"^(?:la|el|los)\s+", "", lugar)
        lugar = re.sub(
            r"^(?:ciudad|municipio|localidad|distrito)\s+de\s+",
            "", lugar,
        )
        if not lugar or _CALLE.search(lugar) or _DESTINO.search(lugar):
            continue
        if re.search(
            r"\b(?:recicl\w*|arregl\w*|repar\w*|su|mi|un|una)\b",
            lugar,
        ):
            continue
        return lugar, especifica_ciudad
    return None, False


def evaluar_ambito(
    pregunta: str,
    *,
    resolver: Callable[[str], dict | None] | None = None,
) -> str:
    """Madrid por defecto; rechaza localidades externas explícitas.

    Los distritos de Madrid tienen preferencia sobre nombres homónimos
    de municipios. No se consulta ningún servicio o coordenada.
    """
    lugar, explicita = _extraer_lugar(pregunta)
    if lugar is None:
        return "madrid"
    normalizado = normalizar_texto(lugar)
    if normalizado in {"madrid", "madrid ciudad", "ciudad de madrid"}:
        return "madrid"
    if not explicita and normalizar_distrito(lugar) in DISTRITOS_MADRID:
        return "madrid"
    # Compatibilidad con pruebas existentes que inyectan un resolutor
    # simulado. En producción resolver=None y no hay ninguna consulta.
    if resolver is not None:
        datos = resolver(lugar)
        if datos is None:
            return "ambiguedad"
        ciudad = normalizar_texto(str(
            datos.get("city") or datos.get("town")
            or datos.get("village") or datos.get("municipality") or ""
        ))
        if not ciudad:
            return "ambiguedad"
        return "madrid" if ciudad == "madrid" else "fuera_de_dominio"
    return "fuera_de_dominio"


def _es_otra_ciudad(pregunta: str) -> bool:
    """Compatibilidad con los llamadores previos a la centralización."""
    return evaluar_ambito(pregunta) == "fuera_de_dominio"
