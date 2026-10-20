"""Detección conservadora de intenciones del asistente RAG.

No accede a Gemini, a ChromaDB ni al estado de los agentes.
"""

import re

from src.domain import CIUDADES_FUERA_DE_MADRID, _es_otra_ciudad
from src.text_utils import normalizar_texto as _normalizar


PATRON_MOVIL = re.compile(r"\bpuntos?\s+limpios?\s+movil(?:es)?\b")
PATRON_HORARIO = re.compile(
    r"\b(?:horarios?|horas?|dias?|cuando|pasa|visita)\b"
)
PATRON_24H = re.compile(r"\b24\s*(?:h|horas?)\b")
PATRON_DISPOSITIVO_TELEFONICO = re.compile(
    r"\b(?:movil(?:es)?|telefonos?|smartphones?|celular(?:es)?)\b"
)
PATRON_REPARACION = re.compile(
    r"\b(?:reparac(?:ion|iones)|repar(?:ar|an|a)|"
    r"arregl(?:ar|an|o)|servicios?\s+tecnicos?)\b"
)
PATRON_BUSQUEDA_SERVICIO = re.compile(
    r"\b(?:donde|como|cual(?:es)?|hay|cerca|busco|necesito|"
    r"taller(?:es)?|tiendas?|servicios?|tecnicos?|precio|"
    r"coste|cuanto|cuesta)\b"
)
PATRON_CONTEXTO_RESIDUOS = re.compile(
    r"\b(?:recicl\w*|residuos?|basura|contenedores?|"
    r"puntos?\s+limpios?|depositar|desech\w*|"
    r"tirar|tiro|reutiliz\w*|recogida|deshacerme)\b"
)
PATRON_DIRECCION = re.compile(
    r"\b(?:calle|avenida|avda|plaza|paseo|ronda|camino)\s+"
    r"[a-z0-9\s.,\-/]+"
)
DIAS = {
    "lunes": "lunes",
    "martes": "martes",
    "miercoles": "miercoles",
    "jueves": "jueves",
    "viernes": "viernes",
    "sabado": "sabado",
    "sabados": "sabado",
    "domingo": "domingo",
    "domingos": "domingo",
}
PATRON_DIA = re.compile(
    r"\b(?:" + "|".join(DIAS) + r")\b"
)


def _es_consulta_catalogo(pregunta: str) -> bool:
    """Selecciona solo peticiones explícitas de panorama general."""
    if re.search(
        r"\b(?:donde|como|tirar|tiro|reciclo|contenedor|"
        r"punto limpio|llevar|depositar)\b",
        pregunta,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:que|cuales)\s+(?:son\s+)?(?:los\s+)?"
            r"tipos\s+de\s+residuos\b",
            pregunta,
        )
        or re.search(
            r"\b(?:catalogo|lista(?:do)?|categorias|clasificacion)\b"
            r".*\b(?:residuos?|destinos?)\b",
            pregunta,
        )
        or re.search(
            r"\b(?:residuos?|destinos?)\b.*"
            r"\b(?:catalogo|lista(?:do)?)\b",
            pregunta,
        )
    )


def _es_consulta_reparacion_telefonos(pregunta: str) -> bool:
    """Reconoce solicitudes explícitas de reparación, no de reciclaje.

    «Móvil» puede designar un teléfono o un punto limpio itinerante.
    Por eso se exige contexto de reparación de dispositivos y se evita
    interceptar preguntas que mencionen la gestión de residuos.
    """
    if PATRON_CONTEXTO_RESIDUOS.search(pregunta):
        return False
    return bool(
        PATRON_DISPOSITIVO_TELEFONICO.search(pregunta)
        and PATRON_REPARACION.search(pregunta)
        and PATRON_BUSQUEDA_SERVICIO.search(pregunta)
    )


def _es_consulta_horarios_moviles(pregunta: str) -> bool:
    """Excluye la modalidad 24h y preguntas conceptuales."""
    return bool(
        PATRON_MOVIL.search(pregunta)
        and PATRON_HORARIO.search(pregunta)
        and not PATRON_24H.search(pregunta)
    )


def _extraer_dia(pregunta: str) -> str | None:
    """Detecta un único día sin elegir entre fechas distintas."""
    dias = {DIAS[item.group()] for item in PATRON_DIA.finditer(pregunta)}
    return next(iter(dias)) if len(dias) == 1 else None


def _limpiar_ubicacion(texto: str) -> str:
    """Quita conectores y fechas posteriores a una ubicación."""
    texto = re.split(
        r"\s+(?:el|los|un|una|para)\s+(?:" + "|".join(DIAS)
        + r")\b",
        texto,
        maxsplit=1,
    )[0]
    return texto.strip(" .,:;¿?¡!")


def _extraer_ubicacion_movil(pregunta: str) -> tuple[str | None, str | None]:
    """Extrae una calle o distrito explícito sin adivinar paradas."""
    direccion = PATRON_DIRECCION.search(pregunta)
    if direccion:
        calle = _limpiar_ubicacion(direccion.group())
        return None, calle or None

    distrito = re.search(
        r"\bdistrito\s+(?:de\s+)?([^?.,;]+)", pregunta
    )
    if distrito:
        nombre = _limpiar_ubicacion(distrito.group(1))
        if nombre and nombre != "madrid":
            return nombre, None
        return None, None

    movil = PATRON_MOVIL.search(pregunta)
    if movil is None:
        return None, None
    cola = pregunta[movil.end():]
    lugar = re.match(
        r"\s+(?:de|del|en|por)\s+(?:el\s+|la\s+)?"
        r"([^?.,;]+)",
        cola,
    )
    if lugar:
        nombre = _limpiar_ubicacion(lugar.group(1))
        if nombre not in {"", "madrid", "ciudad de madrid"}:
            if not PATRON_DIA.fullmatch(nombre):
                return nombre, None
    return None, None


def _es_fuera_de_dominio_estructurado(
    pregunta: str,
    *,
    es_horario_movil: bool,
) -> bool:
    """Detecta ciudades externas antes de consultar las paradas de Madrid.

    Reutiliza la lista de ciudades de generación, sin duplicarla.
    También cubre «punto limpio móvil de Barcelona», que no coincide
    con el patrón genérico de «en Barcelona». No interpreta calles
    como distritos ni afecta a las consultas semánticas ordinarias.
    """
    if _es_otra_ciudad(pregunta):
        return True
    if not es_horario_movil:
        return False

    distrito, direccion = _extraer_ubicacion_movil(pregunta)
    return bool(
        distrito
        and not direccion
        and _normalizar(distrito) in CIUDADES_FUERA_DE_MADRID
    )


