"""Orquesta el RAG de ReciclaTIA y las consultas estructuradas.

El retrieval conserva el acceso a ChromaDB; la generación mantiene
su propio contrato con Gemini. Este módulo no depende de la UI ni de
los agentes LangGraph y no modifica el índice vectorial.
"""

import re
import unicodedata
from time import perf_counter
from typing import Any

from config import GENERATION_MODEL, TOP_K
from src.generate import (
    MENSAJE_ABSTENCION,
    _es_otra_ciudad,
    generar_respuesta,
)
from src.retrieve import (
    MAX_PREGUNTA_CARACTERES,
    buscar_chunks_relevantes,
    consultar_catalogo_residuos,
    consultar_paradas_moviles,
)


PATRON_MOVIL = re.compile(r"\bpuntos?\s+limpios?\s+movil(?:es)?\b")
PATRON_HORARIO = re.compile(
    r"\b(?:horarios?|horas?|dias?|cuando|pasa|visita)\b"
)
PATRON_24H = re.compile(r"\b24\s*(?:h|horas?)\b")
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


def _validar_top_k(top_k: int) -> int:
    """Impide pedir más chunks de los que procesa la generación."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k debe ser un número entero.")
    if not 1 <= top_k <= TOP_K:
        raise ValueError(
            f"top_k debe estar entre 1 y {TOP_K} para este RAG."
        )
    return top_k


def _validar_pregunta(pregunta: str) -> str:
    """Valida la entrada también en las vías sin retrieval semántico."""
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta debe ser un texto no vacío.")
    limpia = pregunta.strip()
    if len(limpia) > MAX_PREGUNTA_CARACTERES:
        raise ValueError(
            "La pregunta no puede superar "
            f"{MAX_PREGUNTA_CARACTERES} caracteres."
        )
    return limpia


def _normalizar(texto: str) -> str:
    """Normaliza texto solo para detectar intención y ubicación."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return " ".join(texto.split())


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


def _resultado_abstencion(texto: str, motivo: str) -> dict:
    """Crea una aclaración sin fuentes inventadas."""
    return {
        "respuesta": texto,
        "abstencion": True,
        "motivo": motivo,
        "fuentes": [],
    }


def _resultado_catalogo() -> tuple[dict, list[dict]]:
    """Resume destinos documentados, sin llamarlos tipos oficiales."""
    catalogo = consultar_catalogo_residuos()
    destinos = catalogo["destinos"]
    fuente = catalogo["source"]
    ruta = catalogo["path"]
    if not destinos or not ruta:
        return _resultado_abstencion(
            "No hay relaciones válidas en el catálogo disponible.",
            "sin_contexto",
        ), []

    nombres = [item["destino"] for item in destinos]
    total = catalogo["total_relaciones"]
    cantidad = catalogo["total_destinos"]
    texto = (
        f"El archivo {fuente} del corpus municipal contiene {total} "
        f"relaciones entre residuos y lugares de depósito, agrupadas "
        f"en {cantidad} destinos diferentes: "
        + "; ".join(nombres)
        + ". [C1]\n"
        "Estos destinos no constituyen una clasificación exhaustiva "
        "de todos los tipos de residuos reconocidos por el Ayuntamiento."
    )
    evidencia = (
        f"Resumen derivado de {fuente}; "
        f"{total} relaciones y {cantidad} destinos.\n"
        "Destinos documentados:\n"
        + "\n".join(
            f"- {item['destino']}: {item['cantidad']} relaciones"
            for item in destinos
        )
    )
    chunks = [{
        "text": evidencia,
        "metadata": {
            "source": fuente,
            "path": ruta,
            "document_type": "resumen_catalogo_derivado",
            "derived": True,
            "total_relaciones": total,
            "total_destinos": cantidad,
        },
    }]
    return {
        "respuesta": texto,
        "abstencion": False,
        "motivo": None,
        "fuentes": [{
            "chunk_id": "C1", "source": fuente, "path": ruta,
        }],
    }, chunks


def _resultado_horario_movil(
    pregunta: str, top_k: int
) -> tuple[dict, list[dict]]:
    """Atribuye cada franja a su dirección y día documentados."""
    distrito, direccion = _extraer_ubicacion_movil(pregunta)
    if not distrito and not direccion:
        return _resultado_abstencion(
            "¿En qué distrito o calle de Madrid quieres consultar "
            "el horario del punto limpio móvil?",
            "ambiguedad",
        ), []

    dia = _extraer_dia(pregunta)
    paradas = consultar_paradas_moviles(
        distrito=distrito, direccion=direccion, dia=dia
    )
    if not paradas:
        return _resultado_abstencion(
            "No encuentro una parada móvil con esos criterios "
            "en los registros disponibles de Madrid.",
            "sin_contexto",
        ), []

    plural = bool(re.search(r"\bhorarios\b|\bpuntos\s+limpios", pregunta))
    if distrito and not direccion and len(paradas) > 1 and not plural:
        return _resultado_abstencion(
            f"Hay {len(paradas)} registros de paradas móviles en "
            f"{distrito}. ¿De qué calle o parada quieres el horario?",
            "ambiguedad",
        ), []

    mostradas = paradas[:top_k]
    lineas = []
    chunks = []
    fuentes = []
    for indice, parada in enumerate(mostradas, start=1):
        identificador = f"C{indice}"
        lineas.append(
            f"{parada['direccion']} ({parada['distrito']}): "
            f"{parada['dia']}, {parada['horario']} "
            f"[{identificador}]"
        )
        chunks.append({
            "text": parada["text"],
            "metadata": dict(parada["metadata"]),
        })
        fuentes.append({
            "chunk_id": identificador,
            "source": parada["source"],
            "path": parada["path"],
        })

    intro = "Horarios documentados por parada (no en tiempo real):\n"
    if len(paradas) > len(mostradas):
        intro = (
            f"Hay {len(paradas)} registros coincidentes. "
            f"Se muestran {len(mostradas)}; indica una calle o día "
            "para consultar los demás.\n"
        )
    return {
        "respuesta": intro + "\n".join(lineas),
        "abstencion": False,
        "motivo": None,
        "fuentes": fuentes,
    }, chunks


def _empaquetar(
    pregunta: str,
    top_k: int,
    resultado: dict,
    chunks: list[dict],
    inicio: float,
    fin_retrieval: float,
    fin_generacion: float,
) -> dict[str, Any]:
    """Mantiene el contrato público y las métricas originales."""
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


def responder(
    pregunta: str,
    *,
    top_k: int = TOP_K,
    query_embedding: list[float] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Responde mediante retrieval semántico o consulta estructurada.

    Mantiene los parámetros y el contrato usados por los futuros
    agentes. Los errores técnicos se propagan como excepciones.
    Las aclaraciones y resúmenes deterministas no consumen Gemini.
    """
    top_k = _validar_top_k(top_k)
    limpia = _validar_pregunta(pregunta)
    normalizada = _normalizar(limpia)
    inicio = perf_counter()

    es_catalogo = _es_consulta_catalogo(normalizada)
    es_horario_movil = _es_consulta_horarios_moviles(normalizada)
    if (es_catalogo or es_horario_movil) and _es_otra_ciudad(limpia):
        resultado = _resultado_abstencion(
            MENSAJE_ABSTENCION, "fuera_de_dominio"
        )
        chunks = []
        fin_retrieval = perf_counter()
    elif es_catalogo:
        resultado, chunks = _resultado_catalogo()
        fin_retrieval = perf_counter()
    elif es_horario_movil:
        resultado, chunks = _resultado_horario_movil(
            normalizada, top_k
        )
        fin_retrieval = perf_counter()
    else:
        chunks = buscar_chunks_relevantes(
            pregunta,
            top_k=top_k,
            query_embedding=query_embedding,
        )
        fin_retrieval = perf_counter()
        resultado = generar_respuesta(pregunta, chunks, client=client)

    fin_generacion = perf_counter()
    return _empaquetar(
        limpia, top_k, resultado, chunks,
        inicio, fin_retrieval, fin_generacion,
    )
