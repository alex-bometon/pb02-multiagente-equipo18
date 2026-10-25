
"""Consultas geográficas de puntos limpios de Madrid."""

from math import asin, cos, isfinite, radians, sin, sqrt

import requests

from langchain_core.tools import tool

from src.retrieve import _abrir_coleccion
from src.retrieve_structured import consultar_puntos_limpios_locales

from config import (
    API_PUNTOS_LIMPIOS_URL,
    API_TIMEOUT_SECONDS,
    API_RESULT_RECORD_COUNT,
    GEO_DEFAULT_RADIUS_METERS,
    GEO_MIN_RADIUS_METERS,
    GEO_MAX_RADIUS_METERS,
    GEO_MAX_RESULTS,
)

TIPOS_PUNTOS_LIMPIOS = {
    "todos": None,
    "fijo": "Fijo",
    "movil": "Móvil",
    "proximidad": "Proximidad",
}


def _consultar_api_puntos_limpios(
    latitud: float,
    longitud: float,
    radio_metros: int = GEO_DEFAULT_RADIUS_METERS,
    tipo: str = "todos",
) -> list[dict]:
    """
    Consulta puntos limpios dentro de un radio geográfico.

    Devuelve los registros originales de ArcGIS, sin
    transformarlos ni calcular distancias.

    Raises:
        ValueError: Parámetros de búsqueda inválidos.
        RuntimeError: Error HTTP o respuesta inválida.
    """

    # Validación de coordenadas.
    for nombre, valor, minimo, maximo in (
        ("latitud", latitud, -90, 90),
        ("longitud", longitud, -180, 180),
    ):
        if (
            isinstance(valor, bool)
            or not isinstance(valor, (int, float))
            or not isfinite(valor)
            or not minimo <= valor <= maximo
        ):
            raise ValueError(f"{nombre} no es una coordenada válida.")

    # Validación del radio.
    if (
        isinstance(radio_metros, bool)
        or not isinstance(radio_metros, int)
        or not GEO_MIN_RADIUS_METERS <= radio_metros <= GEO_MAX_RADIUS_METERS
    ):
        raise ValueError(
            f"El radio debe estar entre {GEO_MIN_RADIUS_METERS} "
            f"y {GEO_MAX_RADIUS_METERS} metros."
        )

    # Validación del tipo.
    if not isinstance(tipo, str):
        raise ValueError("Tipo de punto limpio inválido.")

    tipo = tipo.strip().lower()

    if tipo not in TIPOS_PUNTOS_LIMPIOS:
        raise ValueError("Tipo de punto limpio no reconocido.")

    tipo_api = TIPOS_PUNTOS_LIMPIOS[tipo]

    filtro = (
        "1=1"
        if tipo_api is None
        else f"TIPO_DATO = '{tipo_api}'"
    )

    # Parámetros de ArcGIS REST.
    parametros = {
        "f": "json",
        "where": filtro,
        "geometry": f"{longitud},{latitud}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "distance": radio_metros,
        "units": "esriSRUnit_Meter",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "resultRecordCount": API_RESULT_RECORD_COUNT,
    }

    try:
        respuesta = requests.get(
            API_PUNTOS_LIMPIOS_URL,
            params=parametros,
            timeout=API_TIMEOUT_SECONDS,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()

    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            "No se pudo consultar la API de puntos limpios."
        ) from error

    if not isinstance(datos, dict):
        raise RuntimeError("La API devolvió un formato inesperado.")

    if "error" in datos:
        raise RuntimeError(
            f"Error del servicio ArcGIS: {datos['error']}"
        )

    registros = datos.get("features")

    if not isinstance(registros, list):
        raise RuntimeError("La API no devolvió una lista de registros.")

    if datos.get("exceededTransferLimit"):
        raise RuntimeError(
            "La consulta contiene más registros de los recibidos."
        )

    return registros



def _normalizar_registro_api(registro: dict) -> dict | None:
    """
    Transforma un registro ArcGIS en un formato común.

    Devuelve None si faltan datos esenciales o las
    coordenadas son inválidas.
    """

    if not isinstance(registro, dict):
        return None

    atributos = registro.get("attributes")
    geometria = registro.get("geometry")

    if not isinstance(atributos, dict):
        return None

    if not isinstance(geometria, dict):
        geometria = {}

    # Preferimos las coordenadas de la geometría GeoJSON-like
    # devuelta por ArcGIS en WGS84 (EPSG:4326).
    latitud = geometria.get("y")
    longitud = geometria.get("x")

    if latitud is None:
        latitud = atributos.get("LATITUD")

    if longitud is None:
        longitud = atributos.get("LONGITUD")

    # Comprobar coordenadas numéricas y finitas.
    if (
        isinstance(latitud, bool)
        or not isinstance(latitud, (int, float))
        or not isfinite(latitud)
        or not -90 <= latitud <= 90
        or isinstance(longitud, bool)
        or not isinstance(longitud, (int, float))
        or not isfinite(longitud)
        or not -180 <= longitud <= 180
    ):
        return None

    # Relación entre los campos originales y nuestro contrato.
    campos = {
        "tipo": "TIPO_DATO",
        "direccion": "Direccion",
        "distrito": "NOMBRE_DIS",
        "horario_apertura": "ABIERTO",
        "horario_cierre": "CERRADO",
        "dia_semana": "DIA_SEMANA",
        "turno": "TURNO",
        "residuos_url": "Residuos",
    }

    datos = {}

    for campo_salida, campo_api in campos.items():
        valor = atributos.get(campo_api)

        datos[campo_salida] = (
            " ".join(valor.split())
            if isinstance(valor, str) and valor.strip()
            else None
        )

    # Sin tipo o dirección no podemos identificar el punto.
    if not datos["tipo"] or not datos["direccion"]:
        return None

    return {
        "id": atributos.get("OBJECTID_1"),
        **datos,
        "latitud": float(latitud),
        "longitud": float(longitud),
        "fuente": "api",
    }



def _calcular_distancia_metros(
    latitud_origen: float,
    longitud_origen: float,
    latitud_destino: float,
    longitud_destino: float,
) -> float:
    """
    Calcula la distancia geográfica entre dos coordenadas
    utilizando la fórmula de Haversine.

    Las coordenadas deben estar previamente validadas.
    Devuelve la distancia en metros.
    """

    radio_tierra = 6_371_000

    lat1 = radians(latitud_origen)
    lon1 = radians(longitud_origen)
    lat2 = radians(latitud_destino)
    lon2 = radians(longitud_destino)

    diferencia_latitud = lat2 - lat1
    diferencia_longitud = lon2 - lon1

    a = (
        sin(diferencia_latitud / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin(diferencia_longitud / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * radio_tierra * asin(sqrt(a))



def _preparar_puntos_cercanos(
    registros: list[dict],
    latitud_origen: float,
    longitud_origen: float,
) -> list[dict]:
    """
    Normaliza los registros de la API, calcula su distancia
    al origen y los ordena de menor a mayor proximidad.

    Descarta los registros que no pueden normalizarse.
    """

    puntos = []

    for registro in registros:

        punto = _normalizar_registro_api(registro)

        if punto is None:
            continue

        distancia = _calcular_distancia_metros(
            latitud_origen,
            longitud_origen,
            punto["latitud"],
            punto["longitud"],
        )

        puntos.append({
            **punto,
            "distancia_metros": round(distancia, 2),
        })

    return sorted(
        puntos,
        key=lambda punto: punto["distancia_metros"],
    )



@tool
def buscar_puntos_limpios_cercanos(
    latitud: float,
    longitud: float,
    radio_metros: int = GEO_DEFAULT_RADIUS_METERS,
    tipo: str = "todos",
) -> dict:
    """
    Busca puntos limpios cercanos en Madrid.

    Consulta primero la API geográfica municipal.
    Si falla, utiliza los registros locales de ChromaDB.

    Devuelve los resultados más cercanos hasta el límite
    configurado e identifica la fuente utilizada.
    """

    tipo_normalizado = tipo.strip().lower() if isinstance(tipo, str) else tipo

    try:
        # Fuente principal: API municipal.
        registros = _consultar_api_puntos_limpios(
            latitud=latitud,
            longitud=longitud,
            radio_metros=radio_metros,
            tipo=tipo,
        )

    except RuntimeError:
        # Solo se activa ante fallos técnicos de la API.
        # Los errores de validación siguen propagándose.

        registros_locales = consultar_puntos_limpios_locales(
            abrir_coleccion=_abrir_coleccion
        )

        equivalencias = {
            "todos": {"Fijo", "Móvil", "Móvil 24h", "Proximidad"},
            "fijo": {"Fijo"},
            "movil": {"Móvil", "Móvil 24h"},
            "proximidad": {"Proximidad"},
        }

        tipos_permitidos = equivalencias[tipo_normalizado]

        puntos = []

        for punto in registros_locales:

            if punto["tipo"] not in tipos_permitidos:
                continue

            distancia = _calcular_distancia_metros(
                latitud,
                longitud,
                punto["latitud"],
                punto["longitud"],
            )

            if distancia > radio_metros:
                continue

            puntos.append({
                **punto,
                "distancia_metros": round(distancia, 2),
            })

        puntos.sort(
            key=lambda punto: punto["distancia_metros"]
        )

        fuente = "corpus_local"

    else:
        # La API respondió correctamente.
        # Los errores de procesamiento no activan el fallback.
        puntos = _preparar_puntos_cercanos(
            registros,
            latitud,
            longitud,
        )

        fuente = "api"

    # Comprobación final del radio, también para la API.
    cercanos = [
        punto for punto in puntos
        if punto["distancia_metros"] <= radio_metros
    ]

    return {
        "fuente": fuente,
        "tipo_solicitado": tipo_normalizado,
        "radio_metros": radio_metros,
        "total_encontrados": len(cercanos),
        "puntos": cercanos[:GEO_MAX_RESULTS],
        # Candidatos adicionales para priorizar horarios sin sacrificar
        # el contrato original de 10 puntos más cercanos para la UI.
        "puntos_candidatos": cercanos[:GEO_MAX_RESULTS * 6],
        "aviso": (
            "Datos del corpus local indexado; "
            "pueden estar desactualizados."
            if fuente == "corpus_local"
            else None
        ),
    }

