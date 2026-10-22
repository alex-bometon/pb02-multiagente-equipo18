"""API oficial de puntos limpios FIJOS de Madrid; fallback local trazable.

El endpoint ArcGIS está documentado. Ejecutar el smoke test HTTP real antes
 de afirmar que la integración externa funciona en producción.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

import config

CAMPOS = (
    "OBJECTID_1", "TIPO_DATO", "NOMBRE_DIS", "Direccion", "Ubicacion",
    "ABIERTO", "CERRADO", "Residuos", "PAGINA_WEB", "TELEFONO",
)
PAGINA = 100
MAX_PAGINAS = 20


def _normalizar(texto: str) -> str:
    simple = unicodedata.normalize("NFD", texto.casefold())
    return re.sub(r"[^a-z0-9]+", " ", "".join(
        c for c in simple if unicodedata.category(c) != "Mn"
    )).strip()


def _validar(distrito: str, limite: int) -> tuple[str, int]:
    if not isinstance(distrito, str) or len(distrito) > 80:
        raise ValueError("distrito debe ser texto de hasta 80 caracteres")
    if isinstance(limite, bool) or not isinstance(limite, int):
        raise ValueError("limite debe ser un entero")
    if not 1 <= limite <= config.TOOLS_API_MAX_RESULTADOS:
        raise ValueError("limite fuera de rango")
    return _normalizar(distrito), limite


def _item(atributos: dict) -> dict[str, Any]:
    """Conserva los textos de ArcGIS (ABIERTO y CERRADO NO significan 'ahora')."""
    def texto(k: str) -> str | None:
        val = atributos.get(k)
        return val.strip() if isinstance(val, str) and val.strip() else None

    oid = atributos.get("OBJECTID_1")
    return {
        "id": oid if isinstance(oid, int) and not isinstance(oid, bool) else None,
        "distrito": texto("NOMBRE_DIS"),
        "direccion": texto("Direccion"),
        "ubicacion": texto("Ubicacion"),
        "abierto_texto": texto("ABIERTO"),
        "cerrado_texto": texto("CERRADO"),
        "residuos_admitidos": texto("Residuos"),
        "web_municipal": texto("PAGINA_WEB"),
        "telefono": texto("TELEFONO"),
        "horario_csv": texto("HORARIO_CSV"),
    }


def _leer_api() -> list[dict]:
    registros: list[dict] = []
    offset = 0
    for _ in range(MAX_PAGINAS):
        params = {
            "f": "json",
            "where": "TIPO_DATO = 'Fijo'",  # Nunca interpolar entrada del usuario.
            "outFields": ",".join(CAMPOS),
            "returnGeometry": "false",
            "orderByFields": "OBJECTID_1 ASC",
            "resultOffset": offset,
            "resultRecordCount": PAGINA,
        }
        respuesta = requests.get(
            config.TOOLS_PUNTOS_FIJOS_QUERY_URL,
            params=params,
            timeout=config.TOOLS_HTTP_TIMEOUT_SECONDS,
        )
        respuesta.raise_for_status()
        body = respuesta.json()
        if not isinstance(body, dict) or "error" in body:
            raise ValueError("Respuesta ArcGIS con error o estructura inválida")
        features = body.get("features")
        if not isinstance(features, list):
            raise ValueError("Respuesta ArcGIS sin lista features")
        if any(not isinstance(f, dict) or not isinstance(f.get("attributes"), dict)
               for f in features):
            raise ValueError("La lista features contiene elementos inválidos")
        for feature in features:
            attr = feature["attributes"]
            if _normalizar(str(attr.get("TIPO_DATO") or "")) == "fijo":
                registros.append(attr)
        offset += len(features)
        if not features or (not body.get("exceededTransferLimit") and len(features) < PAGINA):
            return registros
    raise ValueError("Demasiadas páginas al consultar ArcGIS")


def _leer_fallback() -> tuple[list[dict], dict]:
    path: Path = config.TOOLS_FALLBACK_PATH
    documento = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documento, dict) or documento.get("modalidad") != "fijo":
        raise ValueError("Fallback inválido: modalidad")
    if not documento.get("fuente_original") or not documento.get("fecha_captura"):
        raise ValueError("Fallback sin procedencia o fecha de captura")
    filas = documento.get("records")
    if not isinstance(filas, list) or not filas:
        raise ValueError("Fallback sin registros")
    if any(not isinstance(item, dict) or
           _normalizar(str(item.get("TIPO_DATO") or "")) != "fijo" or
           not item.get("Direccion") or not item.get("NOMBRE_DIS")
           for item in filas):
        raise ValueError("Fallback contiene filas inválidas")
    return filas, documento


def API_consultar_puntos_limpios_fijos(distrito: str = "", limite: int = 5) -> dict[str, Any]:
    """Consulta datos oficiales de puntos limpios fijos de Madrid por distrito.

    No informa de apertura en tiempo real, ni incluye móviles o proximidad.
    """
    nombre, limite = _validar(distrito, limite)
    fecha = datetime.now(ZoneInfo("Europe/Madrid")).isoformat(timespec="seconds")
    origen = "api_oficial"
    aviso = "Datos municipales publicados; no implican apertura en este instante."
    fecha_datos: str | None = None
    cobertura_parcial = False
    try:
        registros = _leer_api()
    except (requests.RequestException, ValueError, TypeError, KeyError) as original:
        try:
            registros, metadata = _leer_fallback()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as err:
            raise RuntimeError(
                "La API municipal no está disponible y no existe fallback local válido"
            ) from err
        origen = "fallback_local"
        fecha_datos = metadata.get("fecha_datos")
        cobertura_parcial = bool(metadata.get("cobertura_parcial", False))
        aviso = (
            "ATENCIÓN: copia local histórica, no actualizada en tiempo real. "
            "La cobertura puede ser parcial; comprobar antes de desplazarse."
        )

    puntos = [
        _item(fila) for fila in registros
        if not nombre or _normalizar(str(fila.get("NOMBRE_DIS") or "")) == nombre
    ]
    puntos.sort(key=lambda x: ((x["distrito"] or ""), (x["direccion"] or ""), x["id"] or 0))
    total = len(puntos)
    return {
        "origen": origen,
        "url_fuente": config.TOOLS_PUNTOS_FIJOS_FUENTE_URL,
        "fecha_consulta": fecha,
        "fecha_datos": fecha_datos,
        "modalidad": "fijo",
        "distrito_filtro": distrito.strip(),
        "total_coincidencias": total,
        "recortado": total > limite,
        "cobertura_parcial": cobertura_parcial,
        "puntos": puntos[:limite],
        "aviso": aviso,
    }
