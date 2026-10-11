"""Tool: API_consultar_calidad_aire — HTTP open data Madrid + fallback local."""

import json

import requests
from langchain_core.tools import tool

import config


def _ultimo_valor_validado(record: dict) -> tuple[str, str]:
    ultimo = ("?", "?")
    for h in range(1, 25):
        clave_h = f"H{h:02d}"
        clave_v = f"V{h:02d}"
        if record.get(clave_v) == "V" and record.get(clave_h) not in (None, ""):
            ultimo = (clave_h, str(record.get(clave_h)))
    return ultimo


def _resumir_records(items: list, limite: int, fuente: str) -> str:
    if not items:
        return f"No hay mediciones (fuente={fuente})."

    lineas = [f"Fuente: {fuente}", ""]
    for item in items[:limite]:
        estacion = item.get("ESTACION", "?")
        mag = str(item.get("MAGNITUD", "?"))
        nombre = config.MAGNITUDES.get(mag, f"magnitud_{mag}")
        hora, valor = _ultimo_valor_validado(item)
        fecha = f"{item.get('ANO', '?')}-{item.get('MES', '?')}-{item.get('DIA', '?')}"
        lineas.append(
            f"- estación {estacion} | {nombre} ({mag}) | {hora}={valor} | fecha {fecha}"
        )
    lineas.append("")
    lineas.append(
        "Nota: datos orientativos de la red; no inventes umbrales oficiales."
    )
    return "\n".join(lineas)


@tool
def API_consultar_calidad_aire(limite: int = 5) -> str:
    """Trae un resumen de mediciones de calidad del aire (Madrid). Si falla la red, usa fallback."""
    try:
        limite = int(limite)
    except (TypeError, ValueError):
        limite = 5
    limite = max(1, min(limite, 10))

    try:
        resp = requests.get(
            config.MADRID_AIRE_URL,
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("records") or []
        fuente = "api"
    except Exception:
        data = json.loads(config.AIRE_FALLBACK_PATH.read_text(encoding="utf-8"))
        items = data.get("records") or []
        fuente = "fallback"

    return _resumir_records(items, limite, fuente)
