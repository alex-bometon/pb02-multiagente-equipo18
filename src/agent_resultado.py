"""Síntesis geográfica basada en evidencias, sin nuevas llamadas al modelo.

Combina una clasificación validada por RAG, la API geográfica y la hora
obtenida mediante la herramienta autorizada. No certifica apertura real.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import isfinite
from typing import Any, Mapping

from src.agent_schedule import (
    _hora,
    _priorizar_abiertos,
    _tipo_normalizado,
    _tipos_admitidos,
    estado_horario,
)
from src.text_utils import normalizar_texto


_ESTADOS = {"dentro": 0, "desconocido": 1, "fuera": 2}


def _datos(estado: Mapping[str, Any], nombre: str) -> list[dict[str, Any]]:
    """Lee solo evidencia de herramientas ejecutadas por el grafo."""
    salida = []
    for evidencia in estado.get("evidencias", []):
        if not isinstance(evidencia, dict):
            continue
        if evidencia.get("herramienta") != nombre:
            continue
        datos = evidencia.get("datos")
        if isinstance(datos, dict):
            salida.append(datos)
    return salida


def _clasificacion(estado: Mapping[str, Any]) -> str | None:
    """Retiene el texto sustentado por ChromaDB y sus citas originales."""
    for evidencia in estado.get("evidencias", []):
        if not isinstance(evidencia, dict):
            continue
        if (evidencia.get("rol") != "residuos"
                or evidencia.get("herramienta") != "consultar_rag"):
            continue
        datos = evidencia.get("datos")
        if (isinstance(datos, dict)
                and datos.get("abstencion") is False
                and datos.get("fuentes")
                and isinstance(datos.get("respuesta"), str)
                and datos["respuesta"].strip()):
            return datos["respuesta"].strip()
    return None


def _candidatos(geografia: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Usa candidatos extendidos sin perder el contrato API original."""
    puntos = geografia.get("puntos_candidatos")
    if not isinstance(puntos, list):
        puntos = geografia.get("puntos")
    if not isinstance(puntos, list):
        return []
    return [p for p in puntos if isinstance(p, dict)]


def _agrupar_puntos(puntos: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Une registros del mismo lugar, que pueden representar días distintos."""
    grupos: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for punto in puntos:
        direccion = punto.get("direccion")
        tipo = punto.get("tipo")
        distancia = punto.get("distancia_metros")
        if (not isinstance(direccion, str) or not direccion.strip()
                or not isinstance(tipo, str) or not tipo.strip()
                or isinstance(distancia, bool)
                or not isinstance(distancia, (int, float))
                or not isfinite(distancia) or distancia < 0):
            continue
        lat = punto.get("latitud")
        lon = punto.get("longitud")
        ubicacion = (
            f"{lat:.5f},{lon:.5f}"
            if isinstance(lat, (float, int))
            and isinstance(lon, (float, int))
            else ""
        )
        clave = (
            normalizar_texto(direccion),
            normalizar_texto(tipo),
            ubicacion,
        )
        grupos[clave].append(punto)
    return list(grupos.values())


def _estado_grupo(grupo: list[dict[str, Any]], instante: datetime | None) -> str:
    if instante is None:
        return "desconocido"
    estados = [estado_horario(punto, instante) for punto in grupo]
    if "dentro" in estados:
        return "dentro"
    if "desconocido" in estados:
        return "desconocido"
    return "fuera"


def _resumir_horario(grupo: list[dict[str, Any]]) -> str:
    textos = []
    for punto in grupo:
        componentes = []
        for campo in ("dia_semana", "horario_apertura", "horario_cierre"):
            valor = punto.get(campo)
            if isinstance(valor, str) and valor.strip():
                limpio = valor.strip().rstrip(". ")
                if limpio not in componentes:
                    componentes.append(limpio)
        texto = ", ".join(componentes)
        if texto and texto not in textos:
            textos.append(texto)
    if not textos:
        return "No disponible en el registro"
    return "; ".join(textos[:2])


def _ficha(grupo: list[dict[str, Any]], instante: datetime | None) -> dict[str, Any]:
    menor = min(grupo, key=lambda p: float(p["distancia_metros"]))
    return {
        "direccion": menor["direccion"],
        "tipo": menor["tipo"],
        "distancia": float(menor["distancia_metros"]),
        "estado": _estado_grupo(grupo, instante),
        "horario": _resumir_horario(grupo),
    }


def _etiqueta(estado: str) -> str:
    etiquetas = {
        "dentro": "Dentro del horario publicado",
        "fuera": "Fuera del horario publicado",
        "desconocido": "Horario no comprobable",
    }
    return etiquetas[estado]


def respuesta_geografica_integrada(
    estado: Mapping[str, Any],
) -> tuple[str, list[str]] | None:
    """Devuelve UNA respuesta coherente y el plan académico para HITL.

    El LLM orquesta consultas ambiguas; las comprobaciones y el orden
    del resultado dependen de Python y de las herramientas ya ejecutadas.
    """
    resultados = _datos(estado, "buscar_puntos_limpios_cercanos")
    if not resultados:
        return None
    geografia = resultados[-1]
    if geografia.get("fuente") not in {"api", "corpus_local"}:
        return None

    requiere_residuo = estado.get("necesita_residuos") is True
    clasificacion = _clasificacion(estado) if requiere_residuo else None
    if requiere_residuo and clasificacion is None:
        # No se puede fingir aceptación si el RAG se abstuvo o falló.
        return None

    tipos = _tipos_admitidos(estado) if requiere_residuo else None
    if requiere_residuo and not tipos:
        # El RAG debe identificar modalidades explícitas; no asumir que
        # cualquier punto admite cualquier tipo de residuo.
        return None

    instante = _hora(estado)
    preferir_abiertos = _priorizar_abiertos(str(estado.get("pedido") or ""))
    grupos = _agrupar_puntos(_candidatos(geografia))
    fichas = []
    for grupo in grupos:
        modalidad = _tipo_normalizado(grupo[0])
        if tipos is not None and not (
            modalidad in tipos
            or (modalidad == "movil 24h" and "movil" in tipos)
        ):
            continue
        fichas.append(_ficha(grupo, instante))

    fichas.sort(key=lambda f: (
        _ESTADOS[f["estado"]] if preferir_abiertos else 0,
        f["distancia"],
    ))

    fuente = (
        "API municipal consultada"
        if geografia["fuente"] == "api"
        else "corpus local de respaldo (potencialmente desactualizado)"
    )
    encabezado = (
        f"{clasificacion}\n\n" if clasificacion is not None else ""
    )
    if instante is not None:
        encabezado += (
            f"Para las coordenadas facilitadas, a las "
            f"{instante:%H:%M} del {instante:%d/%m/%Y} "
            f"(hora de Madrid), estas son las opciones registradas "
            f"en la {fuente}. "
        )
    else:
        encabezado += (
            "No se pudo obtener la hora actual de Madrid; "
            "no es posible comprobar qué opciones están en horario. "
            f"Estas son las opciones de la {fuente}. "
        )

    if requiere_residuo:
        encabezado += (
            "Se muestran únicamente modalidades de recogida "
            "respaldadas por la clasificación documental."
        )
    else:
        encabezado += (
            "No se ha verificado qué residuos admite cada instalación."
        )

    if not fichas:
        respuesta = (
            encabezado + "\n\nNo se encontraron instalaciones compatibles "
            "con la información disponible dentro del radio consultado."
        )
    else:
        abiertas = [f for f in fichas if f["estado"] == "dentro"]
        desconocidas = [f for f in fichas if f["estado"] == "desconocido"]
        cerradas = [f for f in fichas if f["estado"] == "fuera"]

        if preferir_abiertos:
            seleccion = (abiertas[:5] + desconocidas[:2] + cerradas[:2])[:7]
        else:
            seleccion = fichas[:7]

        introduccion = (
            "Opciones dentro del horario publicado, ordenadas "
            "de menor a mayor distancia:"
            if abiertas and preferir_abiertos
            else "Opciones cercanas y situación según horario publicado:"
        )
        if not abiertas:
            introduccion += (
                " No se ha confirmado ninguna opción dentro "
                "del horario publicado en esta búsqueda."
            )
        lineas = []
        for indice, f in enumerate(seleccion, 1):
            linea = (
                f"{indice}. {f['tipo']} — {f['direccion']} "
                f"({f['distancia']:.0f} m en línea recta). "
                f"**{_etiqueta(f['estado'])}**. "
                f"Horario registrado: {f['horario']}."
            )
            lineas.append(linea)
        respuesta = encabezado + "\n\n" + introduccion + "\n" + "\n".join(lineas)

    respuesta += (
        "\n\nLa comparación refleja horarios publicados, no una "
        "confirmación de apertura en tiempo real ni posibles incidencias "
        "o festivos. Las distancias son en línea recta."
    )
    plan = [
        "Confirmar que el residuo coincide con las condiciones "
        "documentadas para estas modalidades de recogida.",
        "Elegir la instalación compatible más conveniente según "
        "horario y distancia; confirmar incidencias antes de acudir.",
    ]
    return respuesta, plan
