"""Respuesta geográfica útil con evidencia de RAG, API y hora.

Este módulo no llama a herramientas ni al LLM: interpreta sus resultados
ya registrados en el estado de LangGraph. No confirma apertura física.
"""

from __future__ import annotations

import re
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
_MAX_ALTERNATIVAS_ABIERTAS = 1
_MAX_OPCIONES_CERCANAS = 3


def _datos(estado: Mapping[str, Any], nombre: str) -> list[dict[str, Any]]:
    """Consulta únicamente las evidencias auténticas del turno."""
    resultados = []
    for registro in estado.get("evidencias", []):
        if not isinstance(registro, dict):
            continue
        if registro.get("herramienta") != nombre:
            continue
        datos = registro.get("datos")
        if isinstance(datos, dict):
            resultados.append(datos)
    return resultados


def _clasificacion(estado: Mapping[str, Any]) -> str | None:
    """Conserva la clasificación y las citas emitidas por el RAG."""
    for registro in estado.get("evidencias", []):
        if not isinstance(registro, dict):
            continue
        if (
            registro.get("rol") != "residuos"
            or registro.get("herramienta") != "consultar_rag"
        ):
            continue
        datos = registro.get("datos")
        if (
            isinstance(datos, dict)
            and datos.get("abstencion") is False
            and datos.get("fuentes")
            and isinstance(datos.get("respuesta"), str)
            and datos["respuesta"].strip()
        ):
            return datos["respuesta"].strip()
    return None


def _candidatos(geografia: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prefiere candidatos completos al subconjunto visual de la API."""
    puntos = geografia.get("puntos_candidatos")
    if not isinstance(puntos, list):
        puntos = geografia.get("puntos")
    return [p for p in puntos if isinstance(p, dict)] if isinstance(puntos, list) else []


def _agrupar_puntos(
    puntos: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Reúne registros de una instalación que difieren en el día de visita."""
    grupos: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for punto in puntos:
        direccion = punto.get("direccion")
        tipo = punto.get("tipo")
        distancia = punto.get("distancia_metros")
        if (
            not isinstance(direccion, str)
            or not direccion.strip()
            or not isinstance(tipo, str)
            or not tipo.strip()
            or isinstance(distancia, bool)
            or not isinstance(distancia, (int, float))
            or not isfinite(distancia)
            or distancia < 0
        ):
            continue
        lat = punto.get("latitud")
        lon = punto.get("longitud")
        coordenadas = ""
        if (
            isinstance(lat, (int, float))
            and not isinstance(lat, bool)
            and isinstance(lon, (int, float))
            and not isinstance(lon, bool)
            and isfinite(lat)
            and isfinite(lon)
        ):
            coordenadas = f"{lat:.5f},{lon:.5f}"
        clave = (
            normalizar_texto(direccion),
            normalizar_texto(tipo),
            coordenadas,
        )
        grupos[clave].append(punto)
    return list(grupos.values())


def _estado_grupo(
    grupo: list[dict[str, Any]],
    instante: datetime | None,
) -> str:
    if instante is None:
        return "desconocido"
    estados = [estado_horario(punto, instante) for punto in grupo]
    if "dentro" in estados:
        return "dentro"
    if "desconocido" in estados:
        return "desconocido"
    return "fuera"


def _resumir_horario(
    grupo: list[dict[str, Any]],
    *,
    max_horarios: int = 2,
) -> str:
    """Muestra horarios municipales identificables sin inventar horarios."""
    horarios = []
    for punto in grupo:
        partes = []
        for campo in ("horario_apertura", "horario_cierre"):
            valor = punto.get(campo)
            if isinstance(valor, str) and valor.strip():
                limpio = valor.strip().rstrip(". ")
                if limpio not in partes:
                    partes.append(limpio)
        dia = punto.get("dia_semana")
        if isinstance(dia, str) and dia.strip():
            dia_limpio = dia.strip().rstrip(". ")
            contenido = normalizar_texto(" ".join(partes))
            if normalizar_texto(dia_limpio) not in contenido:
                partes.insert(0, dia_limpio)
        texto = ", ".join(partes)
        if texto and texto not in horarios:
            horarios.append(texto)
    if not horarios:
        return "No consta en el registro"
    descripcion = "; ".join(horarios[:max_horarios])
    if len(horarios) > max_horarios:
        descripcion += "; otros días documentados no mostrados"
    return descripcion


def _ficha(
    grupo: list[dict[str, Any]],
    instante: datetime | None,
) -> dict[str, Any]:
    punto = min(grupo, key=lambda p: float(p["distancia_metros"]))
    return {
        "direccion": punto["direccion"],
        "tipo": punto["tipo"],
        "distancia": float(punto["distancia_metros"]),
        "estado": _estado_grupo(grupo, instante),
        "horario": _resumir_horario(grupo),
    }


def _estado_visible(estado: str) -> str:
    if estado == "dentro":
        return "Dentro del horario publicado"
    if estado == "fuera":
        return "Fuera del horario publicado"
    return "Horario no comprobable"


def _descripcion_punto(ficha: Mapping[str, Any]) -> str:
    return (
        f"{ficha['tipo']} — {ficha['direccion']} "
        f"({ficha['distancia']:.0f} m en línea recta). "
        f"**{_estado_visible(ficha['estado'])}**. "
        f"Horario: {ficha['horario']}."
    )


def _opcion_marquesinas(
    clasificacion: str | None,
    pedido: str,
) -> str:
    """Informa del límite de geolocalización sin inventar aceptación."""
    if clasificacion is None:
        return ""
    texto = normalizar_texto(clasificacion)
    if re.search(r"\bmarquesinas?\b", texto):
        return (
            "\n\n**Marquesinas de autobús:** la documentación también "
            "menciona esta modalidad, pero la API de puntos limpios "
            "no devuelve sus ubicaciones. No puedo comparar su "
            "distancia ni comprobar su disponibilidad."
        )
    if re.search(r"\b(?:pilas?|baterias?)\b", normalizar_texto(pedido)):
        return (
            "\n\n**Límite geográfico:** esta API no localiza "
            "contenedores para pilas en marquesinas de autobús; "
            "no puedo determinar si existe alguno más cercano. "
            "La evidencia del RAG de este turno no confirma esa modalidad."
        )
    return ""


def _plan(
    clasificacion: str | None,
    recomendacion: Mapping[str, Any] | None,
) -> list[str]:
    """Proporciona pasos revisables por HITL, no acciones ya ejecutadas."""
    if recomendacion is None:
        return [
            "Comprobar que la modalidad elegida admite el residuo "
            "según las fuentes documentales disponibles.",
            "Consultar las condiciones y horarios actualizados antes "
            "de desplazarse a una instalación.",
        ]
    paso = (
        "Considerar como primera opción "
        f"{recomendacion['direccion']} "
        f"({recomendacion['tipo']}, "
        f"{recomendacion['distancia']:.0f} m en línea recta), "
        "siempre que la clasificación documental corresponda "
        "al residuo concreto."
    )
    return [
        paso,
        "Antes de acudir, confirmar posibles incidencias o cambios "
        "en el horario publicado; la apertura física no está verificada.",
    ]


def respuesta_geografica_integrada(
    estado: Mapping[str, Any],
) -> tuple[str, list[str]] | None:
    """Sintetiza un destino útil con RAG + API + hora, sin nuevas tools.

    Por defecto: la opción abierta más cercana, hasta otra opción abierta
    y los puntos físicamente más cercanos, indicando si están cerrados.
    Si el usuario exige orden estricto por distancia, se respeta.
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
        return None

    tipos = _tipos_admitidos(estado) if requiere_residuo else None
    if requiere_residuo and not tipos:
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

    cercanos = sorted(fichas, key=lambda f: f["distancia"])
    abiertos = [f for f in cercanos if f["estado"] == "dentro"]
    fuente = (
        "API municipal"
        if geografia["fuente"] == "api"
        else "corpus local de respaldo, potencialmente desactualizado"
    )
    prefijo = f"{clasificacion}\n\n" if clasificacion else ""
    if instante is not None:
        prefijo += (
            f"**Consulta:** {instante:%d/%m/%Y}, "
            f"{instante:%H:%M} (hora de Madrid). "
            f"Ubicaciones de la {fuente}."
        )
    else:
        prefijo += (
            "**No se pudo obtener la hora actual de Madrid**. "
            f"Ubicaciones de la {fuente}; no se puede verificar "
            "disponibilidad por horario."
        )
    if not requiere_residuo:
        prefijo += (
            " No se ha verificado qué residuos acepta cada instalación."
        )

    if not cercanos:
        respuesta = (
            prefijo
            + "\n\nNo se encontraron instalaciones compatibles dentro "
            "del radio consultado con la evidencia disponible."
        )
        respuesta += _opcion_marquesinas(
            clasificacion, str(estado.get("pedido") or "")
        )
        return respuesta, _plan(clasificacion, None)

    secciones = [prefijo]
    principal = abiertos[0] if abiertos else None
    if preferir_abiertos:
        if principal is not None:
            secciones.append(
                "**Mejor opción para acudir ahora, según el horario "
                "publicado:**\n\n" + _descripcion_punto(principal)
            )
            otras_abiertas = abiertos[1:1 + _MAX_ALTERNATIVAS_ABIERTAS]
            if otras_abiertas:
                secciones.append(
                    "**Alternativa también dentro de horario:**\n\n"
                    + "\n".join(
                        "- " + _descripcion_punto(f) for f in otras_abiertas
                    )
                )
        else:
            secciones.append(
                "**No se encontró una instalación con disponibilidad "
                "confirmable por el horario publicado** dentro del radio "
                "consultado."
            )

        # Esta sección prioriza literalmente los metros, no repite la mejor
        # opción y revela que un móvil próximo puede estar cerrado.
        presentados = {id(f) for f in ([principal] if principal else [])}
        presentados.update(id(f) for f in abiertos[1:1 + _MAX_ALTERNATIVAS_ABIERTAS])
        restantes = [f for f in cercanos if id(f) not in presentados]
        opciones_cercanas = restantes[:_MAX_OPCIONES_CERCANAS]
        if opciones_cercanas:
            secciones.append(
                "**Otros puntos cercanos, ordenados por distancia:**\n\n"
                + "\n".join(
                    "- " + _descripcion_punto(f) for f in opciones_cercanas
                )
            )
    else:
        seleccion = cercanos[:4]
        secciones.append(
            "**Puntos más cercanos, ordenados por distancia:**\n\n"
            + "\n".join(
                "- " + _descripcion_punto(f) for f in seleccion
            )
        )
        if principal is not None and principal not in seleccion:
            secciones.append(
                "**Primera opción dentro del horario publicado:**\n\n"
                + _descripcion_punto(principal)
            )

    respuesta = "\n\n".join(secciones)
    respuesta += _opcion_marquesinas(
        clasificacion, str(estado.get("pedido") or "")
    )
    respuesta += (
        "\n\n**Alcance de la comprobación:** estar dentro del horario "
        "publicado no garantiza que una instalación esté abierta "
        "físicamente. Puede haber festivos, cambios o incidencias. "
        "Las distancias son en línea recta."
    )
    return respuesta, _plan(clasificacion, principal)
