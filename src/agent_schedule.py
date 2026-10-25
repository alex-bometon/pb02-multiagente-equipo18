"""Evaluaciones temporales trazables para el asistente de reciclaje de Madrid.

No infiere apertura física en tiempo real. Solo utiliza la hora de la
herramienta ``hora_actual`` y horarios/documentación identificables.
Las clasificaciones proceden exclusivamente del RAG.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any, Mapping

from src.text_utils import normalizar_texto


FUENTE_HORARIO_VIDRIO = (
    "https://www.madrid.es/portales/munimadrid/es/Recogida-de-vidrio/"
    "?vgnextchannel=f81379ed268fe410VgnVCM1000000b205a0aRCRD"
    "&vgnextoid=e3ab6c9c78eb2910VgnVCM2000001f4a900aRCRD"
)

# Ordenanza de Madrid: envases de vidrio en la vía pública, 08:00-22:30.
VIDRIO_INICIO = time(8, 0)
VIDRIO_FIN = time(22, 30)

_DIAS = (
    "lunes", "martes", "miercoles", "jueves", "viernes",
    "sabado", "domingo",
)


def _evidencias(
    estado: Mapping[str, Any], nombre: str, rol: str | None = None
) -> list[Any]:
    """Recupera resultados ya ejecutados; nunca vuelve a llamar a tools."""
    return [
        item.get("datos")
        for item in estado.get("evidencias", [])
        if isinstance(item, dict)
        and item.get("herramienta") == nombre
        and (rol is None or item.get("rol") == rol)
    ]


def es_vidrio_envase_confirmado(estado: Mapping[str, Any]) -> bool:
    """Aplica el horario especial SOLO a envases clasificados por el RAG.

    No confunde vasos/cristal/espejos con la fracción vidrio.
    """
    consulta = normalizar_texto(str(estado.get("pedido") or ""))
    if not re.search(r"\b(?:vidrio|botellas?|tarros?|frascos?|envases?)\b", consulta):
        return False

    for datos in _evidencias(estado, "consultar_rag", "residuos"):
        if not isinstance(datos, dict) or datos.get("abstencion") is not False:
            continue
        respuesta = normalizar_texto(str(datos.get("respuesta") or ""))
        # Una prohibición del contenedor verde no implica que esté permitido.
        if re.search(
            r"\b(?:no|nunca|tampoco)\s+(?:\w+\s+){0,8}"
            r"contenedor(?:es)?\s+verde\b", respuesta
        ):
            continue
        if (
            "vidrio" in respuesta
            and re.search(r"\bcontenedor(?:es)?\s+(?:verde(?:s)?|de\s+(?:envases\s+de\s+)?vidrio)\b", respuesta)
            and not re.search(r"\b(?:vasos?|copas?|espejos?|ventanas?|vitroceramicas?)\b", consulta)
        ):
            return True
    return False


def requiere_hora(estado: Mapping[str, Any], rol: str) -> bool:
    """Determina necesidad temporal tras consultar las tools primarias."""
    if rol == "residuos":
        return es_vidrio_envase_confirmado(estado)

    if rol != "puntos":
        return False

    for datos in _evidencias(
        estado, "buscar_puntos_limpios_cercanos", "puntos"
    ):
        if isinstance(datos, dict) and datos.get("puntos"):
            return True

    consulta = normalizar_texto(str(estado.get("pedido") or ""))
    hay_documentacion = any(
        isinstance(datos, dict) and datos.get("abstencion") is False
        for datos in _evidencias(estado, "consultar_rag", "puntos")
    )
    return bool(
        re.search(r"\b(?:horarios?|abiert\w*|cerrad\w*|ahora|hoy)\b", consulta)
        and hay_documentacion
    )


def debe_buscar_puntos_automaticamente(estado: Mapping[str, Any]) -> bool:
    """Activa cercanía para un destino limpio documentado CON coordenadas.

    Solo se usa tras ejecutar residuos y verificar una clasificación.
    """
    if estado.get("ubicacion") is None:
        return False
    consulta = normalizar_texto(str(estado.get("pedido") or ""))
    if not re.search(r"\b(?:tir\w*|recicl\w*|deposit\w*|desech\w*|donde\s+van?)\b", consulta):
        return False

    for datos in _evidencias(estado, "consultar_rag", "residuos"):
        if not isinstance(datos, dict) or datos.get("abstencion") is not False:
            continue
        if not datos.get("fuentes"):
            continue
        respuesta = normalizar_texto(str(datos.get("respuesta") or ""))
        if not re.search(r"\bpuntos?\s+limpios?\b", respuesta):
            continue
        # Evitar activar puntos a partir de una prohibición/negación.
        if re.search(r"\b(?:no|nunca)\s+(?:(?:se|debe|van|va|al|en|a)\s+){0,4}(?:los?\s+)?puntos?\s+limpios?\b", respuesta):
            continue
        return True
    return False



def tipo_punto_documentado(estado: Mapping[str, Any]) -> str:
    """Filtra por una modalidad si el RAG documenta exactamente una.

    Una clasificación genérica o con varias modalidades mantiene 'todos'.
    Las instalaciones nunca se inventan a partir de esta información.
    """
    modalidades: set[str] = set()
    for datos in _evidencias(estado, "consultar_rag", "residuos"):
        if not isinstance(datos, dict) or datos.get("abstencion") is not False:
            continue
        texto = normalizar_texto(str(datos.get("respuesta") or ""))
        if not re.search(r"\bpuntos?\s+limpios?\b", texto):
            continue
        if re.search(r"\bpuntos?\s+limpios?\s+(?:movil|moviles)\b", texto):
            modalidades.add("movil")
        if re.search(r"\bpuntos?\s+limpios?\s+(?:fijo|fijos)\b", texto):
            modalidades.add("fijo")
        if re.search(r"\bpuntos?\s+limpios?\s+de\s+proximidad\b", texto):
            modalidades.add("proximidad")
    return next(iter(modalidades)) if len(modalidades) == 1 else "todos"

def _hora(estado: Mapping[str, Any]) -> datetime | None:
    for dato in _evidencias(estado, "hora_actual"):
        if not isinstance(dato, str):
            continue
        try:
            instante = datetime.fromisoformat(dato.strip())
        except ValueError:
            continue
        if instante.tzinfo is not None and instante.utcoffset() is not None:
            return instante
    return None


def _extraer_hora(valor: Any) -> time | None:
    if not isinstance(valor, str):
        return None
    texto = valor.strip()
    # Solo aceptar campos que contienen UNA hora, no intervalos ambiguos.
    patron = r"(?:\b|^)([01]?\d|2[0-3])[:.]([0-5]\d)(?:\b|$)"
    coincidencias = list(re.finditer(patron, texto))
    if len(coincidencias) != 1:
        return None
    m = coincidencias[0]
    return time(int(m.group(1)), int(m.group(2)))


def _dias_permitidos(valor: Any) -> set[int] | None:
    """Interpreta días inequívocos; None implica horario no verificable."""
    if not isinstance(valor, str) or not valor.strip():
        return None
    texto = normalizar_texto(valor)
    texto = re.sub(r"\bsabados\b", "sabado", texto)
    texto = re.sub(r"\bdomingos\b", "domingo", texto)
    rango = re.fullmatch(
        r"(?:de\s+)?(" + "|".join(_DIAS) + r")\s+a\s+("
        + "|".join(_DIAS) + r")", texto,
    )
    if rango:
        inicio = _DIAS.index(rango.group(1))
        fin = _DIAS.index(rango.group(2))
        return set(range(inicio, fin + 1)) if inicio <= fin else None
    if texto in {"todos los dias", "diario", "cada dia", "lunes a domingo"}:
        return set(range(7))
    if texto in {"laborables", "lunes a viernes", "de lunes a viernes"}:
        return set(range(5))
    if texto in {"fin de semana", "sabados y domingos"}:
        return {5, 6}
    tokens = re.findall(r"\b(?:" + "|".join(_DIAS) + r")\b", texto)
    if not tokens:
        return None
    sin_dias = re.sub(r"\b(?:" + "|".join(_DIAS) + r")\b", "", texto)
    if re.sub(r"[\s,;/y]+", "", sin_dias):
        return None
    return {_DIAS.index(dia) for dia in tokens}


def _situacion_punto(punto: Mapping[str, Any], instante: datetime) -> str:
    apertura = _extraer_hora(punto.get("horario_apertura"))
    cierre = _extraer_hora(punto.get("horario_cierre"))
    dias = _dias_permitidos(punto.get("dia_semana"))
    if apertura is None or cierre is None or dias is None:
        return "No se puede determinar su horario vigente con estos datos"
    ahora = instante.timetz().replace(tzinfo=None)
    hoy = instante.weekday()
    if apertura < cierre:
        coincide = hoy in dias and apertura <= ahora < cierre
    elif apertura > cierre:
        coincide = (
            (hoy in dias and ahora >= apertura)
            or ((hoy - 1) % 7 in dias and ahora < cierre)
        )
    else:
        return "Horario de apertura y cierre ambiguo"
    if coincide:
        return "Dentro del horario publicado (apertura real no verificada)"
    return "Fuera del horario publicado; no consta disponible ahora"


def _horarios_documentales(
    estado: Mapping[str, Any], instante: datetime
) -> list[str]:
    """Evalúa SOLO registros estructurados de paradas móviles del RAG.

    Las frases abiertas de Gemini no se tratan como horarios verificados.
    """
    consulta = normalizar_texto(str(estado.get("pedido") or ""))
    if not re.search(r"\b(?:horarios?|abiert\w*|cerrad\w*|ahora|hoy)\b", consulta):
        return []

    salida = []
    for datos in _evidencias(estado, "consultar_rag", "puntos"):
        if not isinstance(datos, dict) or datos.get("abstencion") is not False:
            continue
        fuentes = {
            item.get("chunk_id")
            for item in datos.get("fuentes", [])
            if isinstance(item, dict)
        }
        for indice, chunk in enumerate(datos.get("chunks", []), start=1):
            if f"C{indice}" not in fuentes or not isinstance(chunk, dict):
                continue
            texto = chunk.get("text")
            if not isinstance(texto, str):
                continue
            campos = {}
            for campo in ("Dirección", "Día", "Horario"):
                coincidencia = re.search(
                    rf"(?:^|\n){campo}\s*:\s*([^\n]+)",
                    texto, flags=re.IGNORECASE,
                )
                if coincidencia:
                    campos[campo] = coincidencia.group(1).strip()
            if len(campos) != 3:
                continue
            horas = re.findall(
                r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)",
                campos["Horario"],
            )
            if len(horas) != 2:
                salida.append(
                    f"- {campos['Dirección']}: horario documentado "
                    "no interpretable para disponibilidad actual."
                )
                continue
            punto = {
                "horario_apertura": ":".join(horas[0]),
                "horario_cierre": ":".join(horas[1]),
                "dia_semana": campos["Día"],
            }
            salida.append(
                f"- {campos['Dirección']}: {_situacion_punto(punto, instante)} "
                f"(horario documentado: {campos['Día']}, {campos['Horario']})."
            )
        if salida:
            break
    return salida[:5]


def anexos_temporales(
    estado: Mapping[str, Any]
) -> tuple[str, list[dict[str, str]]]:
    """Construye verificaciones sin LLM adicional ni horarios inventados."""
    instante = _hora(estado)
    if instante is None:
        if es_vidrio_envase_confirmado(estado) or _evidencias(
            estado, "buscar_puntos_limpios_cercanos", "puntos"
        ):
            return (
                "No se pudo verificar la hora actual de Madrid; "
                "no afirmo disponibilidad en este momento.",
                [],
            )
        return "", []

    prefijo = f"Comprobación horaria (Madrid, {instante:%d/%m/%Y %H:%M}):"
    textos: list[str] = []
    fuentes: list[dict[str, str]] = []

    if es_vidrio_envase_confirmado(estado):
        permitido = VIDRIO_INICIO <= instante.timetz().replace(tzinfo=None) < VIDRIO_FIN
        if permitido:
            textos.append(
                "Envases de vidrio: está permitido depositarlos en el "
                "contenedor verde entre las 08:00 y las 22:30. "
                "En este momento estás dentro del horario permitido."
            )
        else:
            textos.append(
                "Envases de vidrio: ahora NO está permitido depositarlos "
                "en el contenedor verde de la vía pública. "
                "Horario permitido: 08:00–22:30, todos los días."
            )
        fuentes.append({
            "chunk_id": "NORMA-VIDRIO",
            "source": "Ayuntamiento de Madrid — Recogida de vidrio",
            "path": FUENTE_HORARIO_VIDRIO,
        })

    for datos in _evidencias(estado, "buscar_puntos_limpios_cercanos", "puntos"):
        if not isinstance(datos, dict) or not isinstance(datos.get("puntos"), list):
            continue
        puntos = datos["puntos"]
        if puntos:
            textos.append(
                "Estado orientativo de horarios de los puntos más cercanos "
                "(distancias en línea recta):"
            )
        for punto in puntos[:5]:
            if not isinstance(punto, dict):
                continue
            direccion = str(punto.get("direccion") or "Sin dirección")
            distancia = punto.get("distancia_metros")
            distancia_str = (
                f"{distancia:.0f} m" if isinstance(distancia, (int, float))
                and not isinstance(distancia, bool) else "distancia desconocida"
            )
            franja = ", ".join(
                str(punto[campo]).strip()
                for campo in ("dia_semana", "horario_apertura", "horario_cierre")
                if punto.get(campo)
            )
            texto = (
                f"- {direccion} ({distancia_str}): "
                f"{_situacion_punto(punto, instante)}."
            )
            if franja:
                texto += f" Datos del horario: {franja}."
            textos.append(texto)
        if puntos:
            textos.append(
                "Los horarios proceden de los registros disponibles; "
                "festivos, incidencias y modificaciones pueden alterar la apertura."
            )
        break

    if not _evidencias(estado, "buscar_puntos_limpios_cercanos", "puntos"):
        textos.extend(_horarios_documentales(estado, instante))

    if not textos:
        return "", []
    return prefijo + "\n" + "\n".join(textos), fuentes
