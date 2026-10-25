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


def _franja_horaria(punto: Mapping[str, Any]) -> tuple[time, time] | None:
    """Lee horas separadas o un campo municipal con intervalo completo.

    Ejemplo real de ArcGIS: ABIERTO='Sábado de 10:30 a 12:00 h',
    CERRADO=None. No interpreta intervalos con más de dos horas.
    """
    apertura = _extraer_hora(punto.get("horario_apertura"))
    cierre = _extraer_hora(punto.get("horario_cierre"))
    if apertura is not None and cierre is not None:
        return apertura, cierre

    for campo in ("horario_apertura", "horario_cierre"):
        valor = punto.get(campo)
        if not isinstance(valor, str):
            continue
        horas = re.findall(
            r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)", valor
        )
        if len(horas) == 2:
            return (time(int(horas[0][0]), int(horas[0][1])),
                    time(int(horas[1][0]), int(horas[1][1])))
    return None


def _dias_punto(punto: Mapping[str, Any]) -> set[int] | None:
    """Prioriza el campo de días; usa texto horario si lleva día explícito."""
    dias = _dias_permitidos(punto.get("dia_semana"))
    if dias is not None:
        return dias
    for campo in ("horario_apertura", "horario_cierre"):
        valor = punto.get(campo)
        if not isinstance(valor, str):
            continue
        texto = normalizar_texto(valor)
        encontrados = re.findall(r"\b(?:" + "|".join(_DIAS) + r")\b", texto)
        if len(set(encontrados)) == 1:
            return {_DIAS.index(encontrados[0])}
    return None


def _es_24_horas(punto: Mapping[str, Any]) -> bool:
    """Reconoce modalidad 24h o un horario explícito todos los días 24h."""
    tipo = normalizar_texto(str(punto.get("tipo") or ""))
    if re.search(r"\b24\s*h(?:oras)?\b", tipo):
        return True
    horario = normalizar_texto(" ".join(
        str(punto.get(campo) or "")
        for campo in ("horario_apertura", "horario_cierre")
    ))
    dia = normalizar_texto(str(punto.get("dia_semana") or ""))
    diario = bool(re.search(
        r"\b(?:todos los dias|de lunes a domingo|lunes a domingo|diario)\b",
        horario + " " + dia,
    ))
    sin_restricciones = bool(re.search(r"\b24\s*(?:h|horas)\b", horario))
    return diario and sin_restricciones


def _intervalos_semanales(
    punto: Mapping[str, Any],
) -> dict[int, list[tuple[time, time]]] | None:
    """Lee horarios municipales con varios días y varias franjas.

    Solo declara días cerrados cuando todos los segmentos relevantes son
    legibles; si hay cualquier segmento de horas incompleto, se abstiene.
    """
    texto = normalizar_texto(str(punto.get("horario_apertura") or ""))
    patron_dias = re.compile(
        r"\b(?:todos los dias|(?:de\s+)?lunes\s+a\s+"
        r"(?:viernes|sabado|domingo)|lunes|martes|miercoles|jueves|"
        r"viernes|sabados?|domingos?)\b"
    )
    etiquetas = list(patron_dias.finditer(texto))
    if not etiquetas:
        return None
    # La fecha es distinta por registro cuando el campo representa un
    # horario móvil simple, pero puede incluir varias franjas para un día.
    tabla: dict[int, list[tuple[time, time]]] = {}
    patron_hora = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")
    for i, etiqueta in enumerate(etiquetas):
        nombre = etiqueta.group(0).removeprefix("de ")
        dias = _dias_permitidos(nombre)
        if dias is None:
            return None
        fin = etiquetas[i+1].start() if i+1 < len(etiquetas) else len(texto)
        segmento = texto[etiqueta.end():fin]
        horas = [time(int(m.group(1)), int(m.group(2)))
                 for m in patron_hora.finditer(segmento)]
        if not horas or len(horas) % 2:
            return None
        ventanas = list(zip(horas[::2], horas[1::2]))
        if any(a == b for a, b in ventanas):
            return None
        for dia in dias:
            tabla.setdefault(dia, []).extend(ventanas)
    return tabla


def estado_horario(punto: Mapping[str, Any], instante: datetime) -> str:
    """Compara horario documentado con hora real; no confirma apertura física."""
    if _es_24_horas(punto):
        return "dentro"

    # Un horario semanal compuesto debe interpretarse completo, no solo su
    # primer intervalo. Un domingo omitido queda fuera de horario.
    tabla = _intervalos_semanales(punto)
    hoy = instante.weekday()
    ahora = instante.timetz().replace(tzinfo=None)
    if tabla is not None:
        if any(a <= ahora < b for a, b in tabla.get(hoy, []) if a < b):
            return "dentro"
        if any(a > b and ahora >= a for a, b in tabla.get(hoy, [])):
            return "dentro"
        ayer = (hoy - 1) % 7
        if any(a > b and ahora < b for a, b in tabla.get(ayer, [])):
            return "dentro"
        return "fuera"

    franja = _franja_horaria(punto)
    dias = _dias_punto(punto)
    if franja is None or dias is None:
        return "desconocido"
    apertura, cierre = franja
    if apertura < cierre:
        coincide = hoy in dias and apertura <= ahora < cierre
    elif apertura > cierre:
        coincide = (
            (hoy in dias and ahora >= apertura)
            or ((hoy - 1) % 7 in dias and ahora < cierre)
        )
    else:
        return "desconocido"
    return "dentro" if coincide else "fuera"


def _situacion_punto(punto: Mapping[str, Any], instante: datetime) -> str:
    situacion = estado_horario(punto, instante)
    if situacion == "dentro":
        return "Dentro del horario publicado (apertura real no verificada)"
    if situacion == "fuera":
        return "Fuera del horario publicado; no consta disponible ahora"
    return "No se puede determinar el horario vigente con estos datos"


def _tipos_admitidos(estado: Mapping[str, Any]) -> set[str] | None:
    """Extrae modalidades afirmadas en el RAG; no infiere compatibilidad."""
    textos = []
    for datos in _evidencias(estado, "consultar_rag", "residuos"):
        if (isinstance(datos, dict) and datos.get("abstencion") is False
                and datos.get("fuentes")):
            textos.append(normalizar_texto(str(datos.get("respuesta") or "")))
    if not textos:
        return None
    contenido = " ".join(textos)
    tipos: set[str] = set()
    if re.search(r"\bpuntos?\s+limpios?\s+de\s+proximidad\b", contenido):
        tipos.add("proximidad")
    if re.search(r"\bpuntos?\s+limpios?\s+fijos?\b", contenido):
        tipos.add("fijo")
    if re.search(r"\bpuntos?\s+limpios?\s+movil(?:es)?\b", contenido):
        tipos.add("movil")
    if re.search(r"\bpuntos?\s+limpios?\s+moviles?\s+24\s*h", contenido):
        tipos.add("movil 24h")
    return tipos or None


def _tipo_normalizado(punto: Mapping[str, Any]) -> str:
    tipo = normalizar_texto(str(punto.get("tipo") or ""))
    if "proximidad" in tipo:
        return "proximidad"
    if "fijo" in tipo:
        return "fijo"
    if "movil" in tipo:
        if "24h" in tipo or "24 h" in tipo:
            return "movil 24h"
        return "movil"
    return "otro"


def priorizar_puntos(
    puntos: list[dict[str, Any]],
    instante: datetime,
    *,
    preferir_abiertos: bool,
    tipos_admitidos: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Ordena candidatos sin hacer nuevas llamadas ni inventar aceptación.

    Cuando hay evidencia de modalidades admitidas, descarta las no
    mencionadas. Dentro de cada prioridad conserva la menor distancia.
    """
    candidatos = [p for p in puntos if isinstance(p, dict)]
    if tipos_admitidos is not None:
        candidatos = [p for p in candidatos if _tipo_normalizado(p) in tipos_admitidos]

    pesos = {"dentro": 0, "desconocido": 1, "fuera": 2}

    def clave(punto: Mapping[str, Any]) -> tuple[float, float]:
        distancia = punto.get("distancia_metros")
        metros = float(distancia) if isinstance(distancia, (int, float)) and not isinstance(distancia, bool) else float("inf")
        disponibilidad = pesos[estado_horario(punto, instante)] if preferir_abiertos else 0
        return disponibilidad, metros

    return sorted(candidatos, key=clave)


def _priorizar_abiertos(pedido: str) -> bool:
    """Una solicitud explícita de ranking por distancia prima sobre horario."""
    texto = normalizar_texto(pedido)
    pide_mas_cercanos = bool(re.search(
        r"\b(?:mas\s+cercan[oa]s?|ordenad\w*\s+por\s+distancia|"
        r"solo\s+por\s+distancia)\b", texto
    ))
    pide_abierto = bool(re.search(
        r"\b(?:abiert\w*|disponible\w*\s+ahora|puedo\s+ir\s+ahora|"
        r"en\s+este\s+momento)\b", texto
    ))
    return pide_abierto or not pide_mas_cercanos


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
        candidatos = datos.get("puntos_candidatos") or datos["puntos"]
        if not isinstance(candidatos, list):
            candidatos = datos["puntos"]
        preferir = _priorizar_abiertos(str(estado.get("pedido") or ""))
        # Solo se filtran modalidades si la clasificación ha sido respaldada
        # por las fuentes. Consultas genéricas sobre puntos no se filtran.
        admitidos = _tipos_admitidos(estado) if estado.get("necesita_residuos") else None
        puntos = priorizar_puntos(
            candidatos, instante,
            preferir_abiertos=preferir,
            tipos_admitidos=admitidos,
        )
        if puntos:
            etiqueta = (
                "Opciones de modalidades documentadas para el residuo"
                if admitidos is not None else
                "Instalaciones encontradas (aceptación del residuo no confirmada)"
            )
            if preferir:
                textos.append(
                    etiqueta + ", priorizadas por horario publicado "
                    "y luego por distancia en línea recta:"
                )
            else:
                textos.append(
                    etiqueta + ", ordenadas por distancia en línea recta "
                    "(se indica su situación horaria):"
                )
            if preferir and not any(
                estado_horario(p, instante) == "dentro" for p in puntos
            ):
                textos.append(
                    "Ninguna de estas opciones tiene disponibilidad "
                    "confirmable por el horario publicado en este momento."
                )
        else:
            textos.append(
                "No se encontró un punto con compatibilidad documentada "
                "entre los candidatos consultados."
            )
        for punto in puntos[:5]:
            direccion = str(punto.get("direccion") or "Sin dirección")
            tipo = str(punto.get("tipo") or "Tipo desconocido")
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
                f"- {tipo}, {direccion} ({distancia_str}): "
                f"{_situacion_punto(punto, instante)}."
            )
            if franja:
                texto += f" Horario documentado: {franja}."
            textos.append(texto)
        fuente_geo = datos.get("fuente")
        if puntos:
            textos.append(
                "Fuente de ubicaciones: "
                + ("API municipal." if fuente_geo == "api" else "corpus local.")
                + " El horario publicado no garantiza apertura real; "
                  "puede haber cambios, festivos o incidencias."
            )
        if len(candidatos) >= 60:
            textos.append(
                "La selección utiliza un conjunto limitado de candidatos; "
                "pueden existir otras opciones dentro del radio de búsqueda."
            )
        break

    if not _evidencias(estado, "buscar_puntos_limpios_cercanos", "puntos"):
        textos.extend(_horarios_documentales(estado, instante))

    if not textos:
        return "", []
    return prefijo + "\n" + "\n".join(textos), fuentes
