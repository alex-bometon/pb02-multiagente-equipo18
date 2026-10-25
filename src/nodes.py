"""Nodos del grafo multiagente ReciclaTIA.

El grafo y sus aristas se definen en ``src.graph``.
Los especialistas usan el ejecutor central de herramientas.
La aprobación humana corresponde a ``src.agent``.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from math import isfinite
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from config import MAX_CONTEXTO_CARACTERES
from src.agent_prompts import PROMPT_SINTESIS, PROMPT_SUPERVISOR
from src.agent_routing import (
    requiere_llm_supervisor,
    resolver_enrutamiento,
)
from src.agent_schedule import (
    anexos_temporales,
    debe_buscar_puntos_automaticamente,
)
from src.agent_tool_loop import (
    _consulta_proximidad,
    ejecutar_especialista,
)
from src.llm import get_llm
from src.state import EstadoGrafo, agregar_evento, registrar_turno
from src.text_utils import normalizar_texto


# =========================================================
# GENERACIÓN ESTRUCTURADA
# =========================================================


def _generar_json(
    instrucciones: str,
    entrada: dict[str, Any],
    *,
    supervisor_llm: bool = False,
) -> dict[str, Any]:
    """Consulta a Gemini solo cuando no basta una decisión local."""
    mensajes = [
        SystemMessage(content=instrucciones),
        HumanMessage(content=json.dumps(entrada, ensure_ascii=False)),
    ]

    respuesta = get_llm(supervisor=supervisor_llm).invoke(mensajes)
    datos = JsonOutputParser().invoke(respuesta)

    if not isinstance(datos, dict):
        raise ValueError("El modelo no ha devuelto un objeto JSON.")

    return datos


# =========================================================
# NODO 1 — SUPERVISOR
# =========================================================


def supervisor(estado: EstadoGrafo) -> dict[str, Any]:
    """Selecciona el siguiente nodo con el mínimo uso del LLM."""
    consultar_modelo = requiere_llm_supervisor(estado)
    propuesta = None
    fallo_llm = False

    if consultar_modelo:
        entrada = {
            "pedido": estado["pedido"],
            "contexto_revision": estado["contexto_revision"],
            "hay_imagen": bool(estado["ruta_imagen"]),
            "hay_ubicacion": estado["ubicacion"] is not None,
            "necesita_residuos": estado["necesita_residuos"],
            "necesita_puntos": estado["necesita_puntos"],
            "residuos_completado": estado["residuos_completado"],
            "puntos_completado": estado["puntos_completado"],
            "siguiente_anterior": estado["siguiente"],
            "hay_respuesta": bool(estado["respuesta"]),
            "hay_plan": len(estado["plan"]) >= 2,
        }

        try:
            propuesta = _generar_json(
                PROMPT_SUPERVISOR,
                entrada,
                supervisor_llm=True,
            )
        except Exception:
            fallo_llm = True

    decision = resolver_enrutamiento(estado, propuesta)

    if not consultar_modelo:
        accion = "ruta_directa"
    elif fallo_llm or decision["corregida"]:
        accion = "fallback"
    else:
        accion = "ruta_llm"

    detalle = (
        f"Siguiente: {decision['siguiente']}; "
        f"{decision['motivo_correccion'] or 'ruta validada'}"
    )

    return {
        "necesita_residuos": decision["necesita_residuos"],
        "necesita_puntos": decision["necesita_puntos"],
        "siguiente": decision["siguiente"],
        "iteracion": estado["iteracion"] + 1,
        "log": agregar_evento(
            estado["log"],
            nodo="supervisor",
            accion=accion,
            detalle=detalle,
        ),
    }


# =========================================================
# NODOS 2 Y 3 — ESPECIALISTAS
# =========================================================


def residuos(estado: EstadoGrafo) -> dict[str, Any]:
    """Ejecuta el especialista de residuos solo si está pendiente."""
    if not estado["necesita_residuos"] or estado["residuos_completado"]:
        raise RuntimeError("Especialista residuos no requerido o completado.")

    resultado = ejecutar_especialista(estado, rol="residuos")
    estado_parcial = {**estado, **resultado}
    if (
        not estado["necesita_puntos"]
        and debe_buscar_puntos_automaticamente(estado_parcial)
    ):
        resultado["necesita_puntos"] = True
        resultado["log"] = agregar_evento(
            resultado["log"], nodo="residuos",
            accion="puntos_requeridos",
            detalle=(
                "El RAG ha documentado un destino de punto limpio "
                "y existen coordenadas voluntarias."
            ),
        )
    return resultado


def puntos(estado: EstadoGrafo) -> dict[str, Any]:
    """Ejecuta el especialista de puntos solo si está pendiente."""
    if not estado["necesita_puntos"] or estado["puntos_completado"]:
        raise RuntimeError("Especialista puntos no requerido o completado.")

    return ejecutar_especialista(estado, rol="puntos")


# =========================================================
# EVIDENCIAS Y REQUISITOS DE INFORMACIÓN
# =========================================================


def _requiere_proximidad(estado: EstadoGrafo) -> bool:
    """Detecta una búsqueda espacial sin coordenadas aportadas.

    La modalidad «punto limpio de proximidad» no implica por sí
    sola que se solicite encontrar un punto cercano.
    """
    if not estado["necesita_puntos"] or estado["ubicacion"] is not None:
        return False

    return _consulta_proximidad(estado["pedido"])


def _solo_descripcion(estado: EstadoGrafo) -> bool:
    """Distingue descripción de imagen de clasificación de residuos."""
    if not estado.get("ruta_imagen"):
        return False

    pedido = normalizar_texto(estado["pedido"])
    describe = re.search(
        r"\b(?:describe|descripcion|identifica|que ves)\b",
        pedido,
    )
    clasifica = re.search(
        r"\b(?:recicl\w*|tir\w*|contenedor\w*|"
        r"deposit\w*|destino\w*|clasific\w*|separ\w*)\b",
        pedido,
    )
    return bool(describe and not clasifica)


def _evidencia_suficiente(estado: EstadoGrafo,) -> tuple[str, str | None]:
    """Distingue evidencia suficiente, parcial y ausente."""
    soporte = {"residuos": False, "puntos": False}
    motivo_abstencion = None

    for evidencia in estado["evidencias"]:
        if not isinstance(evidencia, dict):
            continue

        rol = evidencia.get("rol")
        herramienta = evidencia.get("herramienta")
        datos = evidencia.get("datos")

        if rol not in soporte:
            continue

        if herramienta == "consultar_rag" and isinstance(datos, dict):
            if datos.get("abstencion") is True:
                motivo_abstencion = motivo_abstencion or datos.get("motivo")
            elif (
                datos.get("abstencion") is False
                and isinstance(datos.get("respuesta"), str)
                and bool(datos["respuesta"].strip())
                and isinstance(datos.get("fuentes"), list)
                and bool(datos["fuentes"])
            ):
                soporte[rol] = True

        elif herramienta == "buscar_puntos_limpios_cercanos":
            if (
                rol == "puntos"
                and isinstance(datos, dict)
                and datos.get("fuente") in {"api", "corpus_local"}
                and isinstance(datos.get("puntos"), list)
                and type(datos.get("total_encontrados")) is int
            ):
                # Cero resultados es una búsqueda válida, no un hallazgo.
                soporte["puntos"] = True

        elif herramienta == "describir_imagen":
            if (
                rol == "residuos"
                and _solo_descripcion(estado)
                and isinstance(datos, str)
                and bool(datos.strip())
            ):
                soporte["residuos"] = True

    if _requiere_proximidad(estado):
        soporte["puntos"] = False

    requeridos = [rol for rol in soporte if estado[f"necesita_{rol}"]]
    encontrados = sum(soporte[rol] for rol in requeridos)

    if requeridos and encontrados == len(requeridos):
        return "suficiente", None

    motivo = motivo_abstencion or "evidencia_insuficiente"
    if encontrados:
        return "parcial", motivo

    return "insuficiente", motivo


def _hay_fallo_bloqueante(estado: EstadoGrafo) -> bool:
    """Reconoce errores técnicos sin confundirlos con datos ausentes."""
    if estado.get("error"):
        return True

    for traza in estado["traza_tools"]:
        if traza.get("status") != "error":
            continue
        if traza.get("tool") == "hora_actual":
            continue  # Fallo auxiliar: conservar evidencia RAG/API.
        if (
            traza.get("tool") == "buscar_puntos_limpios_cercanos"
            and estado["ubicacion"] is None
        ):
            continue
        if (
            traza.get("tool") == "describir_imagen"
            and not estado["ruta_imagen"]
        ):
            continue
        return True

    return False


# =========================================================
# SÍNTESIS SIN NUEVA GENERACIÓN
# =========================================================


def _sintesis_directa(
    estado: EstadoGrafo,
    evidencia: str,
) -> tuple[str, list[str]] | None:
    """Utiliza datos íntegros de una única herramienta verificable.

    No combina evidencias distintas ni reformula clasificaciones del RAG.
    Los casos complejos continúan utilizando el prompt de síntesis.
    """
    if evidencia != "suficiente" or _hay_fallo_bloqueante(estado):
        return None

    principales = [
        e for e in estado["evidencias"]
        if isinstance(e, dict) and e.get("herramienta") != "hora_actual"
    ]
    if len(principales) != 1:
        return None

    registro = principales[0]
    if not isinstance(registro, dict):
        return None

    rol = registro.get("rol")
    herramienta = registro.get("herramienta")
    datos = registro.get("datos")

    if (
        herramienta == "consultar_rag"
        and isinstance(datos, dict)
        and datos.get("abstencion") is False
        and isinstance(datos.get("respuesta"), str)
        and datos["respuesta"].strip()
        and isinstance(datos.get("fuentes"), list)
        and datos["fuentes"]
        and not estado.get("ruta_imagen")
    ):
        if rol == "residuos" and not estado["necesita_puntos"]:
            plan = [
                "Comprobar que el objeto y las condiciones descritas "
                "coinciden con la documentación citada.",
                "Seguir las indicaciones documentadas aplicables "
                "al residuo, sin asumir condiciones no verificadas.",
            ]
            return datos["respuesta"].strip(), plan

        if rol == "puntos" and not estado["necesita_residuos"]:
            plan = [
                "Revisar las condiciones y los datos documentados "
                "en las fuentes citadas.",
                "Confirmar cualquier dato variable antes de acudir "
                "a una instalación de recogida.",
            ]
            return datos["respuesta"].strip(), plan

    if (
        herramienta == "describir_imagen"
        and rol == "residuos"
        and isinstance(datos, str)
        and datos.strip()
        and _solo_descripcion(estado)
        and not estado["necesita_puntos"]
    ):
        return datos.strip(), [
            "Comprobar que la descripción coincide con la imagen.",
            "Consultar el RAG antes de asignar un contenedor "
            "a cualquiera de los objetos descritos.",
        ]

    if (
        herramienta == "buscar_puntos_limpios_cercanos"
        and rol == "puntos"
        and not estado["necesita_residuos"]
        and isinstance(datos, dict)
        and estado["ubicacion"] is not None
    ):
        return _sintesis_geografica(datos)

    return None


def _sintesis_geografica(
    datos: dict[str, Any],
) -> tuple[str, list[str]] | None:
    """Presenta únicamente puntos y distancias devueltos por la tool."""
    fuente = datos.get("fuente")
    puntos = datos.get("puntos")
    total = datos.get("total_encontrados")
    radio = datos.get("radio_metros")

    if (
        fuente not in {"api", "corpus_local"}
        or not isinstance(puntos, list)
        or type(total) is not int
        or total < 0
        or type(radio) is not int
        or radio <= 0
    ):
        return None

    procedencia = "API municipal" if fuente == "api" else "corpus local"

    if total == 0:
        if puntos:
            return None
        respuesta = (
            "No se encontraron puntos limpios dentro del radio "
            f"consultado de {radio} m. Fuente: {procedencia}."
        )
    else:
        if not puntos:
            return None
        lineas = []
        for punto in puntos:
            if not isinstance(punto, dict):
                return None

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
                return None

            lineas.append(f"- {tipo}: {direccion} ({distancia:.0f} m).")

        respuesta = (
            f"Puntos limpios encontrados en un radio de {radio} m "
            f"(fuente: {procedencia}):\n" + "\n".join(lineas)
        )

    if fuente == "corpus_local":
        respuesta += "\nDatos del corpus local; pueden estar desactualizados."

    plan = [
        "Comprobar la dirección y el tipo de punto mostrado.",
        "Verificar el horario y las condiciones del centro "
        "antes de desplazarse; no constan como abiertos ahora.",
    ]
    return respuesta, plan


def _respuesta_sin_evidencia(estado: EstadoGrafo) -> str:
    """Prioriza aclaraciones documentadas sin inventar respuestas."""
    if _requiere_proximidad(estado):
        return (
            "Para buscar puntos limpios cercanos, facilita "
            "voluntariamente latitud y longitud."
        )

    for registro in estado["evidencias"]:
        if not isinstance(registro, dict):
            continue
        if registro.get("herramienta") != "consultar_rag":
            continue

        datos = registro.get("datos")
        if (
            isinstance(datos, dict)
            and datos.get("abstencion") is True
            and isinstance(datos.get("respuesta"), str)
            and datos["respuesta"].strip()
        ):
            return datos["respuesta"].strip()

    return (
        "No dispongo de información suficiente para resolver "
        "esta consulta con datos verificables."
    )


# =========================================================
# NODO 4 — SINTETIZAR
# =========================================================


def sintetizar(estado: EstadoGrafo) -> dict[str, Any]:
    """Prepara respuesta y plan sin duplicar generaciones del RAG."""
    for rol in ("residuos", "puntos"):
        if estado[f"necesita_{rol}"] and not estado[f"{rol}_completado"]:
            raise RuntimeError("No se puede sintetizar: faltan especialistas.")

    if estado["siguiente"] != "sintetizar":
        raise RuntimeError("Síntesis fuera de la ruta autorizada.")

    evidencia, motivo = _evidencia_suficiente(estado)
    limitaciones = deepcopy(estado["limitaciones"])
    modo = "directo"

    if evidencia == "insuficiente":
        if _hay_fallo_bloqueante(estado):
            raise RuntimeError(
                "No se pudo obtener evidencia por un fallo técnico."
            )

        respuesta = _respuesta_sin_evidencia(estado)
        plan = [
            "Aclarar los datos necesarios para responder la consulta.",
            "Volver a consultar y verificar el resultado con sus fuentes.",
        ]

    else:
        resultado = _sintesis_directa(estado, evidencia)

        if resultado is not None:
            respuesta, plan = resultado
        else:
            modo = "llm"
            limite_nota = MAX_CONTEXTO_CARACTERES // 2
            entrada = {
                "pedido": estado["pedido"],
                "ubicacion_disponible": estado["ubicacion"] is not None,
                "notas_residuos": estado["notas_residuos"][:limite_nota],
                "notas_puntos": estado["notas_puntos"][:limite_nota],
                "fuentes": estado["fuentes"],
                "limitaciones": limitaciones,
                "estado_evidencia": evidencia,
            }

            try:
                propuesta = _generar_json(PROMPT_SINTESIS, entrada)
                respuesta = propuesta.get("respuesta")
                plan = propuesta.get("plan")
                if (
                    not isinstance(respuesta, str)
                    or not respuesta.strip()
                    or not isinstance(plan, list)
                    or len(plan) < 2
                    or any(
                        not isinstance(paso, str) or not paso.strip()
                        for paso in plan
                    )
                ):
                    raise ValueError("Respuesta o plan inválidos.")
            except Exception as exc:
                raise RuntimeError(
                    "No se pudo generar un plan válido para el HITL."
                ) from exc

    anexo, fuentes_horario = anexos_temporales(estado)
    tiene_geo = any(
        isinstance(e, dict)
        and e.get("herramienta") == "buscar_puntos_limpios_cercanos"
        and isinstance(e.get("datos"), dict)
        for e in estado["evidencias"]
    )
    if anexo and tiene_geo:
        # Evitar una recomendación LLM contradictoria con la
        # priorización temporal comprobada por Python.
        clasificaciones = [
            e["datos"]["respuesta"].strip()
            for e in estado["evidencias"]
            if isinstance(e, dict)
            and e.get("rol") == "residuos"
            and e.get("herramienta") == "consultar_rag"
            and isinstance(e.get("datos"), dict)
            and e["datos"].get("abstencion") is False
            and isinstance(e["datos"].get("respuesta"), str)
            and e["datos"].get("fuentes")
        ]
        if clasificaciones:
            respuesta = "\n\n".join(clasificaciones)
        elif not estado["necesita_residuos"]:
            respuesta = ""
    if anexo:
        respuesta = (respuesta.rstrip() + "\n\n" + anexo).strip()

    return {
        "respuesta": respuesta.strip(),
        "fuentes": [*estado["fuentes"], *fuentes_horario],
        "plan": [paso.strip() for paso in plan],
        "abstencion": evidencia == "insuficiente",
        "motivo": motivo,
        "estado_evidencia": evidencia,
        "limitaciones": limitaciones,
        "plan_aprobado": None,
        "pendiente_hitl": False,
        "done": False,
        "log": agregar_evento(
            estado["log"],
            nodo="sintetizar",
            accion="plan_preparado",
            detalle=f"Evidencia: {evidencia}; síntesis: {modo}.",
        ),
    }


# =========================================================
# NODO 5 — APLICAR
# =========================================================


def aplicar(estado: EstadoGrafo) -> dict[str, Any]:
    """Registra la aprobación o el rechazo, sin acciones externas."""
    aprobado = estado.get("plan_aprobado")
    if type(aprobado) is not bool:
        raise RuntimeError("Aplicación sin decisión humana válida.")

    historial = registrar_turno(estado, aprobado=aprobado)
    accion = "aprobado" if aprobado else "rechazado"

    cambios: dict[str, Any] = {
        "done": aprobado,
        "pendiente_hitl": False,
        "historial_turnos": historial,
        "log": agregar_evento(
            estado["log"],
            nodo="aplicar",
            accion=accion,
            detalle="Decisión humana registrada; sin acciones externas.",
        ),
    }

    if not aprobado:
        cambios.update({
            "contexto_revision": (
                "Plan rechazado. Consulta original: "
                + estado["pedido_original"]
            ),
            "plan": [],
            "respuesta": (
                "Plan rechazado. Puedes reformular tu consulta."
            ),
            "evidencias": [],
            "fuentes": [],
            "notas_residuos": "",
            "notas_puntos": "",
            "estado_evidencia": "insuficiente",
            "abstencion": True,
            "motivo": None,
        })

    return cambios
