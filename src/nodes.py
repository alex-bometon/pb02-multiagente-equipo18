"""Nodos del grafo multiagente ReciclaTIA.

El cableado de StateGraph pertenece a src.graph.
La autorización y reanudación del HITL pertenecen a src.agent.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from config import MAX_CONTEXTO_CARACTERES
from src.agent_prompts import PROMPT_SUPERVISOR, PROMPT_SINTESIS
from src.agent_routing import resolver_enrutamiento
from src.agent_tool_loop import ejecutar_especialista
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
    """Invoca Gemini y obtiene un objeto JSON."""

    mensajes = [
        SystemMessage(content=instrucciones),
        HumanMessage(
            content=json.dumps(
                entrada,
                ensure_ascii=False,
            )
        ),
    ]

    respuesta = get_llm(
        supervisor=supervisor_llm
    ).invoke(mensajes)

    datos = JsonOutputParser().invoke(respuesta)

    if not isinstance(datos, dict):
        raise ValueError(
            "El modelo no ha devuelto un objeto JSON."
        )

    return datos


# =========================================================
# NODO 1 — SUPERVISOR
# =========================================================

def supervisor(estado: EstadoGrafo) -> dict[str, Any]:
    """Selecciona el siguiente nodo mediante decisión validada."""

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

    fallo_llm = False

    try:
        propuesta = _generar_json(
            PROMPT_SUPERVISOR,
            entrada,
            supervisor_llm=True,
        )

    except Exception:
        propuesta = None
        fallo_llm = True

    # Python decide la ruta ejecutable.
    decision = resolver_enrutamiento(
        estado,
        propuesta,
    )

    accion = (
        "fallback"
        if fallo_llm or decision["corregida"]
        else "ruta"
    )

    log = agregar_evento(
        estado["log"],
        nodo="supervisor",
        accion=accion,
        detalle=(
            f"Siguiente: {decision['siguiente']}; "
            f"{decision['motivo_correccion'] or 'ruta validada'}"
        ),
    )

    return {
        "necesita_residuos": decision["necesita_residuos"],
        "necesita_puntos": decision["necesita_puntos"],
        "siguiente": decision["siguiente"],
        "iteracion": estado["iteracion"] + 1,
        "log": log,
    }


# =========================================================
# NODO 2 — ESPECIALISTA DE RESIDUOS
# =========================================================

def residuos(estado: EstadoGrafo) -> dict[str, Any]:
    """Ejecuta el especialista de residuos cuando corresponde."""

    if (
        not estado["necesita_residuos"]
        or estado["residuos_completado"]
    ):
        raise RuntimeError(
            "Especialista residuos no requerido o completado."
        )

    return ejecutar_especialista(
        estado,
        rol="residuos",
    )


# =========================================================
# NODO 3 — ESPECIALISTA DE PUNTOS LIMPIOS
# =========================================================

def puntos(estado: EstadoGrafo) -> dict[str, Any]:
    """Ejecuta el especialista geográfico cuando corresponde."""

    if (
        not estado["necesita_puntos"]
        or estado["puntos_completado"]
    ):
        raise RuntimeError(
            "Especialista puntos no requerido o completado."
        )

    return ejecutar_especialista(
        estado,
        rol="puntos",
    )


# =========================================================
# EVALUACIÓN DE EVIDENCIAS
# =========================================================

def _evidencia_suficiente(
    estado: EstadoGrafo,
) -> tuple[str, str | None]:
    """Evalúa qué especialistas tienen resultados verificables."""

    soporte = {
        "residuos": False,
        "puntos": False,
    }

    motivo_abstencion = None
    pedido = normalizar_texto(estado["pedido"])

    # Una imagen puede bastar para describir objetos.
    # Para clasificarlos se sigue necesitando el RAG.
    solo_descripcion = bool(
        estado.get("ruta_imagen")
        and re.search(
            r"\b(?:describe|descripcion|identifica|que ves)\b",
            pedido,
        )
        and not re.search(
            r"\b(?:recicl\w*|tir\w*|contenedor\w*|"
            r"deposit\w*|destino\w*|clasific\w*|separ\w*)\b",
            pedido,
        )
    )

    for evidencia in estado["evidencias"]:

        if not isinstance(evidencia, dict):
            continue

        rol = evidencia.get("rol")
        herramienta = evidencia.get("herramienta")
        datos = evidencia.get("datos")

        if rol not in soporte:
            continue

        # Evidencia documental del RAG.
        if (
            herramienta == "consultar_rag"
            and isinstance(datos, dict)
        ):

            if datos.get("abstencion") is True:
                if motivo_abstencion is None:
                    motivo_abstencion = datos.get("motivo")

            elif (
                datos.get("abstencion") is False
                and isinstance(datos.get("respuesta"), str)
                and bool(datos["respuesta"].strip())
                and isinstance(datos.get("fuentes"), list)
                and bool(datos["fuentes"])
            ):
                soporte[rol] = True

        # Evidencia geográfica.
        elif (
            herramienta == "buscar_puntos_limpios_cercanos"
            and rol == "puntos"
        ):

            if (
                isinstance(datos, dict)
                and datos.get("fuente") in {
                    "api",
                    "corpus_local",
                }
                and isinstance(datos.get("puntos"), list)
                and type(datos.get("total_encontrados")) is int
            ):
                # Una búsqueda válida sin coincidencias
                # también es un resultado verificable.
                soporte[rol] = True

        # Evidencia visual solo para descripción.
        elif (
            herramienta == "describir_imagen"
            and rol == "residuos"
            and solo_descripcion
            and isinstance(datos, str)
            and bool(datos.strip())
        ):
            soporte["residuos"] = True

    # Una explicación general del RAG no sustituye
    # una búsqueda espacial si faltan coordenadas.
    if (
        estado["necesita_puntos"]
        and estado["ubicacion"] is None
        and re.search(
            r"\b(?:cerca|cercan\w*|proximos?|"
            r"proximas?|alrededor|localiza\w*)\b",
            pedido,
        )
    ):
        soporte["puntos"] = False

    requeridos = [
        rol
        for rol in soporte
        if estado[f"necesita_{rol}"]
    ]

    encontrados = sum(
        soporte[rol]
        for rol in requeridos
    )

    if requeridos and encontrados == len(requeridos):
        return "suficiente", None

    if encontrados:
        return (
            "parcial",
            motivo_abstencion or "evidencia_insuficiente",
        )

    return (
        "insuficiente",
        motivo_abstencion or "evidencia_insuficiente",
    )


# =========================================================
# NODO 4 — SINTETIZAR
# =========================================================

def sintetizar(estado: EstadoGrafo) -> dict[str, Any]:
    """Combina evidencias y prepara el plan para el HITL."""

    # No sintetizar con especialistas pendientes.
    for rol in ("residuos", "puntos"):
        if (
            estado[f"necesita_{rol}"]
            and not estado[f"{rol}_completado"]
        ):
            raise RuntimeError(
                "No se puede sintetizar: "
                "faltan especialistas."
            )

    if estado["siguiente"] != "sintetizar":
        raise RuntimeError(
            "Síntesis fuera de la ruta autorizada."
        )

    evidencia, motivo = _evidencia_suficiente(estado)
    limitaciones = deepcopy(estado["limitaciones"])

    # -----------------------------------------------------
    # SIN EVIDENCIA: ACLARACIÓN SEGURA
    # -----------------------------------------------------

    if evidencia == "insuficiente":

        # Distinguir datos ausentes de fallos técnicos.
        error_bloqueante = bool(estado.get("error"))

        for traza in estado["traza_tools"]:
            if traza.get("status") != "error":
                continue

            if (
                traza.get("tool")
                == "buscar_puntos_limpios_cercanos"
                and estado["ubicacion"] is None
            ):
                continue

            if (
                traza.get("tool") == "describir_imagen"
                and not estado["ruta_imagen"]
            ):
                continue

            error_bloqueante = True

        if error_bloqueante:
            raise RuntimeError(
                "No se pudo obtener evidencia "
                "por un fallo técnico."
            )

        respuesta = (
            "No dispongo de información suficiente para "
            "resolver la consulta con datos verificables. "
            "Si solicitas puntos cercanos, facilita "
            "voluntariamente latitud y longitud."
        )

        plan = [
            (
                "Aclarar el residuo, el tipo de consulta "
                "o los datos que faltan."
            ),
            (
                "Repetir la consulta y contrastar los "
                "resultados con las fuentes."
            ),
        ]

    # -----------------------------------------------------
    # EVIDENCIA SUFICIENTE O PARCIAL: GENERACIÓN
    # -----------------------------------------------------

    else:
        limite_nota = MAX_CONTEXTO_CARACTERES // 2

        entrada = {
            "pedido": estado["pedido"],
            "ubicacion_disponible": (
                estado["ubicacion"] is not None
            ),
            "notas_residuos": (
                estado["notas_residuos"][:limite_nota]
            ),
            "notas_puntos": (
                estado["notas_puntos"][:limite_nota]
            ),
            "fuentes": estado["fuentes"],
            "limitaciones": limitaciones,
            "estado_evidencia": evidencia,
        }

        try:
            propuesta = _generar_json(
                PROMPT_SINTESIS,
                entrada,
            )

            respuesta = propuesta.get("respuesta")
            plan = propuesta.get("plan")

            if (
                not isinstance(respuesta, str)
                or not respuesta.strip()
                or not isinstance(plan, list)
                or len(plan) < 2
                or any(
                    not isinstance(paso, str)
                    or not paso.strip()
                    for paso in plan
                )
            ):
                raise ValueError(
                    "Respuesta o plan inválidos."
                )

        except Exception as exc:
            raise RuntimeError(
                "No se pudo generar un plan válido "
                "para el HITL."
            ) from exc

    # -----------------------------------------------------
    # RESULTADO DE LA SÍNTESIS
    # -----------------------------------------------------

    return {
        "respuesta": respuesta.strip(),
        "plan": [
            paso.strip()
            for paso in plan
        ],
        "abstencion": evidencia != "suficiente",
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
            detalle=(
                f"Estado de evidencia: {evidencia}"
            ),
        ),
    }


# =========================================================
# NODO 5 — APLICAR
# =========================================================

def aplicar(estado: EstadoGrafo) -> dict[str, Any]:
    """Registra la decisión humana sin ejecutar acciones externas."""

    aprobado = estado.get("plan_aprobado")

    if type(aprobado) is not bool:
        raise RuntimeError(
            "Aplicación sin decisión humana válida."
        )

    # Registrar el turno antes de limpiar un plan rechazado.
    historial = registrar_turno(
        estado,
        aprobado=aprobado,
    )

    accion = (
        "aprobado"
        if aprobado
        else "rechazado"
    )

    cambios: dict[str, Any] = {
        "done": aprobado,
        "pendiente_hitl": False,
        "historial_turnos": historial,
        "log": agregar_evento(
            estado["log"],
            nodo="aplicar",
            accion=accion,
            detalle=(
                "Decisión humana registrada; "
                "sin acciones externas."
            ),
        ),
    }

    # El rechazo permite un nuevo turno en el mismo hilo.
    if not aprobado:
        cambios.update({
            "contexto_revision": (
                "Plan rechazado. Consulta original: "
                + estado["pedido_original"]
            ),
            "plan": [],
            "respuesta": (
                "Plan rechazado. "
                "Puedes reformular tu consulta."
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
