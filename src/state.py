"""Contratos de estado compartidos por LangGraph, CLI y Streamlit.

No ejecuta herramientas, no llama al LLM y no decide rutas ni HITL.
Las listas se sobrescriben: no utilizar reducers de concatenación.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

from config import (
    MAX_HISTORY_TURNS,
    MAX_STEPS_DEFAULT,
    MAX_STEPS_LIMIT,
    MAX_TURNS_DEFAULT,
    MAX_TURNS_LIMIT,
)


# =========================================================
# TIPOS DE DATOS COMPARTIDOS
# =========================================================

class Ubicacion(TypedDict):
    """Coordenadas aportadas por el usuario."""

    latitud: float
    longitud: float


EstadoEvidencia = Literal[
    "suficiente",
    "parcial",
    "insuficiente",
]


class EventoLog(TypedDict):
    nodo: str
    accion: str
    detalle: str


class TrazaTool(TypedDict, total=False):
    nodo: str
    paso: int
    tool: str
    status: str
    fuente: str | None
    preview: str


class RegistroTurno(TypedDict):
    pedido: str
    respuesta: str
    plan: list[str]
    aprobado: bool


# =========================================================
# ESTADO INTERNO DE LANGGRAPH
# =========================================================

class EstadoGrafo(TypedDict):
    """Información compartida por los nodos del grafo."""

    # Petición e información aportada por el usuario.
    pedido: str
    pedido_original: str
    contexto_revision: str
    ubicacion: Ubicacion | None
    ruta_imagen: str | None

    # Selección y progreso de especialistas.
    necesita_residuos: bool
    necesita_puntos: bool
    residuos_completado: bool
    puntos_completado: bool
    siguiente: str
    iteracion: int

    # Resultados y evidencia.
    notas_residuos: str
    notas_puntos: str
    evidencias: list[dict[str, Any]]
    fuentes: list[dict[str, Any]]
    abstencion: bool
    motivo: str | None
    limitaciones: list[str]
    estado_evidencia: EstadoEvidencia

    # Respuesta y aprobación humana.
    respuesta: str
    plan: list[str]
    plan_aprobado: bool | None
    pendiente_hitl: bool
    done: bool
    error: str | None

    # Límites de ejecución y conversación.
    max_steps: int
    max_turns: int
    turnos_consumidos: int
    tool_calls_turno: int

    # Trazabilidad e historial.
    log: list[EventoLog]
    traza_tools: list[TrazaTool]
    historial_turnos: list[RegistroTurno]


# =========================================================
# ESTADO PÚBLICO: CLI Y STREAMLIT
# =========================================================

class EstadoProducto(EstadoGrafo):
    """Estado visible para los clientes del agente."""

    thread_id: str
    traza: list[EventoLog]


# =========================================================
# VALIDACIÓN DE LÍMITES
# =========================================================

def _validar_limite(
    valor: int,
    maximo: int,
    nombre: str,
) -> int:
    """Comprueba que el límite está dentro del rango permitido."""

    if type(valor) is not int or not 1 <= valor <= maximo:
        raise ValueError(
            f"{nombre} debe ser un entero entre 1 y {maximo}."
        )

    return valor


# =========================================================
# CREACIÓN DEL ESTADO INTERNO
# =========================================================

def crear_estado_grafo(
    pedido: str = "",
    *,
    max_steps: int = MAX_STEPS_DEFAULT,
    max_turns: int = MAX_TURNS_DEFAULT,
    ubicacion: Ubicacion | None = None,
    ruta_imagen: str | None = None,
) -> EstadoGrafo:
    """Crea un estado completo con valores independientes."""

    _validar_limite(max_steps, MAX_STEPS_LIMIT, "max_steps")
    _validar_limite(max_turns, MAX_TURNS_LIMIT, "max_turns")

    return {
        "pedido": pedido,
        "pedido_original": pedido,
        "contexto_revision": "",
        "ubicacion": deepcopy(ubicacion),
        "ruta_imagen": ruta_imagen,

        "necesita_residuos": False,
        "necesita_puntos": False,
        "residuos_completado": False,
        "puntos_completado": False,
        "siguiente": "",
        "iteracion": 0,

        "notas_residuos": "",
        "notas_puntos": "",
        "evidencias": [],
        "fuentes": [],
        "abstencion": True,
        "motivo": None,
        "limitaciones": [],
        "estado_evidencia": "insuficiente",

        "respuesta": "",
        "plan": [],
        "plan_aprobado": None,
        "pendiente_hitl": False,
        "done": False,
        "error": None,

        "max_steps": max_steps,
        "max_turns": max_turns,
        "turnos_consumidos": 0,
        "tool_calls_turno": 0,

        "log": [],
        "traza_tools": [],
        "historial_turnos": [],
    }


# =========================================================
# CREACIÓN DEL ESTADO DE PRODUCTO
# =========================================================

def crear_estado_producto(
    thread_id: str = "",
    *,
    max_turns: int = MAX_TURNS_DEFAULT,
) -> EstadoProducto:
    """Inicializa el estado utilizado por CLI y Streamlit."""

    return EstadoProducto(
        **crear_estado_grafo(max_turns=max_turns),
        thread_id=thread_id,
        traza=[],
    )


# =========================================================
# PREPARACIÓN DE UN NUEVO TURNO
# =========================================================

def preparar_nuevo_turno(
    estado: EstadoGrafo,
    mensaje: str,
    *,
    max_steps: int | None = None,
    ubicacion: Ubicacion | None = None,
    ruta_imagen: str | None = None,
) -> EstadoGrafo:
    """Reinicia los datos del turno sin perder el contexto permitido.

    El orquestador comprobará también la posición real del
    checkpoint de LangGraph antes de aceptar otro mensaje.
    """

    if estado["done"]:
        raise ValueError(
            "La conversación ya ha finalizado."
        )

    if estado["pendiente_hitl"]:
        raise ValueError(
            "Hay un plan pendiente de aprobación o rechazo."
        )

    if estado["turnos_consumidos"] >= estado["max_turns"]:
        raise ValueError(
            "Se ha alcanzado el límite de turnos."
        )

    nuevo = crear_estado_grafo(
        mensaje,
        max_steps=(
            estado["max_steps"]
            if max_steps is None
            else max_steps
        ),
        max_turns=estado["max_turns"],
        ubicacion=ubicacion,
        ruta_imagen=ruta_imagen,
    )

    nuevo["turnos_consumidos"] = (
        estado["turnos_consumidos"] + 1
    )

    nuevo["historial_turnos"] = deepcopy(
        estado["historial_turnos"][-MAX_HISTORY_TURNS:]
    )

    nuevo["contexto_revision"] = estado["contexto_revision"]

    return nuevo


# =========================================================
# HISTORIAL DE CONVERSACIÓN
# =========================================================

def registrar_turno(
    estado: EstadoGrafo,
    *,
    aprobado: bool,
) -> list[RegistroTurno]:
    """Devuelve el historial acotado con el turno resuelto.

    Debe llamarse antes de retirar un plan rechazado.
    """

    registro: RegistroTurno = {
        "pedido": estado["pedido_original"],
        "respuesta": estado["respuesta"],
        "plan": deepcopy(estado["plan"]),
        "aprobado": aprobado,
    }

    return [
        *deepcopy(estado["historial_turnos"]),
        registro,
    ][-MAX_HISTORY_TURNS:]


# =========================================================
# REGISTRO DE EVENTOS
# =========================================================

def agregar_evento(
    log: list[EventoLog],
    *,
    nodo: str,
    accion: str,
    detalle: str = "",
) -> list[EventoLog]:
    """Añade un evento sin modificar la lista del checkpoint."""

    evento: EventoLog = {
        "nodo": nodo,
        "accion": accion,
        "detalle": detalle,
    }

    return [*log, evento]


# =========================================================
# ADAPTACIÓN GRAFO → PRODUCTO
# =========================================================

def estado_desde_grafo(
    resultado: Mapping[str, Any],
    base: EstadoProducto | None = None,
    *,
    thread_id: str | None = None,
    pendiente_hitl: bool | None = None,
) -> EstadoProducto:
    """Adapta los canales conocidos al contrato CLI/Streamlit.

    src.agent debe confirmar el HITL real mediante el checkpoint.
    Esta función no decide aprobaciones ni modifica el grafo.
    """

    nuevo = (
        deepcopy(base)
        if base is not None
        else crear_estado_producto()
    )

    for campo in EstadoGrafo.__annotations__:
        if campo in resultado:
            nuevo[campo] = deepcopy(resultado[campo])

    if thread_id is not None:
        nuevo["thread_id"] = thread_id

    if pendiente_hitl is not None:
        nuevo["pendiente_hitl"] = pendiente_hitl

    nuevo["traza"] = deepcopy(nuevo["log"])

    return nuevo
