"""Orquestación — COMPLETA procesar_turno (aprobar_plan y run_demo vienen DADOS)."""

import uuid
from typing import Any

import config
from src.graph import (
    config_hilo,
    estado_inicial_grafo,
    get_app,
    hay_interrupt,
)
from src.state import crear_estado, estado_desde_grafo


def _marcar_hitl(estado: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """DADO: si el grafo está pausado ante aplicar, marca pendiente_hitl para la UI."""
    pendiente = hay_interrupt(thread_id)
    estado = dict(estado)
    estado["pendiente_hitl"] = pendiente
    if pendiente:
        estado["done"] = False
        resp = (estado.get("respuesta") or "").rstrip()
        aviso = "\n\n_(Plan listo: aprueba o rechaza antes de cerrar.)_"
        if "aprueba o rechaza" not in resp.lower():
            estado["respuesta"] = resp + aviso
    return estado


def procesar_turno(
    estado: dict[str, Any],
    mensaje: str,
    thread_id: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Un mensaje → invoke del grafo → estado de producto + HITL.

    Puede terminar en pausa HITL (antes de aplicar). El mismo thread_id
    reutiliza el checkpointer MemorySaver.

    TODO [turno]: dentro de try/except (mensaje vacío ya está resuelto arriba del raise):
      1. app = get_app(); cfg = config_hilo(thread_id)
      2. inicial = estado_inicial_grafo(mensaje, max_steps=max_steps)
      3. resultado = app.invoke(inicial, cfg)
      4. nuevo = estado_desde_grafo(resultado, estado)
      5. Si no hay pedido_original → cópialo del mensaje
      6. nuevo = _marcar_hitl(nuevo, thread_id)
      7. Append a traza (turno, mensaje, status ok, done, pendiente_hitl, log, tools)
      8. return nuevo
      En except: error en estado + respuesta amigable + traza error
    """
    mensaje = (mensaje or "").strip()
    if not mensaje:
        nuevo = dict(estado)
        nuevo["respuesta"] = (
            "Escribe un mensaje (zona, contaminante, tipo de consulta…)."
        )
        return nuevo

    # Número de turno para la traza (úsalo al hacer append en el TODO).
    n = len(estado.get("traza") or []) + 1
    raise NotImplementedError(
        "Implementa procesar_turno en src/agent.py "
        f"(pista: el turno actual sería n={n})"
    )


def aprobar_plan(thread_id: str, aprobado: bool, estado: dict[str, Any]) -> dict[str, Any]:
    """DADO: HITL — guarda la decisión en el checkpoint y reanuda el nodo aplicar.

    1) update_state(plan_aprobado=...)
    2) invoke(None) → continúa desde interrupt_before sin input nuevo
    """
    app = get_app()
    cfg = config_hilo(thread_id)
    if not hay_interrupt(thread_id):
        nuevo = dict(estado)
        nuevo["respuesta"] = "No hay plan pendiente de aprobación."
        return nuevo

    app.update_state(cfg, {"plan_aprobado": bool(aprobado)})
    resultado = app.invoke(None, cfg)
    nuevo = estado_desde_grafo(resultado, estado)
    nuevo["pendiente_hitl"] = False
    nuevo.setdefault("traza", []).append(
        {
            "turno": len(estado.get("traza") or []) + 1,
            "mensaje": "HITL:" + ("aprobar" if aprobado else "rechazar"),
            "status": "ok",
            "done": nuevo.get("done"),
            "pendiente_hitl": False,
            "tools": [],
        }
    )
    return nuevo


def nuevo_thread_id() -> str:
    return f"aire-{uuid.uuid4().hex[:8]}"


def run_demo(
    mensajes: list[str] | None = None,
    max_turns: int | None = None,
    max_steps: int | None = None,
    auto_aprobar: bool = True,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """DADO: demo CLI; si hay HITL y auto_aprobar, llama aprobar_plan."""
    msgs = list(mensajes) if mensajes is not None else list(config.DEMO_MENSAJES)
    limit = max_turns if max_turns is not None else config.MAX_TURNS_DEFAULT
    tid = thread_id or nuevo_thread_id()
    estado = crear_estado(msgs[0] if msgs else "")

    for mensaje in msgs[:limit]:
        if estado.get("done") or estado.get("error"):
            break
        estado = procesar_turno(estado, mensaje, tid, max_steps=max_steps)
        if estado.get("pendiente_hitl") and auto_aprobar:
            estado = aprobar_plan(tid, True, estado)

    estado["thread_id"] = tid
    return estado
