"""Orquestador público del sistema multiagente ReciclaTIA.

Centraliza la ejecución de turnos y la decisión humana sobre el plan.

CLI y Streamlit deben utilizar estas funciones, sin invocar
directamente los nodos ni manipular checkpoints por su cuenta.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from src.graph import (
    config_hilo,
    get_app,
    hay_hitl_pendiente,
)
from src.input_validation import validar_pregunta
from src.state import (
    EstadoProducto,
    Ubicacion,
    crear_estado_producto,
    estado_desde_grafo,
    preparar_nuevo_turno,
)


# =========================================================
# IDENTIFICADOR DE CONVERSACIÓN
# =========================================================

def nuevo_thread_id() -> str:
    """Genera un identificador único para una conversación."""

    return str(uuid4())


# =========================================================
# VALIDACIÓN DEL HILO
# =========================================================

def _validar_hilo(
    estado: EstadoProducto,
    thread_id: str,
) -> dict[str, Any]:
    """Valida la asociación entre el estado y la conversación."""

    configuracion = config_hilo(thread_id)

    if not isinstance(estado, dict):
        raise ValueError("Estado de conversación inválido.")

    identificador = estado.get("thread_id", "")

    if identificador and identificador != thread_id:
        raise ValueError(
            "El estado pertenece a otra conversación."
        )

    return configuracion


# =========================================================
# LECTURA DEL CHECKPOINT
# =========================================================

def _valores_checkpoint(snapshot: Any) -> Mapping[str, Any]:
    """Extrae los valores guardados sin asumir que existe un hilo."""

    if snapshot is None:
        return {}

    valores = getattr(snapshot, "values", None)

    return valores if isinstance(valores, Mapping) else {}


def _estado_actual(
    thread_id: str,
    base: EstadoProducto,
) -> EstadoProducto:
    """Construye el estado público desde el checkpoint real."""

    app = get_app()
    snapshot = app.get_state(config_hilo(thread_id))

    return estado_desde_grafo(
        _valores_checkpoint(snapshot),
        base,
        thread_id=thread_id,
        pendiente_hitl=hay_hitl_pendiente(thread_id),
    )


# =========================================================
# PROCESAMIENTO DE TURNOS
# =========================================================

def procesar_turno(
    estado: EstadoProducto,
    mensaje: str,
    thread_id: str,
    *,
    max_steps: int | None = None,
    ubicacion: Ubicacion | None = None,
    ruta_imagen: str | None = None,
) -> EstadoProducto:
    """Ejecuta una consulta hasta el HITL o un fallo técnico.

    Los turnos se contabilizan por mensajes válidos del usuario.
    No se permite iniciar otro turno mientras hay una pausa.
    """

    configuracion = _validar_hilo(estado, thread_id)

    # Reutilizamos la validación existente del RAG.
    # Una entrada vacía o inválida no ejecuta el grafo.
    mensaje = validar_pregunta(mensaje)

    app = get_app()
    snapshot = app.get_state(configuracion)

    # -----------------------------------------------------
    # COMPROBAR EL ESTADO PERSISTIDO DEL HILO
    # -----------------------------------------------------

    if hay_hitl_pendiente(thread_id):
        raise ValueError(
            "Hay un plan pendiente de aprobación o rechazo."
        )

    if snapshot is not None and snapshot.next:
        raise RuntimeError(
            "La ejecución anterior está incompleta. "
            "Revisa el estado del hilo antes de continuar."
        )

    # El checkpoint tiene prioridad sobre el estado
    # proporcionado por la interfaz.
    estado_base = estado_desde_grafo(
        _valores_checkpoint(snapshot),
        estado,
        thread_id=thread_id,
        pendiente_hitl=False,
    )

    if estado_base["done"]:
        raise ValueError(
            "La conversación ha finalizado. "
            "Inicia una nueva para realizar otra consulta."
        )

    # -----------------------------------------------------
    # PREPARAR UN TURNO INDEPENDIENTE
    # -----------------------------------------------------

    nuevo_estado = preparar_nuevo_turno(
        estado_base,
        mensaje,
        max_steps=max_steps,
        ubicacion=ubicacion,
        ruta_imagen=ruta_imagen,
    )

    # -----------------------------------------------------
    # EJECUTAR LANGGRAPH
    # -----------------------------------------------------

    try:
        app.invoke(
            nuevo_estado,
            config=configuracion,
        )

    except Exception:
        # No damos por aprobado ni completado un plan
        # cuando el grafo ha sufrido un fallo.
        resultado = _estado_actual(
            thread_id,
            estado_base,
        )

        resultado["error"] = (
            "Error técnico durante la ejecución del agente. "
            "No se ha completado el turno. "
            "Revisa la traza o inicia una nueva conversación."
        )

        if not resultado["pendiente_hitl"]:
            resultado["plan"] = []
            resultado["done"] = False

        return resultado

    # -----------------------------------------------------
    # CONFIRMAR LA PAUSA REAL
    # -----------------------------------------------------

    resultado = _estado_actual(
        thread_id,
        estado_base,
    )

    if not resultado["pendiente_hitl"]:
        resultado["error"] = (
            "El grafo terminó sin alcanzar "
            "la interrupción HITL prevista."
        )
        resultado["plan"] = []
        resultado["done"] = False

        return resultado

    if (
        not isinstance(resultado["respuesta"], str)
        or not resultado["respuesta"].strip()
        or not isinstance(resultado["plan"], list)
        or len(resultado["plan"]) < 2
    ):
        resultado["error"] = (
            "El grafo está interrumpido, pero "
            "no dispone de un plan válido."
        )

        return resultado

    # El nodo aplicar aún no ha sido ejecutado.
    resultado["done"] = False

    return resultado


# =========================================================
# APROBACIÓN O RECHAZO HUMANO
# =========================================================

def aprobar_plan(
    thread_id: str,
    aprobado: bool,
    estado: EstadoProducto,
) -> EstadoProducto:
    """Registra la decisión humana y reanuda el checkpoint.

    La aprobación solo puede producirse si LangGraph
    está realmente detenido antes del nodo aplicar.
    """

    configuracion = _validar_hilo(estado, thread_id)

    if type(aprobado) is not bool:
        raise ValueError(
            "La decisión debe ser True o False."
        )

    app = get_app()

    # -----------------------------------------------------
    # VERIFICAR INTERRUPCIÓN Y PLAN
    # -----------------------------------------------------

    if not hay_hitl_pendiente(thread_id):
        raise ValueError(
            "No existe un plan pendiente de decisión humana."
        )

    snapshot = app.get_state(configuracion)
    datos = _valores_checkpoint(snapshot)

    plan = datos.get("plan")

    if (
        not isinstance(plan, list)
        or len(plan) < 2
        or any(
            not isinstance(paso, str)
            or not paso.strip()
            for paso in plan
        )
    ):
        raise RuntimeError(
            "El checkpoint no contiene un plan válido."
        )

    # Evita que una interfaz desactualizada apruebe
    # un plan distinto del almacenado en LangGraph.
    if estado.get("plan") != plan:
        raise ValueError(
            "El plan mostrado no coincide con "
            "el estado actual de la conversación."
        )

    if datos.get("done") is True:
        raise ValueError(
            "El plan ya figura como completado."
        )

    # -----------------------------------------------------
    # ACTUALIZAR EL CHECKPOINT Y REANUDAR
    # -----------------------------------------------------

    try:
        app.update_state(
            configuracion,
            {"plan_aprobado": aprobado},
        )

        # Continúa desde la interrupción estática
        # anterior a aplicar, sin empezar un turno nuevo.
        app.invoke(
            None,
            config=configuracion,
        )

    except Exception as exc:
        raise RuntimeError(
            "No se pudo reanudar LangGraph "
            "después de la decisión humana."
        ) from exc

    # -----------------------------------------------------
    # VERIFICAR QUE APLICAR SE EJECUTÓ
    # -----------------------------------------------------

    snapshot_final = app.get_state(configuracion)

    resultado = _estado_actual(
        thread_id,
        estado,
    )

    historial = resultado.get("historial_turnos", [])

    decision_registrada = bool(
        historial
        and historial[-1].get("aprobado") is aprobado
    )

    if (
        snapshot_final is None
        or snapshot_final.next
        or resultado["pendiente_hitl"]
        or resultado["done"] is not aprobado
        or not decision_registrada
    ):
        resultado["error"] = (
            "No se pudo confirmar la aplicación "
            "de la decisión humana."
        )

        return resultado

    return resultado


# =========================================================
# DEMOSTRACIÓN
# =========================================================

def run_demo(
    mensaje: str = "¿Dónde tiro una botella de plástico?",
    *,
    auto_aprobar: bool = True,
    ubicacion: Ubicacion | None = None,
    ruta_imagen: str | None = None,
) -> EstadoProducto:
    """Ejecuta una demostración usando la API pública del agente.

    Con auto_aprobar=False devuelve el plan pendiente de HITL.
    Con auto_aprobar=True registra una aprobación explícita
    solicitada por el modo demo.
    """

    if type(auto_aprobar) is not bool:
        raise ValueError(
            "auto_aprobar debe ser True o False."
        )

    thread_id = nuevo_thread_id()

    estado = crear_estado_producto(
        thread_id=thread_id,
    )

    resultado = procesar_turno(
        estado,
        mensaje,
        thread_id,
        ubicacion=ubicacion,
        ruta_imagen=ruta_imagen,
    )

    if auto_aprobar and resultado["pendiente_hitl"]:
        resultado = aprobar_plan(
            thread_id,
            True,
            resultado,
        )

    return resultado
