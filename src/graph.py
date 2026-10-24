
"""Construcción y persistencia del grafo multiagente ReciclaTIA.

Responsabilidades:
- Registrar los nodos de LangGraph.
- Definir las aristas y el enrutamiento condicional.
- Compilar el grafo con MemorySaver y HITL.
- Proporcionar una instancia compartida y configuración por hilo.

No ejecuta herramientas, no llama a Gemini y no decide aprobaciones.
"""

from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import GRAPH_RECURSION_LIMIT
from src.agent_routing import destino_condicional
from src.nodes import (
    supervisor,
    residuos,
    puntos,
    sintetizar,
    aplicar,
)
from src.state import EstadoGrafo


# =========================================================
# CONSTRUCCIÓN DEL GRAFO
# =========================================================

def build_graph() -> CompiledStateGraph:
    """Construye y compila un grafo independiente.

    Cada llamada crea su propio checkpointer.
    Resulta útil para pruebas aisladas.
    """

    grafo = StateGraph(EstadoGrafo)

    # -----------------------------------------------------
    # REGISTRO DE LOS CINCO NODOS
    # -----------------------------------------------------

    grafo.add_node("supervisor", supervisor)
    grafo.add_node("residuos", residuos)
    grafo.add_node("puntos", puntos)
    grafo.add_node("sintetizar", sintetizar)
    grafo.add_node("aplicar", aplicar)

    # -----------------------------------------------------
    # PUNTO DE ENTRADA
    # -----------------------------------------------------

    grafo.add_edge(START, "supervisor")

    # -----------------------------------------------------
    # ENRUTAMIENTO DEL SUPERVISOR
    # -----------------------------------------------------

    grafo.add_conditional_edges(
        "supervisor",
        destino_condicional,
        path_map={
            "residuos": "residuos",
            "puntos": "puntos",
            "sintetizar": "sintetizar",
            "aplicar": "aplicar",
        },
    )

    # -----------------------------------------------------
    # RETORNO AL SUPERVISOR
    # -----------------------------------------------------

    grafo.add_edge("residuos", "supervisor")
    grafo.add_edge("puntos", "supervisor")
    grafo.add_edge("sintetizar", "supervisor")

    # -----------------------------------------------------
    # FIN DEL FLUJO
    # -----------------------------------------------------

    grafo.add_edge("aplicar", END)

    # -----------------------------------------------------
    # CHECKPOINTER E INTERRUPCIÓN HUMANA
    # -----------------------------------------------------

    return grafo.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["aplicar"],
    )


# =========================================================
# INSTANCIA COMPARTIDA DEL GRAFO
# =========================================================

@lru_cache(maxsize=1)
def get_app() -> CompiledStateGraph:
    """Devuelve el mismo grafo compilado durante el proceso.

    CLI y Streamlit utilizarán esta instancia para mantener
    los checkpoints de sus conversaciones.
    """

    return build_graph()


# =========================================================
# CONFIGURACIÓN DE CONVERSACIONES
# =========================================================

def config_hilo(thread_id: str) -> dict[str, Any]:
    """Prepara la configuración de ejecución de un hilo."""

    if (
        not isinstance(thread_id, str)
        or not thread_id.strip()
    ):
        raise ValueError(
            "thread_id debe ser una cadena no vacía."
        )

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


# =========================================================
# CONSULTA DEL ESTADO HITL
# =========================================================

def hay_hitl_pendiente(thread_id: str) -> bool:
    """Comprueba si el checkpoint está pausado antes de aplicar.

    No utiliza únicamente el flag pendiente_hitl:
    consulta el estado real del grafo.
    """

    estado = get_app().get_state(
        config_hilo(thread_id)
    )

    return (
        estado is not None
        and tuple(estado.next) == ("aplicar",)
    )
