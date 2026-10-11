"""Grafo multiagente — COMPLETA los TODOs pequeños (_fallback_siguiente, build_graph).

Nodos (supervisor, especialistas, sintetizar, aplicar) y _router_supervisor vienen DADOS.

Flujo:
  START → supervisor ──┬── diagnostico ────┐
                       ├── practico ───────┼──► supervisor → …
                       └── sintetizar ─────┘
                                  │
                     FINISH → aplicar (interrupt_before) → END

Stack: LangChain (ChatGoogleGenerativeAI + @tool) + LangGraph (MemorySaver).
"""

import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import config
from src.llm import get_llm
from src.tools import TOOLS, ejecutar_tool


def _acumular(actual: list, nuevo: list) -> list:
    """Reducer de LangGraph: concatena listas (no pisa el valor anterior)."""
    return (actual or []) + (nuevo or [])


class EstadoGrafo(TypedDict):
    pedido: str  # mensaje del usuario
    notas_diagnostico: str  # las escribe el especialista diagnostico
    notas_practico: str  # las escribe el especialista practico
    respuesta: str  # texto del plan (sintetizar / aplicar)
    plan: list  # viñetas del plan (≥2 para HITL)
    siguiente: str  # lo elige el supervisor: diagnostico|practico|sintetizar|FINISH
    iteracion: int  # contador del supervisor (tope MAX_ITER)
    log: Annotated[list, _acumular]  # traza legible del orquestador
    traza_tools: Annotated[list, _acumular]  # tools usadas en los especialistas
    max_steps: int  # tope modelo↔tools dentro de un especialista
    plan_aprobado: bool  # lo escribe el humano vía update_state (HITL)
    done: bool  # True solo tras aprobar en aplicar
    pendiente_hitl: bool  # True mientras el grafo está pausado antes de aplicar


OPCIONES = ("diagnostico", "practico", "sintetizar", "FINISH")


def _preview(texto: str, n: int = 120) -> str:
    t = (texto or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _texto_llm(content: Any) -> str:
    """Normaliza ai.content (str o lista de bloques Gemini) a texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        partes: list[str] = []
        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                t = bloque.get("text") or bloque.get("content") or ""
                if t:
                    partes.append(str(t))
            else:
                t = getattr(bloque, "text", None)
                if t:
                    partes.append(str(t))
        return "\n".join(p.strip() for p in partes if p and str(p).strip()).strip()
    return str(content).strip()


def _parse_plan(texto: str) -> list[str]:
    """Extrae viñetas (- / • / *) del texto del plan."""
    items: list[str] = []
    for line in (texto or "").splitlines():
        s = line.strip()
        if s.startswith(("-", "•", "*")):
            items.append(s.lstrip("-•* ").strip())
    return [i for i in items if i]


def _ejecutar_con_tools(
    system: str,
    user: str,
    max_steps: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Mini-bucle agent↔tools dentro de un especialista.

    El LLM puede pedir tools; Python las ejecuta (allowlist).
    Se repite hasta que haya texto final o se agote max_steps.
    """
    llm = get_llm().bind_tools(TOOLS)
    messages: list = [SystemMessage(content=system), HumanMessage(content=user)]
    traza: list[dict[str, Any]] = []
    texto = ""

    for paso in range(1, max(1, max_steps) + 1):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            # Sin tool_calls → el modelo ya respondió en texto
            texto = _texto_llm(ai.content)
            break
        for call in calls:
            nombre = call.get("name") or ""
            args = dict(call.get("args") or {})
            resultado = ejecutar_tool(nombre, args)
            status = "blocked" if "no está permitida" in resultado else (
                "error" if resultado.startswith("Error") else "ok"
            )
            traza.append(
                {
                    "step": paso,
                    "tool": nombre,
                    "args": args,
                    "status": status,
                    "preview": _preview(resultado),
                }
            )
            messages.append(
                ToolMessage(content=resultado, tool_call_id=call.get("id") or nombre)
            )
    else:
        texto = texto or "He parado por max_steps en el especialista."

    return texto, traza


def _fallback_siguiente(estado: EstadoGrafo) -> str:
    """TODO [siguiente]: reglas deterministas si el LLM falla.

    - Sin notas_diagnostico → \"diagnostico\"
    - Sin notas_practico → \"practico\"
    - Sin respuesta → \"sintetizar\"
    - Si no → \"FINISH\"
    """
    raise NotImplementedError("Implementa _fallback_siguiente en src/graph.py")


def nodo_supervisor(estado: EstadoGrafo) -> dict:
    """DADO: orquesta; no escribe el plan; solo elige el siguiente nodo (o FINISH).

    MAX_ITER evita bucles infinitos supervisor ↔ especialistas.
    Si el LLM falla o responde mal → _fallback_siguiente (tu TODO).
    """
    it = int(estado.get("iteracion") or 0) + 1
    if it >= config.MAX_ITER:
        return {
            "iteracion": it,
            "siguiente": "FINISH",
            "log": [f"iter {it}: supervisor → FINISH (MAX_ITER)"],
        }

    llm = get_llm(temperature=0)
    prompt = (
        "Eres el supervisor de un agente de calidad del aire (Madrid).\n"
        "Elige el siguiente rol:\n"
        "- diagnostico: datos / guía / interpretar mediciones (RAG, API)\n"
        "- practico: recomendaciones prácticas (hora, exposición, tips)\n"
        "- sintetizar: une notas en un plan final con viñetas\n"
        "- FINISH: ya hay plan/respuesta lista\n"
        "Responde SOLO una palabra: diagnostico, practico, sintetizar o FINISH.\n"
        "Regla: sin notas_diagnostico → diagnostico; sin notas_practico → practico; "
        "con ambas y sin respuesta → sintetizar; con respuesta → FINISH.\n\n"
        f"pedido: {estado.get('pedido')}\n"
        f"notas_diagnostico: {(estado.get('notas_diagnostico') or '')[:400]}\n"
        f"notas_practico: {(estado.get('notas_practico') or '')[:400]}\n"
        f"respuesta: {(estado.get('respuesta') or '')[:200]}\n"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = _texto_llm(out.content)
        palabra = re.findall(r"[A-Za-z_]+", raw)
        siguiente = (palabra[0] if palabra else "").strip()
        if siguiente not in OPCIONES:
            siguiente = _fallback_siguiente(estado)
    except Exception:
        siguiente = _fallback_siguiente(estado)

    return {
        "iteracion": it,
        "siguiente": siguiente,
        "log": [f"iter {it}: supervisor → {siguiente}"],
    }


def nodo_diagnostico(estado: EstadoGrafo) -> dict:
    """DADO: escribe notas_diagnostico (puede usar tools). No cierra el plan."""
    max_steps = int(estado.get("max_steps") or config.MAX_STEPS_DEFAULT)
    system = (
        "Eres el especialista DIAGNÓSTICO de calidad del aire en Madrid. "
        "Usa tools si ayudan (RAG_buscar_en_guia, API_consultar_calidad_aire, "
        "describir_imagen). Si el usuario habla de imagen, foto, cartel, poster "
        "o captura, llama a describir_imagen (sin ruta: usa la imagen demo). "
        "Escribe notas cortas en español (no el plan final)."
    )
    user = (
        f"Pedido del usuario:\n{estado.get('pedido')}\n\n"
        "Devuelve solo las notas de diagnóstico."
    )
    texto, traza = _ejecutar_con_tools(system, user, max_steps)
    return {
        "notas_diagnostico": texto,
        "traza_tools": traza,
        "log": ["diagnostico: notas listas"],
    }


def nodo_practico(estado: EstadoGrafo) -> dict:
    """DADO: escribe notas_practico (recomendaciones; puede usar tools)."""
    max_steps = int(estado.get("max_steps") or config.MAX_STEPS_DEFAULT)
    system = (
        "Eres el especialista PRÁCTICO (recomendaciones). "
        "Usa tools si ayudan (hora_actual, RAG_buscar_en_guia, API…). "
        "Notas cortas: exposición, horarios, tips. No el plan final."
    )
    user = (
        f"Pedido:\n{estado.get('pedido')}\n\n"
        f"Notas diagnóstico:\n{estado.get('notas_diagnostico') or '(aún no)'}\n\n"
        "Devuelve solo las notas prácticas."
    )
    texto, traza = _ejecutar_con_tools(system, user, max_steps)
    return {
        "notas_practico": texto,
        "traza_tools": traza,
        "log": ["practico: notas listas"],
    }


def nodo_sintetizar(estado: EstadoGrafo) -> dict:
    """DADO: une las notas en respuesta + plan (viñetas). Deja el flujo listo para HITL."""
    llm = get_llm()
    prompt = (
        "Une las notas en un plan breve sobre calidad del aire en español.\n"
        "Incluye al menos 2 pasos como viñetas que empiecen por '- '.\n"
        "Una intro de 1-2 frases y luego las viñetas.\n\n"
        f"Pedido: {estado.get('pedido')}\n"
        f"Diagnóstico: {estado.get('notas_diagnostico')}\n"
        f"Práctico: {estado.get('notas_practico')}\n"
    )
    out = llm.invoke([HumanMessage(content=prompt)])
    texto = _texto_llm(out.content)
    plan = _parse_plan(texto)
    if len(plan) < 2:
        # Fallback mínimo para poder pausar en HITL con algo que mostrar
        plan = [
            p
            for p in (
                (estado.get("notas_diagnostico") or "").split(".")[0].strip(),
                (estado.get("notas_practico") or "").split(".")[0].strip(),
            )
            if p
        ]
        if len(plan) < 2:
            plan = [
                "Revisar datos / guía según la zona pedida",
                "Ajustar exposición o horarios según el contexto",
            ]
        texto = texto or "Plan propuesto:\n" + "\n".join(f"- {p}" for p in plan)
    return {
        "respuesta": texto,
        "plan": plan,
        "pendiente_hitl": True,
        "log": ["sintetizar: plan listo (pendiente HITL)"],
    }


def nodo_aplicar(estado: EstadoGrafo) -> dict:
    """DADO: cierra o resetea según plan_aprobado.

    Solo se ejecuta DESPUÉS de la decisión humana: el grafo pausa con
    interrupt_before; el cliente hace update_state + invoke(None).
    """
    if estado.get("plan_aprobado"):
        prev = (estado.get("respuesta") or "").strip()
        cierre = "Plan aprobado. ¡Listo!"
        return {
            "done": True,
            "pendiente_hitl": False,
            "respuesta": f"{prev}\n\n{cierre}".strip() if prev else cierre,
            "log": ["aplicar: plan aprobado"],
        }
    return {
        "done": False,
        "pendiente_hitl": False,
        "plan_aprobado": False,
        "plan": [],
        "respuesta": (
            "Plan rechazado. Indica qué cambiar (zona, contaminante, tipo) "
            "y proponemos otro."
        ),
        "notas_diagnostico": "",
        "notas_practico": "",
        "log": ["aplicar: plan rechazado"],
    }


def _router_supervisor(
    estado: EstadoGrafo,
) -> Literal["diagnostico", "practico", "sintetizar", "aplicar"]:
    """DADO: lee estado['siguiente'] y elige el edge.

    FINISH → nodo aplicar (donde está el interrupt HITL).
    Si aún no hay plan/respuesta → sintetizar antes de pausar.
    """
    sig = (estado.get("siguiente") or "").strip()
    if sig == "FINISH":
        if not (estado.get("plan") or estado.get("respuesta")):
            return "sintetizar"
        return "aplicar"
    if sig in ("diagnostico", "practico", "sintetizar"):
        return sig  # type: ignore[return-value]
    return "sintetizar"


def build_graph():
    """TODO [edges]: monta el StateGraph.

    1. add_node: supervisor, diagnostico, practico, sintetizar, aplicar
    2. START → supervisor
    3. add_conditional_edges(supervisor, _router_supervisor, mapa a los 4 destinos)
    4. diagnostico/practico/sintetizar → supervisor
    5. aplicar → END
    6. compile(checkpointer=MemorySaver(), interrupt_before=[\"aplicar\"])
    """
    raise NotImplementedError("Implementa build_graph en src/graph.py")


# Grafo ya compilado (se crea la 1ª vez que alguien llama get_app).
# Así MemorySaver es el mismo entre turnos del mismo proceso.
_grafo_compilado = None


def get_app():
    """Devuelve el grafo compilado (lo crea una sola vez)."""
    global _grafo_compilado
    if _grafo_compilado is None:
        _grafo_compilado = build_graph()
    return _grafo_compilado


def config_hilo(thread_id: str) -> dict:
    """Diccionario que LangGraph necesita para saber qué conversación es."""
    return {"configurable": {"thread_id": thread_id}}


def estado_inicial_grafo(pedido: str, max_steps: int | None = None) -> EstadoGrafo:
    """Estado limpio al empezar un mensaje nuevo en el grafo."""
    return {
        "pedido": pedido,
        "notas_diagnostico": "",
        "notas_practico": "",
        "respuesta": "",
        "plan": [],
        "siguiente": "",
        "iteracion": 0,
        "log": [],
        "traza_tools": [],
        "max_steps": max_steps if max_steps is not None else config.MAX_STEPS_DEFAULT,
        "plan_aprobado": False,
        "done": False,
        "pendiente_hitl": False,
    }


def hay_interrupt(thread_id: str) -> bool:
    """True si este hilo está pausado esperando aprobar/rechazar (antes de aplicar).

    get_state(...).next incluye el nombre del siguiente nodo bloqueado.
    """
    snap = get_app().get_state(config_hilo(thread_id))
    siguientes = snap.next or ()
    return "aplicar" in siguientes
