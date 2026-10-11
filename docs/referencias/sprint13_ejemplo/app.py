"""App Streamlit — cliente multiagente · calidad del aire (alumno).

  streamlit run app.py
"""

import streamlit as st

import config
from src.agent import aprobar_plan, nuevo_thread_id, procesar_turno
from src.state import crear_estado

st.set_page_config(
    page_title="Multiagente · calidad del aire",
    page_icon="🌬️",
    layout="centered",
)

SALUDO = (
    "Hola — oriento consultas de **calidad del aire** con un grafo multiagente "
    "(supervisor + diagnóstico/práctico + tools RAG/API). "
    "Cuéntame zona, contaminante o tipo de consulta. "
    "Cuando el plan esté listo, lo apruebas en la barra lateral."
)


def _reset() -> None:
    st.session_state.messages = [{"role": "assistant", "content": SALUDO}]
    st.session_state.agent_state = crear_estado("")
    st.session_state.thread_id = nuevo_thread_id()


with st.sidebar:
    st.header("Configuración")
    max_turns = st.slider(
        "max_turns",
        min_value=1,
        max_value=12,
        value=config.MAX_TURNS_DEFAULT,
    )
    max_steps = st.slider(
        "max_steps (tools por especialista)",
        min_value=1,
        max_value=8,
        value=config.MAX_STEPS_DEFAULT,
    )
    st.caption(f"thread_id: `{st.session_state.get('thread_id', '—')}`")

    estado_side = st.session_state.get("agent_state") or {}
    if estado_side.get("pendiente_hitl"):
        st.divider()
        st.subheader("HITL — aprobar plan")
        for item in estado_side.get("plan") or []:
            st.markdown(f"- {item}")
        c1, c2 = st.columns(2)
        if c1.button("Aprobar", use_container_width=True, type="primary"):
            st.session_state.agent_state = aprobar_plan(
                st.session_state.thread_id, True, st.session_state.agent_state
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": st.session_state.agent_state.get("respuesta")
                    or "Plan aprobado.",
                }
            )
            st.rerun()
        if c2.button("Rechazar", use_container_width=True):
            st.session_state.agent_state = aprobar_plan(
                st.session_state.thread_id, False, st.session_state.agent_state
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": st.session_state.agent_state.get("respuesta")
                    or "Plan rechazado.",
                }
            )
            st.rerun()

    if st.button("Nueva conversación", use_container_width=True):
        _reset()
        st.rerun()

st.title("Multiagente · calidad del aire")
st.caption("Sprint 13 LR — supervisor + tools + MemorySaver + interrupt_before")

if "messages" not in st.session_state:
    _reset()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

estado = st.session_state.agent_state
chat_bloqueado = bool(estado.get("done") or estado.get("error"))
turnos_hechos = len(estado.get("traza") or [])
hitl_bloquea = bool(estado.get("pendiente_hitl"))

if chat_bloqueado:
    st.info("Conversación cerrada (`done` o `error`). Pulsa «Nueva conversación».")
elif hitl_bloquea:
    st.warning("Plan pendiente de aprobación — usa los botones de la barra lateral.")
elif turnos_hechos >= max_turns:
    st.warning("Has alcanzado `max_turns`. Pulsa «Nueva conversación» o sube el límite.")

if prompt := st.chat_input(
    "Escribe tu mensaje…",
    disabled=chat_bloqueado or hitl_bloquea or turnos_hechos >= max_turns,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Grafo multiagente (especialistas + tools)…"):
            estado = procesar_turno(
                estado,
                prompt,
                st.session_state.thread_id,
                max_steps=max_steps,
            )
            st.session_state.agent_state = estado

        if estado.get("error"):
            st.error(estado["error"])

        texto = estado.get("respuesta") or "(Sin respuesta)"
        plan = estado.get("plan") or []
        if plan:
            if estado.get("pendiente_hitl"):
                etiqueta = "Plan (pendiente de aprobación)"
            elif estado.get("done"):
                etiqueta = "Plan"
            else:
                etiqueta = "Plan (borrador)"
            bullets = "\n".join(f"- {item}" for item in plan)
            texto = f"{texto}\n\n**{etiqueta}:**\n{bullets}"
        st.markdown(texto)
        st.session_state.messages.append({"role": "assistant", "content": texto})

    if estado.get("pendiente_hitl") or estado.get("done") or estado.get("error"):
        st.rerun()

with st.expander("Estado / log / tools (debug)"):
    st.json(
        {
            "thread_id": st.session_state.get("thread_id"),
            "plan": st.session_state.agent_state.get("plan"),
            "pendiente_hitl": st.session_state.agent_state.get("pendiente_hitl"),
            "plan_aprobado": st.session_state.agent_state.get("plan_aprobado"),
            "done": st.session_state.agent_state.get("done"),
            "log": st.session_state.agent_state.get("log"),
            "error": st.session_state.agent_state.get("error"),
            "traza": st.session_state.agent_state.get("traza"),
        }
    )
