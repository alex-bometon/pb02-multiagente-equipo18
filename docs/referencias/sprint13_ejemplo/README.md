![Cabecera](../../../Sprint_11/assets/cabecera_thebridge.png)

# Práctica Sprint 13 — Multiagente LangGraph · calidad del aire

**Live Review** del Sprint 13: un agente **multiagente** con LangGraph que orienta consultas de **calidad del aire** en Madrid.

Un **supervisor** reparte el trabajo entre especialistas (`diagnostico`, `practico`, `sintetizar`). Los especialistas pueden usar **tools** (incluida una multimodal `describir_imagen`). Un humano **aprueba o rechaza** el plan antes de cerrar (`interrupt_before` + checkpointer).

```text
mensaje → procesar_turno(estado, mensaje, thread_id)
       → grafo multiagente (MemorySaver)
       → pausa antes de aplicar → humano aprueba / rechaza
CLI / Streamlit = clientes (mismo thread_id)
```

> El esqueleto (tools, prompts de nodos, clientes) ya está montado. **Tu trabajo** es cablear lo que hace funcionar el multiagente: reglas del supervisor, edges del grafo con HITL, y el turno que invoca el grafo.

### Flujo del grafo

```text
START → supervisor ──┬── diagnostico ────┐
                     ├── practico ───────┼──► supervisor → …
                     └── sintetizar ─────┘
                                │
                   FINISH → aplicar (interrupt_before) → END
```

### Flujo de un pedido (`procesar_turno`)

```text
Usuario escribe un mensaje (+ thread_id)
        │
        ▼
procesar_turno → app.invoke(...)     ← src/agent.py / graph.py
        │
        ├─ supervisor  → elige diagnostico | practico | sintetizar | FINISH
        ├─ diagnostico / practico  → mini-bucle con @tool (max_steps)
        ├─ sintetizar  → plan en texto + lista `plan`
        └─ (antes de aplicar) interrupt  → pendiente HITL
                │
                ▼
        humano: aprobar / rechazar
                │
                ▼
        update_state + invoke(None) → nodo aplicar → done / reset
```

---

## Empieza aquí

Sigue este orden **de arriba a abajo**:

### Fase 0 — Entorno

- [ ] **1.** Crea el venv, instala dependencias y copia `.env.example` → `.env` con tu `GEMINI_API_KEY`
- [ ] **2.** `python main.py --check` → debe marcar varios `[FAIL]` (aún hay TODOs)

### Fase 1 — Grafo (`src/graph.py`)

- [ ] **3.** Completa `_fallback_siguiente` (reglas: diagnostico → practico → sintetizar → FINISH)
- [ ] **4.** Completa `build_graph` (nodos, conditional edges, `MemorySaver`, `interrupt_before=["aplicar"]`)

### Fase 2 — Orquestación (`src/agent.py`)

- [ ] **5.** Completa `procesar_turno` (`invoke` + `_marcar_hitl` + traza)
- [ ] **6.** `python main.py --check` → todo `[OK]`

### Fase 3 — Demostrar (CLI)

- [ ] **7.** `python main.py` → log supervisor / tools; plan; HITL auto-aprobado en demo
- [ ] **8.** `python main.py --sin-auto-aprobar` → queda en pausa HITL
- [ ] **9.** `python main.py --interactivo` → `aprobar` / `rechazar` a mano

### Fase 4 — Streamlit (app dada)

- [ ] **10.** `streamlit run app.py`
- [ ] **11.** Comprueba: chat, sidebar **Aprobar** / **Rechazar**, expander debug, «Nueva conversación»

### Archivos que **no debes modificar**

En `src/`: `state.py`, `llm.py`, `tools/*`, nodos del grafo (salvo los TODOs), `aprobar_plan`, `run_demo`, `__init__.py`.

En la raíz: `main.py`, `app.py`, `verificar.py`, `config.py` (puedes cambiar defaults como `MAX_TURNS_DEFAULT` / `MAX_STEPS_DEFAULT` / `GEMINI_MODEL`), `gemini_auth.py`, `data/*`.

`app.py` viene **dado**. No metas ahí la lógica del agente.

**Sí implementas:** `_fallback_siguiente`, `build_graph`, `procesar_turno`.

---

## Requisitos

- Python 3.10+
- `GEMINI_API_KEY` en [Google AI Studio](https://aistudio.google.com/apikey)

```bash
python -m venv .venv
source .venv/bin/activate          # Git Bash / macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env               # edita tu clave
python main.py --check
```

Ejecuta siempre **desde la raíz del proyecto**.

---

## Estructura

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── config.py              ← dado (modelo, límites, DEMO_MENSAJES, URL API)
├── gemini_auth.py         ← dado
├── main.py                ← dado (CLI + --check)
├── app.py                 ← dado (Streamlit; cliente de procesar_turno)
├── verificar.py           ← dado
├── .streamlit/config.toml
├── data/
│   ├── guia_calidad_aire.txt   ← corpus del mini-RAG
│   └── aire_fallback.json      ← fallback de la API
├── assets/
│   └── imagen_demo.jpg    ← tú la añades (multimodal)
└── src/
    ├── __init__.py        ← dado
    ├── state.py           ← dado (crear_estado / estado_desde_grafo)
    ├── llm.py             ← dado (ChatGoogleGenerativeAI)
    ├── graph.py           ← nodos DADOS; TODOs: _fallback_siguiente, build_graph
    ├── agent.py           ← aprobar_plan / run_demo DADOS; TODO: procesar_turno
    └── tools/
        ├── __init__.py    ← dado (allowlist @tool)
        ├── hora_actual.py ← dado
        ├── rag_guia.py    ← dado (mini-RAG léxico)
        ├── api_calidad_aire.py  ← dado (API Madrid + fallback)
        └── describir_imagen.py  ← dado (multimodal; lee assets/imagen_demo.jpg)
```

---

## Consultas del agente (y tools)

El agente **orienta** una consulta sobre calidad del aire y cierra con un **plan** (≥2 viñetas) sujeto a HITL.

| Tool | Rol |
|------|-----|
| `hora_actual` | Fecha/hora local |
| `RAG_buscar_en_guia` | Mini-RAG léxico sobre `data/guia_calidad_aire.txt` |
| `API_consultar_calidad_aire` | Open data Madrid (tiempo real) + fallback JSON |
| `describir_imagen` | Multimodal: imagen → texto (`assets/imagen_demo.jpg`) |

| Especialista | Rol |
|--------------|-----|
| `diagnostico` | Datos / guía / imagen / interpretar (RAG, API, `describir_imagen`) |
| `practico` | Recomendaciones (hora, exposición, tips) |
| `sintetizar` | Une notas → `respuesta` + `plan` |
| `aplicar` | Cierra tras aprobar / limpia tras rechazar |

**No** inventes mediciones ni umbrales oficiales si la tool no los aportó.

La imagen demo vive en `assets/imagen_demo.jpg` (`config.IMAGEN_DEFAULT_PATH`). Añade ahí un JPG/PNG para probar multimodal.

---

## Mensajes de prueba (copiar / pegar)

Sirven igual en **`python main.py --interactivo`** (tras `Tú:`) y en el **chat de Streamlit**.

### Pedido completo (guía + API)

```text
Zona centro, contaminante NO2. Consulta la guía y datos de la API de Madrid. Propón un plan breve de qué hacer / cómo interpretar.
```

Qué mirar: log con `supervisor` → `diagnostico` / `practico` → `sintetizar`; tools `RAG_buscar_en_guia` y/o `API_consultar_calidad_aire`; plan pendiente HITL.

### Multimodal (imagen demo)

```text
Mira la imagen de demo y resume qué ves. Luego propón un plan breve de calidad del aire en Madrid usando también la guía.
```

Qué mirar: en la traza debe salir `describir_imagen` (lee `assets/imagen_demo.jpg`).

### Solo guía

```text
¿Qué significa un valor alto de NO2 según la guía?
```

### Solo API

```text
Trae un resumen de mediciones de calidad del aire en Madrid.
```

### Tras el plan (HITL)

| Dónde | Qué escribir / pulsar |
|-------|------------------------|
| CLI `--interactivo` | `aprobar` o `rechazar` |
| Streamlit | Sidebar → **Aprobar** / **Rechazar** |

En CLI, `salir` (o Enter vacío) termina la sesión.

**Evitar:** monólogos enormes o pedir cosas fuera de calidad del aire / Madrid (salvo la prueba multimodal con la imagen demo).

---

## Ejecución (CLI)

```bash
# Demo (DEMO_MENSAJES; HITL auto-aprobado)
python main.py

# Sin auto-aprobar: queda en pausa HITL
python main.py --sin-auto-aprobar

# Límites
python main.py --max-turns 1 --max-steps 2

# Chat en terminal
python main.py --interactivo

# Comprobar TODOs (sin gastar cuota Gemini)
python main.py --check
```

Al terminar, la CLI imprime un **resumen** (`done`, HITL, `thread_id`, log) y el JSON del estado.

Si la CLI falla (API key, red, deps), la UI también fallará: arregla el backend primero.

---

## Secuencia recomendada

```text
python main.py --check
# Completa _fallback_siguiente + build_graph + procesar_turno
python main.py --check
python main.py
python main.py --sin-auto-aprobar
python main.py --interactivo
streamlit run app.py
```

---

## FASE 1 — `_fallback_siguiente` (`src/graph.py`)

### Objetivo

Si el LLM del supervisor falla o responde mal, Python elige el siguiente nodo con reglas claras.

### Pistas

- Sin `notas_diagnostico` → `"diagnostico"`
- Sin `notas_practico` → `"practico"`
- Sin `respuesta` → `"sintetizar"`
- Si no → `"FINISH"`

### Criterios de aceptación

- [ ] `python main.py --check` marca `[OK]` en `_fallback_siguiente`
- [ ] Los 4 casos del verificador pasan

---

## FASE 2 — `build_graph` (`src/graph.py`)

### Objetivo

Montar el `StateGraph`: nodos, edges condicionales, checkpointer e interrupt HITL.

### Pistas

- `add_node` para `supervisor`, `diagnostico`, `practico`, `sintetizar`, `aplicar`
- `START` → `supervisor`
- `add_conditional_edges("supervisor", _router_supervisor, {…})` (el router ya está dado)
- Tras cada especialista: vuelta a `supervisor`
- `aplicar` → `END`
- `compile(checkpointer=MemorySaver(), interrupt_before=["aplicar"])`

### Criterios de aceptación

- [ ] `--check` → `[OK]` en `build_graph`
- [ ] El código usa `MemorySaver`, `interrupt_before` y `add_conditional_edges`

---

## FASE 3 — `procesar_turno` (`src/agent.py`)

### Objetivo

Un mensaje → `app.invoke` → estado de producto + flag HITL + traza.

### Pistas

- Early-return de mensaje vacío ya está (no llames al grafo).
- `inicial = estado_inicial_grafo(mensaje, max_steps=…)`
- `resultado = get_app().invoke(inicial, config_hilo(thread_id))`
- `nuevo = estado_desde_grafo(resultado, estado)` → `_marcar_hitl(nuevo, thread_id)`
- Append a `traza` con `log` y `tools` del resultado
- En `except`: `error` + respuesta amigable (sin tumbar)

`aprobar_plan` y `run_demo` vienen **dados**.

### Criterios de aceptación

- [ ] `--check` → todo `[OK]`
- [ ] `python main.py` produce plan / log / tools y auto-aprueba en demo
- [ ] Mensaje vacío → respuesta amigable **sin** traza nueva

---

## FASE 4 — Streamlit (`app.py` dado)

No reimplementes la UI: solo arráncala cuando `procesar_turno` y el grafo funcionen.

```bash
streamlit run app.py
```

Se abre el navegador en `http://localhost:8501` (o la URL que imprima la terminal).

### Qué verás

1. **Chat** — pega un mensaje de prueba; suele lanzar el grafo hasta el plan.
2. **HITL** — si hay plan pendiente, el input se bloquea y en la **sidebar** aparecen **Aprobar** / **Rechazar**.
3. **Sidebar** — `max_turns`, `max_steps`, `thread_id`, «Nueva conversación».
4. **Expander debug** — plan, log, traza, `done`, errores.

### Tres memorias (+ hilo del grafo)

| Clave | Qué guarda |
|-------|------------|
| `st.session_state.messages` | Burbujas del chat (UI) |
| `st.session_state.agent_state` | Plan, log, traza, flags HITL/`done` |
| `st.session_state.thread_id` | Hilo del checkpointer (`MemorySaver`) |

«Nueva conversación» reinicia chat, estado y **nuevo** `thread_id`.

### Guión rápido en Streamlit

| # | Qué hacer | Qué observar |
|---|-----------|--------------|
| 1 | Pegar el **pedido completo** | Log multiagente; plan pendiente; botones laterales |
| 2 | (Opcional) mensaje **multimodal** | Traza con `describir_imagen` |
| 3 | Sidebar → **Aprobar** | `done=true`; mensaje de cierre |
| 4 | «Nueva conversación» → otro pedido → **Rechazar** | Plan limpio; puedes mandar otro mensaje |

### Criterios de aceptación

- [ ] Tras el plan aparecen **Aprobar** / **Rechazar** sin escribir en el chat
- [ ] Aprobar cierra con `done`; Rechazar permite otro mensaje
- [ ] «Nueva conversación» cambia `thread_id`

Idea clave: **`app.py` es un cliente**. La lógica está en el grafo y en `procesar_turno`.

---

## HITL (human-in-the-loop)

1. Tras `sintetizar`, el grafo se **pausa** antes de `aplicar` (`interrupt_before`).
2. La UI / CLI marcan `pendiente_hitl`.
3. Al aprobar o rechazar: `update_state(plan_aprobado=…)` + `invoke(None)` → entra en `aplicar`.

---

## Qué probar en el proyecto

| Caso | Comando / acción | Éxito cuando… |
|------|------------------|---------------|
| Check | `python main.py --check` | Todo `[OK]` |
| Demo | `python main.py` | Log supervisor; tools; plan; auto-HITL |
| Pausa HITL | `python main.py --sin-auto-aprobar` | `pendiente_hitl` sin `done` |
| Interactivo | `python main.py --interactivo` + `aprobar` | Cierre tras aprobar |
| Corte steps | `python main.py --max-steps 1` | Tools limitadas por especialista |
| Fallback | Sin red | `Fuente: fallback` en preview de tools |
| Multimodal | Mensaje de imagen demo | Traza con `describir_imagen` |
| Streamlit | Pedido + sidebar | Botones HITL; expander con log/traza |

---

## Pistas (no leer antes de intentar)

<details>
<summary>Spoiler — _fallback_siguiente</summary>

```python
if not (estado.get("notas_diagnostico") or "").strip():
    return "diagnostico"
if not (estado.get("notas_practico") or "").strip():
    return "practico"
if not (estado.get("respuesta") or "").strip():
    return "sintetizar"
return "FINISH"
```

</details>

<details>
<summary>Spoiler — build_graph (esqueleto)</summary>

```python
grafo = StateGraph(EstadoGrafo)
# add_node × 5
grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges("supervisor", _router_supervisor, {…})
# diagnostico/practico/sintetizar → supervisor
grafo.add_edge("aplicar", END)
return grafo.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["aplicar"],
)
```

</details>

<details>
<summary>Spoiler — procesar_turno (núcleo)</summary>

```python
app = get_app()
cfg = config_hilo(thread_id)
inicial = estado_inicial_grafo(mensaje, max_steps=max_steps)
resultado = app.invoke(inicial, cfg)
nuevo = estado_desde_grafo(resultado, estado)
nuevo = _marcar_hitl(nuevo, thread_id)
# append traza con log / tools
return nuevo
```

</details>

---

## Criterio de cierre

- [ ] `--check` todo `[OK]`
- [ ] Demo CLI con plan y tools visibles
- [ ] HITL funciona (auto en demo; botones en Streamlit o `aprobar` en interactivo)
- [ ] No inventáis mediciones / umbrales sin tool

---

## Experimentos opcionales (si sobra tiempo)

1. `--max-steps 1` y un pedido que fuerce guía + API.
2. Sin red: comprobar `Fuente: fallback`.
3. Rechazar el plan y pedir otro ajuste (zona / contaminante).
4. Inspeccionar `log` del supervisor en el expander / JSON.

---

## Mini glosario

| Término | Qué significa aquí |
|---------|-------------------|
| **Supervisor** | Nodo que elige el siguiente rol; no escribe el plan al usuario. |
| **Especialista** | Nodo con prompt propio; puede llamar tools. |
| **MemorySaver** | Checkpoints en RAM por `thread_id`. |
| **interrupt_before** | Pausa el grafo antes de un nodo sensible (`aplicar`). |
| **HITL** | Una persona aprueba o rechaza antes de cerrar. |
