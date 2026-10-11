![Cabecera](../assets/cabecera_agentes.png)

# Project Break 2 — Agentes con LangGraph

**Duración:** 2 semanas  

Construir en equipo un **agente multiagente end-to-end** con LangGraph: estado, tools controladas (incluido un **RAG completo**), orquestación con supervisor, **HITL**, persistencia con checkpointer y una tool **multimodal**, con CLI y Streamlit.

Esta práctica integra conceptos vistos de AI Agents Engineering:

- **Estado, `done` y límites de turno** (Sprint 11)
- **Tool use, allowlist, `max_steps`, RAG como tool** (Sprint 12)
- **LangGraph, MemorySaver, HITL, multiagente, multimodal** (Sprint 13)

---

## Contexto: pieza de un puzzle

Este Project Break es la **segunda pieza** de un proyecto incremental del bootcamp:

| Pieza | Proyecto | Rol |
|-------|----------|-----|
| **1** | [Project Break RAG](../PB_01_RAG/) | Sistema RAG modular y usable |
| **2** | **Project Break Agentes · LangGraph** | Ese RAG (u otro de la misma magnitud) se usa como **tool** del agente |
| **3** | Proyecto final MLOps | Empaquetar, desplegar y monitorizar el sistema |

El agente **no sustituye** al RAG: lo **invoca** como herramienta. Diseñad el RAG con una API clara (p. ej. `rag_ask(consulta) -> str` o `responder(pregunta) -> dict`) usable **sin Streamlit**, para poder conectarlo al grafo como tool de LangGraph.

---

# Objetivos generales

Desarrollar un agente multiagente con LangGraph que:

1. Mantenga **estado** de producto y de grafo, con límites (`max_turns` / `MAX_ITER` / `max_steps`) decididos en **Python**.
2. Use un **RAG completo** (magnitud Project Break 1) como tool, más API externa, una herramienta auxiliar (p. ej. hora) y multimodal.
3. Orqueste con **supervisor + ≥2 especialistas + sintetizar + aplicar**, con **HITL** antes de cerrar.
4. Persista turnos con **checkpointer** (`MemorySaver` + `thread_id`).
5. Exponga el mismo contrato en **CLI** y **Streamlit** (clientes del grafo).

Misma arquitectura que los **proyectos y live reviews de los Sprints 11–13**. Hay que elegir una temática propia.

Se recomienda reutilizar la **estructura y buenas prácticas** vistas en clase (proyecto cultural, live reviews). No hace falta inventar un producto distinto: hay que **cerrarlo como entregable de equipo**.

```text
mensaje → procesar_turno(estado, mensaje, thread_id)
       → grafo multiagente (MemorySaver)
       → tools (RAG completo + API + auxiliar + multimodal)
       → interrupt_before[aplicar] → humano aprueba / rechaza
CLI / Streamlit = clientes (mismo thread_id)
```

---

## Entregables finales

1. **Repositorio GitHub** desde el primer día, con código reproducible, `.env.example`, `.gitignore` y README del equipo bien documentado.
2. **Agente funcional (CLI):** demo + modo interactivo / HITL (`aprobar` / `rechazar` o equivalente).
3. **Web App Streamlit:** chat + sidebar **Aprobar** / **Rechazar** + expander debug (log, traza de tools, `thread_id`, flags HITL/`done`).
4. **RAG completo** integrado como tool (reutilizado del PB1 RAG u otro de **la misma magnitud**; ver sección RAG).
5. **Tools documentadas** (allowlist): RAG + ≥1 API/externa + herramienta auxiliar + multimodal, con traza visible.
6. **Informe breve** (`entregables/informe_decisiones.md`): diseño del grafo, HITL, tools, 1 acierto + 1 rechazo HITL, 3 fallos y siguientes pasos (MLOps).
7. **Presentación final** (~10 min) en vivo o vídeo; **sólo se permite un vídeo por grupo**.

---

## Trabajo en equipo y Git

Gestión del proyecto con **GitHub desde el primer día**. La organización interna (quién hace qué) la decidís vosotros.

### Reglas mínimas

- Crear el **repositorio en GitHub** al inicio del proyecto.
- **Mínimo una PR revisada y mergeada por miembro** del equipo.
- Integrar con **pull requests** hacia `develop`; resolver conflictos en la rama antes del merge.
- **No subir claves** — usar `.env` y `.gitignore`.
- README del repo: clonar, instalar, configurar API keys, indexar RAG (si aplica), ejecutar CLI + Streamlit, cómo probar HITL y multimodal.

### Buenas prácticas con Git

| Rama | Uso |
|------|-----|
| `main` | Código estable y entregable. **No trabajéis directamente aquí.** |
| `develop` | Integración del equipo. |
| Ramas secundarias | Una por bloque de trabajo, desde `develop`. |

**Convención de ramas** (ejemplos):

- `feature/grafo-hitl`
- `feature/tools-rag`
- `feature/api-externa`
- `feature/multimodal`
- `feature/streamlit-ui`
- `fix/thread-id`
- `update/requirements`

```text
develop ──► feature/tu-tarea ──► commits ──► PR ──► merge a develop
       └──► cuando todo esté listo ──► merge a main
```

- Commits **pequeños y con mensaje claro**.
- **Nunca** subáis `.env`, `.venv/`, índices vectoriales pesados ni secretos.
- Antes de mergear a `main`, `develop` debe poder ejecutar demo CLI + Streamlit sin errores pendientes.

---

## Forma de trabajo · Spec-Driven Development (SDD)

**SDD forma parte de cómo afrontar este Project Break.** Aprenderlo importa: es una práctica central de **desarrollo con IA**. En lugar de *vibe coding* (prompt → código → “parece que va”), acordáis por escrito **qué** debe hacer cada feature y usáis esa especificación para dirigir al agente de código.

Usadlo **en todo lo que podáis** del proyecto (grafo, HITL, tools, RAG, UI…): cuanto más grande es el alcance, más útil es no improvisar con el asistente. Así evitáis que invente arquitectura o rompa el diseño (multiagente, allowlist, HITL).

No es un ítem del checklist de entrega ni un porcentaje de la rúbrica base (el producto tiene que funcionar igual). Si dejáis evidencia clara en el repo o en el informe, puede sumar en **Extras**.

---

## Estructura de proyecto

Ejemplo **orientativo** para la estructura del proyecto:

```text
project_break_agentes_langgraph/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
├── main.py                 # CLI: demo / interactivo / HITL
├── app.py                  # Streamlit (cliente)
├── gemini_auth.py          # o equivalente
├── src/
│   ├── state.py            # estado de producto (UI / CLI)
│   ├── llm.py
│   ├── graph.py            # StateGraph multiagente + HITL
│   ├── agent.py            # procesar_turno / aprobar_plan / run_demo
│   └── tools/
│       ├── __init__.py     # allowlist + ejecutar_tool
│       ├── rag_ask.py      # tool → RAG completo
│       ├── api_dominio.py  # ≥1 API / fuente externa
│       ├── hora_actual.py  # o herramienta auxiliar equivalente
│       └── describir_imagen.py
├── rag/                    # o paquete/submódulo del RAG completo
│   ├── …                   # load / chunk / embed / index / retrieve / generate
│   └── …
├── data/                   # corpus RAG + fallbacks API si aplica
├── assets/
│   └── imagen_demo.jpg     # demo multimodal
├── entregables/
│   └── informe_decisiones.md
└── .streamlit/config.toml  # opcional
```

Evitad un único notebook o script monolítico como entregable. Mantened **separación**: tools · grafo · orquestación · UI.

Podéis **basaros** en la estructura de los proyectos de clase (tarde cultural / live review LangGraph), adaptando dominio y tools.

---

## Capacidades del sistema

### 1. Grafo multiagente (LangGraph)

```text
START → supervisor ──┬── especialista_A ────┐
                     ├── especialista_B ────┼──► supervisor → …
                     └── sintetizar ────────┘
                                │
                   FINISH → aplicar (interrupt_before) → END
```

- **Supervisor:** elige el siguiente rol (o `FINISH`); con fallback determinista en Python si el LLM falla.
- **≥2 especialistas:** cada uno con prompt propio; pueden usar tools (mini-bucle o `bind_tools` según diseño).
- **Sintetizar:** une notas en `respuesta` + `plan` (viñetas) listo para HITL.
- **Aplicar:** solo tras decisión humana; aprueba (`done=True`) o rechaza (reset parcial / nuevo pedido).

### 2. Tools controladas

Allowlist en Python. El LLM **propone**; el código **ejecuta** solo lo permitido. Tope `max_steps` por especialista (o equivalente).

Mínimo obligatorio:

| Tipo | Requisito |
|------|-----------|
| **RAG completo** | Magnitud PB1 (ver abajo) expuesto como tool |
| **API / externa** | Al menos una (open data, HTTP, etc.) + fallback si falla la red |
| **Herramienta auxiliar** | Determinista y simple: p. ej. `hora_actual` u otra útil al dominio (no RAG ni API) |
| **Multimodal** | Tool imagen → texto (p. ej. `describir_imagen` + `assets/imagen_demo.jpg`) |

### 3. Persistencia y HITL

- `compile(checkpointer=MemorySaver(), interrupt_before=["aplicar"])` (o nodo equivalente).
- Mismo `thread_id` entre turnos del mismo proceso.
- Tras el plan: pausa → humano aprueba/rechaza → `update_state` + `invoke(None)`.

**Obligatorio en CLI y en Streamlit.**

### 4. Clientes

- **CLI:** demo (auto-aprobar opcional) + interactivo / sin auto-aprobar para ejercer HITL a mano.
- **Streamlit:** chat + botones HITL + debug; «Nueva conversación» = nuevo `thread_id`.

```text
Usuario
   │
   ▼
procesar_turno / aprobar_plan   ← src/agent.py
   │
   ▼
grafo LangGraph                 ← src/graph.py
   │
   ├── tools (RAG, API, auxiliar, visión)
   └── MemorySaver (thread_id)
   │
   ▼
CLI  ·  Streamlit
```

**El sistema debe:** usar tools reales (no inventar datos que solo daría una tool), respetar límites en Python, pausar en HITL y cerrar solo tras aprobación.

**No se permite:** monolito sin módulos; Streamlit que contenga la lógica del grafo; tools sin allowlist; cerrar el plan sin HITL; mini-RAG de juguete en lugar de un RAG de magnitud PB1.

---

## RAG completo (magnitud Project Break 1)

El agente debe consumir un **RAG de la misma magnitud que el PB1**, no un “mini-RAG” de un solo TXT:

- Corpus curado (≥2 formatos), documentado.
- Pipeline offline: load → chunk → embed → **índice vectorial persistente** (p. ej. ChromaDB) con metadatos mínimos (`source`).
- Retrieval top-k + generación **anclada al contexto** + abstención si no hay evidencia.
- API interna callable sin UI (`rag_ask` / `responder` o equivalente) que la tool del agente invoque.

**Opciones:**

1. **Reutilizar el repo / módulo del PB1** del equipo (recomendado: proyecto incremental).
2. **Construir otro RAG completo** de dominio distinto, con el mismo nivel de exigencia que el PB1 (no basta un `guia.txt` léxico como única recuperación).

En el README del PB2 documentad: cómo indexar, cómo se llama desde la tool, y si viene del PB1 o es nuevo.

La evaluación formal de 8–15 preguntas del PB1 **no hay que repetirla entera** aquí; sí debe poderse demostrar indexado + una pregunta in-corpus + una abstención / fuera de corpus vía la tool del agente (o CLI del módulo RAG).

---

## Requisitos obligatorios

1. Grafo **multiagente**: supervisor + **≥2** especialistas + sintetizar + aplicar (HITL).
2. Checkpointer **`MemorySaver`** + `thread_id`; `interrupt_before` (o equivalente) antes de aplicar.
3. **HITL** usable en **CLI** y en **Streamlit**.
4. Tools en allowlist: **RAG completo** + **≥1 API** + **herramienta auxiliar** + **multimodal**.
5. Límites en Python: `MAX_ITER` / `max_steps` / `max_turns` (nombres orientativos); `done` no lo decide solo el LLM.
6. Traza / log visible (supervisor, tools, HITL).
7. Código **modular** (grafo ≠ tools ≠ UI).
8. README + `.env.example` + `requirements.txt`.
9. Informe en `entregables/`.
10. Presentación ~10 min (**sólo se permite un vídeo por grupo**).

---

## Temáticas

Aquí damos algunas temáticas de ejemplo. Elegid **una** de estas cuatro (idealmente la del PB1, para reutilizar el RAG). Vuestra tarea incluye encontrar fuentes de datos para el corpus del RAG, descargar, filtrar y documentarlas; y además una **API / fuente externa** y una **imagen demo** multimodal acordes al dominio.

Si os surge alguna temática que no está en la lista, podéis proponerla a los profesores para que analicen su viabilidad.

Se recomienda fijar cuanto antes la temática y tener los datos listos para no consumir demasiado tiempo en la investigación.

**Temáticas de clase (calidad del aire, agenda cultural / tarde cultural) no entran** como dominio del Project Break.

### 1. Transporte / abonos

Asistente sobre tarifas, zonas, abonos y condiciones de viaje (p. ej. CRTM / Metro / Cercanías).

| Fuente | Qué sacar |
|--------|-----------|
| CRTM / Metro / Renfe Cercanías | PDFs de tarifas, zonificación, condiciones del abono |
| Open Data Madrid / EMT | CSV de líneas, paradas; a veces incidencias |
| FAQ oficiales “títulos y tarifas” | Texto ideal para RAG |

**Recursos orientativos:**

| Recurso | URL |
|---------|-----|
| CRTM — Portal datos abiertos (GTFS/CSV) | https://datos.crtm.es/ |
| CRTM — Billetes y tarifas | https://www.crtm.es/billetes-y-tarifas |
| CRTM — PDF tarifas 2026 (BOCM) | https://www.crtm.es/media/sjqj4ggj/bocm-20251231-tarifas_transporte.pdf |
| CRTM — Bonificaciones / precios al usuario PDF | https://www.crtm.es/media/s1qi0nmo/bocm-20251231-precios_transporte.pdf |
| Renfe — Cercanías Madrid abonos | https://www.renfe.com/es/es/cercanias/cercanias-madrid/tarifas/abonos |
| Renfe — Billetes Cercanías Madrid | https://www.renfe.com/es/es/cercanias/cercanias-madrid/tarifas/billetes |
| EMT — Paradas (CSV) | https://datos.emtmadrid.es/dataset/paradas-de-emtmadrid/resource/4f0736a9-865c-428f-8719-128c805baa2e |
| Ayto. Madrid — Líneas EMT | https://datos.madrid.es/dataset/900024-0-emt-lineas-autobus |
| Ayto. Madrid — Paradas EMT | https://datos.madrid.es/dataset/900023-0-emt-paradas-autobus |

**Ejemplos de preguntas:** ¿Qué zonas cubre el abono mensual? ¿Puedo usar el mismo título en Metro y Cercanías?

---

### 2. Becas / trámites

Asistente de **una** convocatoria o trámite concreto (requisitos, plazos, documentación).

| Fuente | Qué sacar |
|--------|-----------|
| sede.educacion.gob.es / becas MEC | Convocatorias, requisitos en PDF |
| SEPE / Seguridad Social (guías ciudadano) | Trámites, documentación |
| Sedes autonómicas | Ayudas (p. ej. alquiler joven) |
| BOE (una convocatoria concreta) | 1 PDF acotado, no todo el histórico |

**Recursos orientativos:**

| Recurso | URL |
|---------|-----|
| MEC — Sede ayuda trámites | https://sede.educacion.gob.es/informacion-ayuda/tramites |
| MEC — BOE extracto becas postobligatorios 2025-2026 (PDF) | https://www.boe.es/boe/dias/2025/03/20/pdfs/BOE-B-2025-10203.pdf |
| MEC — BOE prórroga plazo solicitud (PDF) | https://www.boe.es/boe/dias/2025/05/17/pdfs/BOE-B-2025-17903.pdf |
| MEC — Libro Becas 2025-2026 | https://www.libreria.educacion.gob.es/ebook/186220/free_download/ |
| SEPE — FAQ prestaciones desempleo | https://sepe.es/HomeSepe/preguntas-frecuentes/preguntas-frecuentes-prestaciones-desempleo.html |
| SEPE — Impresos oficiales (PDFs) | https://www.sepe.es/HomeSepe/prestaciones-desempleo/impresos.html |
| SEPE — Hoja informativa prestación contributiva (PDF) | https://www.sepe.es/dam/jcr:22e56e65-c269-4f04-96bd-3dc86daf8fc8/hoja_informativa_contributiva.pdf |
| SEPE — Guía prestación por desempleo (PDF) | https://www.sepe.es/dam/jcr:f31af863-64f7-4504-9e50-60aa0bb65b91/prestacion_por_desempleo.pdf |
| Comunidad de Madrid — Ayudas alquiler y jóvenes | http://sede.comunidad.madrid/node/292435 |
| Comunidad de Madrid — Bono Alquiler Joven | http://sede.comunidad.madrid/node/285843 |
| Comunidad de Madrid — FAQ convocatoria ayuda alquiler (PDF) | https://sede.comunidad.madrid/medias/20251017preguntasfrecuentesconv2025pjspeav22-25pdf/download |

**Ejemplos de preguntas:** ¿Qué requisitos pide esta beca? ¿Qué documentos debo entregar? ¿Hasta qué fecha puedo solicitarla?

---

### 3. Parques / turismo sostenible

Asistente del visitante (normas, rutas, restricciones, temporadas).

| Fuente | Qué sacar |
|--------|-----------|
| miteco.gob.es (parques nacionales) | Normas, PDFs del visitante |
| Webs de parques (Picos, Doñana, Guadarrama…) | Rutas, restricciones, mascotas |
| Open data turismo regional | CSV de puntos de interés |

**Recursos orientativos:**

| Recurso | URL |
|---------|-----|
| MITECO — Red de Parques Nacionales | https://www.miteco.gob.es/es/parques-nacionales-oapn/red-parques-nacionales/parques-nacionales.html |
| Guadarrama — Folletos y guías (PDF) | https://www.parquenacionalsierraguadarrama.es/normativa/descargas/category/4-folletos |
| Guadarrama — Guía de lectura fácil (PDF) | https://www.parquenacionalsierraguadarrama.es/normativa/descargas/download/4-folletos/195-guia-visita-accesible |
| Guadarrama — Folleto presentación del parque (PDF) | https://www.parquenacionalsierraguadarrama.es/normativa/descargas/download/4-folletos/2-folleto-pnsg |
| Picos de Europa — Guía del visitante | https://www.miteco.gob.es/es/parques-nacionales-oapn/red-parques-nacionales/parques-nacionales/picos-europa/guia-visitante.html |
| Picos de Europa — Normas de visita | https://www.miteco.gob.es/es/parques-nacionales-oapn/red-parques-nacionales/parques-nacionales/picos-europa/guia-visitante/normas-recomendaciones.html |
| Doñana — Guía del visitante | https://www.miteco.gob.es/es/parques-nacionales-oapn/red-parques-nacionales/parques-nacionales/donana/guia-visitante.html |
| Doñana — Normas de visita | https://www.miteco.gob.es/es/parques-nacionales-oapn/red-parques-nacionales/parques-nacionales/donana/guia-visitante/recomendaciones.html |
| Ayto. Madrid — Puntos de interés turístico | https://datos.madrid.es/dataset/300030-0-puntos-interes-turistico |
| Ayto. Madrid — Principales parques y jardines | https://datos.madrid.es/dataset/200761-0-parques-jardines |
| Ayto. Madrid — Inventario de zonas verdes (CSV) | https://datos.madrid.es/dataset/300153-0-zonas-verdes-inventario |

**Ejemplos de preguntas:** ¿Puedo llevar perro? ¿Hay restricción de acceso en verano? ¿Qué rutas son para principiantes?

---

### 4. Deporte municipal

Asistente de instalaciones municipales (normas, tarifas, reservas, horarios).

| Fuente | Qué sacar |
|--------|-----------|
| Open data del ayuntamiento | CSV de instalaciones, horarios |
| PDF normas de uso / tarifas deportivas | Reglas del dominio |
| Web municipal de “deportes” | FAQ de reservas |

**Recursos orientativos:**

| Recurso | URL |
|---------|-----|
| Ayto. Madrid — Portal Deportes Web | https://deportesweb.madrid.es |
| Ayto. Madrid — Categoría Deporte (datos abiertos) | https://datos.madrid.es/group/deporte |
| Ayto. Madrid — Polideportivos (CSV) | https://datos.madrid.es/dataset/200186-0-polideportivos |
| Ayto. Madrid — Polideportivos CSV directo | https://datos.madrid.es/egob/catalogo/200186-0-polideportivos.csv |
| Ayto. Madrid — Instalaciones deportivas básicas (CSV) | https://datos.madrid.es/dataset/200215-0-instalaciones-deportivas |
| Ayto. Madrid — Piscinas municipales (CSV) | https://datos.madrid.es/dataset/210227-0-piscinas-publicas |
| Ayto. Madrid — Áreas de actividades deportivas (CSV) | https://datos.madrid.es/dataset/300390-0-areas-deportivas |
| Ayto. Madrid — Agenda de actividades deportivas (CSV) | https://datos.madrid.es/dataset/212504-0-agenda-actividades-deportes |
| Ayto. Madrid — Tarifas de instalaciones deportivas (XLSX/CSV) | https://datos.madrid.es/dataset/300083-0-deportes-tarifas |
| Ayto. Madrid — Descuentos en instalaciones (CSV/XLSX) | https://datos.madrid.es/dataset/300097-0-deportes-descuentos |
| Ayto. Madrid — Abonados en centros deportivos (CSV/XLSX) | https://datos.madrid.es/dataset/300085-0-deportes_abonos |
| Ayto. Madrid — Competiciones municipales temporada vigente (TXT) | https://datos.madrid.es/dataset/211549-0-juegos-deportivos-actual |
| Ayto. Madrid — Precios públicos centros deportivos 2026 (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/PreciosPublicos2026.pdf |
| Ayto. Madrid — Tarifas de servicios en centros deportivos (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/Tarifas_deportivas.pdf |
| Ayto. Madrid — Reglamento instalaciones deportivas (PDF) | https://sede.madrid.es/eli/es-md-01860896/reg/2012/10/15/(1)/dof/spa/pdf |
| Ayto. Madrid — Listado piscinas de verano 2026 aire libre (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/Verano2026/ficheros/PiscinasAireLibre2026.pdf |
| Ayto. Madrid — Decreto anulación reservas con coste (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/ContenidoGenerico/ContenidoGenerico2024/Ficheros/DecretoAnulacionReservasConCoste.pdf |
| Ayto. Madrid — Decreto suspensión reservas a coste 0 (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/Colecciones2023/appMadridMovil/ficheros/Decretosuspension/20230120Decretosuspereservaanticipada.pdf |
| Ayto. Madrid — Infografía Abono Deporte Madrid (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/Faq/ficheros/infografias%20faq%202025/20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf |
| Ayto. Madrid — Normativa Juegos Deportivos Municipales 2025-26 (PDF) | https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/46%20Juegos%20Deportivos%20Municipales%2025-26/normativas/normativaGeneral46jdm.pdf |

**Ejemplos de preguntas:** ¿Cuánto cuesta el abono de piscina? ¿Puedo reservar siendo no empadronado?

---

### API externa y multimodal (además del corpus RAG)

Para la **API / fuente externa**, buscad un endpoint open data o HTTP acorde al dominio (más un JSON/CSV de fallback local si cae la red).  
Para **multimodal**, una imagen demo con texto legible (mapa, cartel tarifario, folleto, panel informativo, etc.) en `assets/imagen_demo.jpg` (o `.png`).

### Requisitos del corpus (RAG)

- Al menos **2 formatos** distintos (p. ej. PDF + MD/TXT, o PDF + CSV).
- Preferible combinar **texto** (guía/FAQ/PDF) con algo estructurado (CSV), no solo tablas.
- Volumen razonable (p. ej. 5–20 documentos, o 1 PDF grande + FAQ + CSV).
- Documentad las fuentes en el README del repo (enlaces y fecha de descarga).
- Evitad PDFs solo imagen (escaneados sin texto) y web scraping agresivo.

**No hay un zip de datos cerrado en este enunciado:** el corpus lo curáis vosotros (o reutilizáis el del PB1).

---

# Partes del proyecto

## Parte 1 — Dominio, contrato y diseño del grafo

**Objetivo:** cerrar alcance antes de conectar todo el código.

1. Elegir temática (reutilizar PB1 o una nueva).
2. Definir qué preguntas / pedidos **sí / no** debe resolver el agente.
3. Dibujar el grafo: nodos, edges, momento HITL, tools por especialista.
4. Acordar en el README (o nota de equipo):

   - Tema en 2–3 líneas
   - Origen del RAG (PB1 vs nuevo)
   - Lista de tools previstas
   - Especialistas y responsabilidad de cada uno
   - Confirmación: datos públicos / uso educativo / sin secretos

---

## Parte 2 — RAG completo + tools

**Objetivo:** allowlist real y RAG que se conecten como tool de LangGraph.

1. Integrar o construir el **RAG completo** (indexado + retrieve + generate + abstención).
2. Exponerlo como tool (`RAG_…` / `rag_ask` envuelto en `@tool` o equivalente).
3. Implementar API externa + herramienta auxiliar + multimodal.
4. Allowlist + ejecución segura; traza de llamadas (nombre, args, status, preview).
5. Probar cada tool **aislada** (CLI o script corto) antes de meterla en el grafo.

**Criterios de aceptación (Parte 2):**

- [ ] RAG indexa y responde in-corpus; se abstiene fuera de corpus
- [ ] Las 4 categorías de tools funcionan fuera del grafo
- [ ] Allowlist bloquea nombres no permitidos

---

## Parte 3 — LangGraph, HITL y CLI

**Objetivo:** orquestación multiagente con pausa humana.

1. `EstadoGrafo` + nodos (supervisor, especialistas, sintetizar, aplicar).
2. `build_graph`: edges, `MemorySaver`, `interrupt_before=["aplicar"]`.
3. `procesar_turno` + `_marcar_hitl` (o equivalente) + `aprobar_plan`.
4. Fallback del supervisor en Python si el LLM falla.
5. **CLI:** demo + interactivo / sin auto-aprobar; poder **aprobar** y **rechazar** a mano.

**Criterios de aceptación (Parte 3):**

- [ ] Un pedido produce log multiagente + plan + pausa HITL
- [ ] Aprobar → `done`; rechazar → se puede continuar con otro mensaje
- [ ] Mismo `thread_id` reutiliza el checkpointer en la sesión

---

## Parte 4 — Streamlit, informe y cierre

**Objetivo:** producto demostrable y entrega documentada.

1. **Streamlit:** chat + **Aprobar** / **Rechazar** + debug + nueva conversación (nuevo `thread_id`).
2. README completo (instalación, indexado RAG, CLI, UI, HITL, multimodal, capturas).
3. Informe en `entregables/informe_decisiones.md` con al menos:

   1. **Grafo:** por qué esos especialistas y ese momento HITL
   2. **Tools:** qué hace cada una; cómo se enchufa el RAG
   3. **HITL:** 1 aprobación + 1 rechazo (qué observasteis)
   4. **Multimodal:** qué imagen y qué devolvió la tool
   5. **3 fallos** y qué haríais en MLOps / siguiente iteración

4. Comprobar que no hay secretos en el repo y que la rama `main` de vuestro repositorio esté listo para entregar.

---

## Experimentos opcionales

No sustituyen los requisitos. Si os sobra tiempo, documentad 1–3 experimentos en el informe:

- [ ] **`SqliteSaver`** — persistir checkpoints en disco entre reinicios del proceso (no visto en clase; explorad la doc de LangGraph).
- [ ] **LangSmith (tracing)** — trazar un `invoke` del grafo (nodos + tool calls) y adjuntar captura o enlace en el informe. Requiere cuenta / API key propias; **no es obligatorio**.
- [ ] **Más especialistas** o un router más fino (p. ej. multimodal solo en un nodo).
- [ ] **Temperatura 0 vs >0** en el supervisor: ¿cambia el enrutado?
- [ ] **`max_steps=1`** en un especialista: ¿corta tools a tiempo?
- [ ] **Misma pregunta en CLI y Streamlit** — ¿mismo contrato?
- [ ] **Inyección ligera** — “ignora las tools y inventa datos”: ¿aguanta el diseño?

**Fuera de alcance del Project Break:**  multiagente en producción, fine-tuning, deploy cloud (MLOps).

---

## Proveedor de modelos de IA

Alineado con el módulo (**Gemini** + LangGraph + Streamlit), pero podéis usar **otro proveedor** si adaptáis el cliente y lo documentáis.

**Stack de referencia:** Python · LangChain / LangGraph · Gemini · Streamlit · índice vectorial (p. ej. Chroma) para el RAG · `google-genai` (o equivalente) para multimodal.

---

## Requisitos técnicos

- Python 3.10+
- API key del LLM (y embeddings si el RAG los usa) en `.env` — **nunca** en el repo
- `pip install -r requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # editar claves
```

---

## Orden sugerido y plan de 2 semanas

| Semana | Orientación |
|--------|-------------|
| **1** | Partes 1–2: dominio, RAG completo + tools, README parcial, PRs en marcha |
| **2** | Partes 3–4: grafo + HITL CLI, Streamlit, informe, presentación |

---

## Presentación final (~10 min)

**Obligatoria.** Podéis presentarla **en vivo** en la sesión dedicada o mediante **vídeo grabado**.

**Sólo se permite un vídeo por grupo.**

Si presentáis en vivo, podéis usar ese mismo vídeo como respaldo ante fallos de red o API.

### Formato y reglas

- Un **mismo soporte visual** en toda la presentación.
- **No superar el tiempo** → se penaliza.
- **Vídeo**
  - Parte presentación del producto
  - Parte técnica que explique cómo funciona vuestro proyecto
- **En directo:** priorizad la demo técnica en vivo; conviene tener el vídeo de respaldo.

### Guion de tiempo (orientativo)

| Bloque | Tiempo | Enfoque |
|--------|--------|---------|
| A · Producto | ~4 min | Problema, valor, demo de uso |
| B · Técnico | ~6 min | Arquitectura, tools, RAG, HITL, decisiones |

### A · Producto (~4 min)

- [ ] Problema del usuario final (claro, sin jerga innecesaria).
- [ ] Qué solución aporta el agente.
- [ ] Temática / dominio y por qué esos datos.
- [ ] Encaje en el puzzle RAG → Agente → MLOps.
- [ ] Demo de uso (Streamlit o CLI) hasta un plan usable.
- [ ] HITL: aprobar **o** rechazar y efecto visible.

### B · Técnico (~6 min)

Mostrad en pantalla (terminal, UI o diapositiva clara):

- [ ] Diagrama del grafo multiagente + momento HITL.
- [ ] Tools (RAG, API, auxiliar, multimodal) y rol de cada especialista.
- [ ] Repo GitHub y organización Git (ramas / PR).
- [ ] README: cómo se instala y se arranca (CLI + Streamlit).
- [ ] Indexado / prepare del RAG (comando o log de éxito).
- [ ] Pregunta in-corpus (con fuentes/chunks si aplica).
- [ ] Abstención / fuera de corpus.
- [ ] Pedido con log de ≥2 especialistas y traza de tools.
- [ ] HITL (aprobar/rechazar) hasta `done` o reset.
- [ ] Multimodal: tool de imagen + resultado.
- [ ] Límites y persistencia (`max_steps` / `MAX_ITER` / `thread_id`).
- [ ] 3 fallos conocidos y siguientes pasos.
- [ ] SDD si lo usasteis.

### Entrega asociada

- Repo (README, `.env.example`, sin secretos) + informe.
- Si optáis por vídeo: **un enlace** al vídeo del grupo + el mismo soporte visual.

---

## Checklist del proyecto

- [ ] Repo GitHub desde el día 1 · PRs · `main` estable  
- [ ] Tema del menú (o alternativa con OK) · no dominio de clase  
- [ ] RAG completo (PB1 u equivalente) como tool  
- [ ] ≥2 especialistas + supervisor + sintetizar + aplicar  
- [ ] MemorySaver + `thread_id` + `interrupt_before`  
- [ ] HITL en **CLI** y en **Streamlit**  
- [ ] Tools: RAG + API + auxiliar + multimodal (allowlist + traza)  
- [ ] Límites en Python (`MAX_ITER` / `max_steps` / …)  
- [ ] Streamlit: chat + HITL + debug + nueva conversación  
- [ ] README + `.env.example` + `requirements.txt`  
- [ ] Informe (grafo, tools, HITL, multimodal, 3 fallos)  
- [ ] (Opcional) SqliteSaver u otrzos extras documentados  
- [ ] Presentación (~12 min) en directo o vídeo (**sólo se permite un vídeo por grupo**)

---

## Criterios de evaluación

Vuestro proyecto se valora en **seis áreas**. La mayoría son de **equipo**; una es **individual** (aportación real en Git). Los porcentajes son el **peso sobre la nota**.

Las cinco áreas base suman el **100 %**: Desarrollo Técnico (35 %), Presentación (23 %), Git (17 %), Trabajo en equipo (15 %) y Memoria (10 %). Los **Extras** suman aparte, hasta un **+10 %** adicional.

### Desarrollo Técnico · equipo · 35 %

- **Estructura y modularización:** tools / grafo / agent / UI separados; `config` centralizado.
- **Funcionalidad:** multiagente + HITL + checkpointer; tools reales; RAG completo usable.
- **Robustez:** allowlist, topes, abstención del RAG, manejo de errores de API/red.
- **Usabilidad:** CLI y Streamlit demuestran el mismo contrato; traza/debug comprensible.

> Cerrar el plan **sin HITL**, inventar datos que deberían venir de tools, o sustituir el RAG completo por un mini-RAG léxico son errores graves.

> Los nombres de ficheros/funciones del enunciado son **orientativos**: se evalúa comportamiento y tecnología (LangGraph, HITL, RAG completo, Streamlit), no la nomenclatura exacta.

### Presentación · equipo · 23 %

- Expresión oral clara y repartida.
- Soporte visual legible.
- Narrativa con hilo conductor.
- **Demo en vivo** con HITL (y, si cabe, multimodal) sin bloqueos irrecuperables.

### Git · equipo · 17 %

- Repo limpio, README completo, ramas/PRs coherentes, `main` estable.
- Reproducibilidad: clonar → `.env` → indexar RAG (si aplica) → CLI / Streamlit.

### Trabajo en equipo · individual · 15 %

Según historial Git: volumen, consistencia, calidad de commits/PRs, revisión de compañeras/os.

> Sin presencia objetiva en Git, esta área puede no superarse de forma individual.

### Memoria · equipo · 10 %

Informe con decisiones reales (grafo, tools, HITL, multimodal, fallos), no relleno.

### Extras · opcional · +10 %

p. ej. `SqliteSaver`, más tools, experimentos documentados, mejoras de UX, o un **uso sostenido de SDD** en el proyecto (evidencia en el repo). Los extras **no** sustituyen los requisitos obligatorios.

**En resumen:** un proyecto sólido orquesta un multiagente con LangGraph, usa un RAG completo como tool, pausa en HITL (CLI y Streamlit), muestra multimodal y traza, y entrega código modular listo para MLOps.
