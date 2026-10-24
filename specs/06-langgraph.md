# 06 — Especificación definitiva de LangGraph · ReciclaTIA PB02

**Estado:** APROBADA PARA IMPLEMENTACIÓN (decisiones de producto confirmadas el 24/10/2026).  
**Rama de trabajo:** `feature/langgraph`. **Base verificada:** `e42fc097281b4963df32fd4cb2f3f1549909c5a4`.  
**Ámbito:** grafo multiagente, estado, LLM, ejecución supervisada de herramientas, HITL y contrato común para CLI y Streamlit.  
**Fuentes normativas:** `docs/referencias/pb02_enunciado.md`, `docs/referencias/pb01_enunciado.md`, `AGENTS.md`, `specs/01-retrieval.md`, `specs/03-generacion.md`, `specs/04-consultas-generales.md`, `specs/05-tools.md`.  
**Referencia pedagógica, no código de producción:** `docs/referencias/sprint13_ejemplo/`.

## 1. Decisiones cerradas y propósito

ReciclaTIA reutiliza **el RAG completo del proyecto anterior**, sobre la gestión de residuos de **Madrid ciudad**. **La continuidad y la temática están autorizadas por los profesores** (confirmación del responsable del proyecto, 24/10/2026). El PB02 añade agentes sobre el RAG; **no lo reemplaza ni reimplementa**.

El sistema debe responder a consultas sobre reciclaje y puntos limpios de Madrid, pedir aclaraciones cuando falte información, utilizar las cuatro herramientas existentes y proponer un plan o pasos concretos **siempre sujeto a aprobación o rechazo humano antes de cerrar**.

**Regla de selección confirmada:** interviene **solo el especialista necesario**; intervienen **ambos, en secuencia**, cuando el pedido requiere clasificación **y** información de puntos limpios. La intervención dual no es el comportamiento por defecto.

Este documento cierra el diseño de los contratos de la fase LangGraph y anticipa las necesidades de **CLI, Streamlit, informe y demo**. La implementación es progresiva y archivo por archivo, pero cada módulo se desarrollará con su responsabilidad y API final previstas; no se crearán versiones transitorias destinadas a ser sustituidas en pasos posteriores.

## 2. Reglas arquitectónicas no negociables

1. **`config.py` es la única fuente de configuración operativa:** modelos, temperaturas, rutas, límites, timeouts, presupuestos, zona horaria y opciones ajustables. Los módulos consumen valores importados desde `config.py`; no definen defaults operativos divergentes ni leen por su cuenta valores de configuración.
2. **Una responsabilidad, un propietario:** el RAG resuelve recuperación/generación; las tools exponen capacidades; el ejecutor central valida/invoca tools; los nodos deciden contenido; el grafo conecta nodos; `agent.py` gestiona turnos/HITL; la CLI y la futura UI solo presentan e invocan el contrato.
3. **Sin lógica duplicada:** no se replicarán validadores de coordenadas, consultas ChromaDB, normalizaciones, prompts equivalentes, parseo de resultados, gestión de claves ni la ejecución allowlist en varios módulos. Se reutilizarán APIs existentes mediante imports.
4. **No alterar los contratos validados:** `src.rag.responder()` conserva sus campos (`respuesta`, `abstencion`, `motivo`, `fuentes`, `chunks`, `metricas`) y las tools conservan su API pública. Los comandos `python main.py rag ...` continúan funcionando.
5. **Índice y corpus protegidos:** no modificar el pipeline offline, el modelo/dimensiones de embeddings, las reglas del RAG ni reconstruir ChromaDB como consecuencia de introducir agentes.
6. **Seguridad y evidencia:** las tools se ejecutan únicamente con allowlist y validación Python. Un texto generado por Gemini, una imagen o un resultado documental no puede imponer instrucciones al orquestador. No se inventan destinos, mediciones, horarios, coordenadas, fuentes ni disponibilidad en tiempo real.
7. **Ciclo SDD:** especificación → implementación acotada de un archivo → pruebas de su contrato y regresión pertinente → revisión → siguiente archivo. No hacer commits, merges ni push sin autorización expresa.

**Nota de mantenimiento:** «definitiva» significa que esta especificación cubre los requisitos conocidos y no prevé reescrituras funcionales durante la integración de CLI/Streamlit. No garantiza inmunidad ante errores reales de bibliotecas o requisitos nuevos; cualquier cambio excepcional deberá documentarse y verificarse antes de editar código.

## 3. Funcionalidad y límites del dominio

**Incluido:** clasificación y destino de residuos basado en el RAG; consulta de tipos de puntos limpios, paradas y horarios documentados; búsqueda de puntos cercanos **solo con latitud y longitud aportadas voluntariamente**; explicación de objetos de una imagen de `assets/`, seguida de consulta al RAG si se pide destino; respuestas fundamentadas, fuentes, aclaraciones, abstención y planes revisables.

**Excluido:** otros municipios, geocodificación automática de direcciones o IP, rutas y tiempos de transporte, predicción de apertura real, escritura en servicios externos y acciones físicas. La hora actual no demuestra que una instalación esté abierta. No atribuir exhaustividad a un catálogo o a cinco chunks cuando la fuente no la acredita.

**Ejemplos de routing exigido:**

| Pedido | Especialistas | Herramientas potenciales |
|---|---|---|
| «¿Dónde tiro una cápsula de café?» | `residuos` | `consultar_rag` |
| «¿Qué significa punto limpio móvil?» | `puntos` | `consultar_rag` |
| «Busca puntos limpios a 2 km de estas coordenadas» | `puntos` | `buscar_puntos_limpios_cercanos` |
| «¿Dónde tiro esta batería y qué punto limpio tengo cerca?» | `residuos` **y** `puntos` | RAG y, si hay coordenadas, API geográfica |
| «Describe esta imagen e indica dónde reciclar el objeto» | `residuos` | `describir_imagen` y **después** `consultar_rag` |
| «¿Qué punto limpio tengo cerca?» sin coordenadas | `puntos` | Aclaración; **no** invocar la búsqueda espacial |

Una petición compuesta no se considera resuelta después del primer especialista. Si un aspecto no tiene datos suficientes, se conserva esa limitación y se puede responder a la parte fundamentada sin inventar la restante.

## 4. Grafo: nodos, flujo y reglas de transición

```text
START
  │
  ▼
supervisor ─────► residuos ────────┐
  │                                 │
  ├─────────────► puntos ───────────┤
  │                                 ├────► supervisor (reevalúa lo pendiente)
  └─────────────► sintetizar ───────┘
  │
  └── FINISH ───► [INTERRUPT ANTES DE aplicar] ──► aplicar ──► END
                                                  ▲
                                         decisión humana externa
```

**`supervisor`:** calcula una única transición válida en cada visita, usando el contenido del pedido, las capacidades requeridas y lo ya completado. Puede utilizar el LLM para seleccionar, pero **Python valida** la opción y aplica fallback. No realiza clasificación de residuos, consultas geográficas ni respuesta final.

**`residuos`:** recopila evidencia sobre materiales, fracciones y contenedores. Para clasificaciones concretas debe obtener evidencia de `consultar_rag`; si hay imagen pertinente, puede invocar `describir_imagen`, pero **nunca convertir solo la visión en recomendación de contenedor**. Devuelve notas y evidencia estructurada.

**`puntos`:** aclara ubicación, tipos, paradas y horarios; puede consultar el RAG y la API geográfica. Solo invoca `buscar_puntos_limpios_cercanos` cuando existen coordenadas válidas recibidas explícitamente. La ausencia de coordenadas produce pregunta de aclaración, no una localización inventada. CLI y Streamlit recogerán latitud/longitud como campos estructurados; el grafo no intentará geocodificar direcciones ni extraer coordenadas arbitrariamente de texto libre.

**`sintetizar`:** combina exclusivamente resultados disponibles, conserva fuentes/abstención/errores parciales, escribe una respuesta comprensible y un `plan` de **dos o más pasos**. Si falta evidencia, los pasos pueden consistir en aclarar o verificar: nunca deben presentar una hipótesis como un destino confirmado. No invoca tools.

**`aplicar`:** después del HITL, si la decisión es aprobar, establece `done=True`; si es rechazar, deja `done=False`, retira el plan vigente y permite reformular. **No realiza operaciones externas ni cambios municipales**: «aplicar» significa cerrar o devolver el plan al ciclo de revisión.

**Transiciones:** `START → supervisor`; desde `supervisor` solo a `residuos`, `puntos`, `sintetizar` o `aplicar` (por `FINISH`); desde `residuos`, `puntos` y `sintetizar` vuelve a `supervisor`; `aplicar → END`. **Sin ejecución paralela** de especialistas: facilita un único escritor de estado por paso y un presupuesto compartido comprobable.

**Orden de especialistas:** el supervisor atiende exclusivamente los necesarios. Si se requieren ambos, el orden preferente es `residuos → puntos` para poder utilizar la clasificación como contexto geográfico, sin que ello obligue a invocar una tool irrelevante. Se ejecuta `sintetizar` una vez reunida la evidencia requerida (o registradas sus carencias). `FINISH` únicamente es válido si ya existe respuesta y plan; de lo contrario se redirige a `sintetizar`.

## 5. Propiedad de módulos y contratos estables

Cada elemento tiene **un único módulo responsable**. Este mapa cubre también las funcionalidades que consumirán CLI y Streamlit posteriormente.

| Archivo | Propiedad exclusiva / contrato |
|---|---|
| `config.py` | Todos los parámetros operativos y política configurable de presupuestos/modelos; sin lógica de dominio |
| `src/gemini_client.py` | Obtención única y segura de credencial Gemini y cliente nativo; conservar `crear_cliente_gemini()` |
| `src/llm.py` (nuevo) | Adaptador `ChatGoogleGenerativeAI` que usa la credencial compartida y configuración global |
| `src/state.py` (nuevo) | Tipos de entrada/ubicación/estado grafo/producto, valores iniciales y adaptación grafo → producto |
| `src/agent_prompts.py` (nuevo) | Prompts especializados y de supervisión/síntesis, sin valores operativos propios |
| `src/agent_routing.py` (nuevo) | Validación de rutas, fallback determinista y comprobación de trabajo pendiente; no llama tools |
| `src/agent_tool_loop.py` (nuevo) | Un solo mini-bucle modelo ↔ tools, enforcement de límites, respuesta tool y traza estructurada |
| `src/nodes.py` (nuevo) | Cinco funciones de nodo: `supervisor`, `residuos`, `puntos`, `sintetizar`, `aplicar` |
| `src/graph.py` (nuevo) | `build_graph()`, `get_app()`, aristas, checkpointer, `config_hilo()` y consulta del interrupt |
| `src/agent.py` (nuevo) | `procesar_turno()`, `aprobar_plan()`, `nuevo_thread_id()`, `run_demo()` y gestión del ciclo conversacional |
| `src/tools/__init__.py` (existente) | **Único** registro allowlist y validación/ejecución de herramientas: ampliar con salida estructurada **sin romper** `ejecutar_tool()` |
| `src/rag.py` (existente) | Única API del RAG: `responder()`; **no se modifica** por el grafo |
| `main.py` (existente) | Despacho CLI `rag` y `agente` sin implementar lógica del grafo |
| `src/cli_agent.py` (nuevo) | Presentación CLI, demo, modo interactivo, comandos de aprobación/rechazo y límites de sesión |
| `app.py` (futuro) | Solo cliente Streamlit sobre las APIs de `src/agent.py`; chat, HITL, debug, nuevo hilo e input opcional de imagen/coordenadas |

**No se crearán** funciones alternativas de retrieval, invocación de tools, lectura `.env`, gestión del `thread_id` o normalización de ubicación dentro de los nodos, `main.py` ni `app.py`. Los prompts permanecen separados de la configuración porque representan instrucciones de negocio, no parámetros ajustables del entorno.

### 5.1. Configuración prevista en `config.py`

Se conservarán todas las constantes actuales del RAG y de las tools. Se incorporarán **en un único paso** los parámetros necesarios para todos los módulos futuros. La configuración inicial **cerrada** será:

| Parámetro | Valor de partida | Sentido |
|---|---|---|
| `AGENT_MODEL` | `GENERATION_MODEL` | Mismo proveedor/modelo configurado, sin duplicar su identificador |
| `AGENT_TEMPERATURE` | `0` | Respuestas estables durante evaluación |
| `SUPERVISOR_TEMPERATURE` | `0` | Enrutamiento reproducible |
| `MAX_ITER` | `8` | Máximo de decisiones del supervisor por pedido |
| `MAX_STEPS_DEFAULT` / `MAX_STEPS_LIMIT` | `4` / `8` | Rondas LLM por especialista: valor habitual / techo |
| `MAX_TOOL_CALLS_PER_SPECIALIST` | `8` | Tope de ejecuciones por especialista y pedido |
| `MAX_TOOL_CALLS_PER_TURN` | `12` | Tope agregado de herramientas por pedido |
| `MAX_TURNS_DEFAULT` / `MAX_TURNS_LIMIT` | `6` / `12` | Pedidos de usuario por conversación; HITL no suma |
| `MAX_HISTORY_TURNS` | `6` | Contexto conversacional acotado para revisiones |
| `TRACE_PREVIEW_MAX_CHARS` | `160` | Límite de previsualización segura por llamada |

Estos valores serán constantes en **`config.py`**, no duplicaciones literales dentro de funciones. La asociación de tools autorizadas por rol se centralizará también en `config.py` como política de autorización, verificando que cada nombre pertenece a la allowlist global. Cambiar una constante en el futuro sería una modificación de configuración, **no de arquitectura ni de contratos**.

Los argumentos opcionales de CLI/Streamlit podrán **reducir o ajustar dentro de límites seguros** los presupuestos predeterminados, pero nunca eliminar los máximos impuestos por Python.

### 5.2. Credenciales y estructura de tool result

La función actual `ejecutar_tool(nombre, args)` devuelve texto y **mantendrá ese contrato** para compatibilidad con sus pruebas. Se añadirá, **en el mismo registro de tools**, una función de ejecución estructurada, p. ej. `ejecutar_tool_estructurada(nombre, args)`, con resultado interno `{status, data, error}`: permite conservar los diccionarios de `consultar_rag` y de la API geográfica **sin convertirlos a cadenas y volverlos a interpretar**. La validación de nombre, argumentos y allowlist será **única**, reutilizada por ambas interfaces.

El adaptador LLM empleará una función compartida de acceso a `GEMINI_API_KEY` de `src/gemini_client.py`; no repetirá lógica de `.env` ni abrirá un cliente nativo para leer una clave. Nunca se incluirán credenciales en trazas ni excepciones presentadas.

## 6. Contrato de entrada y estado

**API pública definitiva del orquestador**, utilizable igual desde CLI y Streamlit:

```python
procesar_turno(
    estado: EstadoProducto,
    mensaje: str,
    thread_id: str,
    *,
    max_steps: int | None = None,
    ubicacion: Ubicacion | None = None,
    ruta_imagen: str | None = None,
) -> EstadoProducto

aprobar_plan(
    thread_id: str,
    aprobado: bool,
    estado: EstadoProducto,
) -> EstadoProducto

nuevo_thread_id() -> str
```

Se mantendrán también `crear_estado_producto()`, `run_demo()`, `get_app()` y `config_hilo(thread_id)` como contratos compartidos. `Ubicacion` es un valor opcional tipado con `latitud` y `longitud`, no una ubicación calculada automáticamente. `ruta_imagen`, si se aporta, debe ser un archivo permitido de `assets/` y validado por la tool existente. **Así Streamlit podrá añadirse sin cambiar las firmas.**

El `EstadoGrafo` tendrá los siguientes grupos de campos, con tipos concretos definidos en `src/state.py`:

| Grupo | Campos | Semántica |
|---|---|---|
| Pedido | `pedido`, `pedido_original`, `contexto_revision` | Turno actual, petición origen y contexto mínimo de un rechazo previo |
| Inputs | `ubicacion`, `ruta_imagen` | Datos opcionales aportados explícitamente |
| Selección | `necesita_residuos`, `necesita_puntos`, `residuos_completado`, `puntos_completado`, `siguiente`, `iteracion` | Determinan qué trabajo sigue sin repetir especialistas |
| Evidencia | `notas_residuos`, `notas_puntos`, `evidencias`, `fuentes`, `abstencion`, `motivo`, `limitaciones`, `estado_evidencia` | Resultados verificables, abstenciones y soporte completo/parcial/insuficiente |
| Salida | `respuesta`, `plan` | Texto y pasos sometidos a HITL |
| Control | `max_steps`, `tool_calls_turno`, `plan_aprobado`, `pendiente_hitl`, `done`, `error` | Presupuestos y estado de operación |
| Observabilidad | `log`, `traza_tools`, `historial_turnos` | Decisiones, tools y contexto acotado para revisión conversacional |

`EstadoProducto` expone a los clientes `respuesta`, `plan`, `fuentes`, `abstencion`, `motivo`, `limitaciones`, `estado_evidencia`, `error`, `done`, `pendiente_hitl`, `plan_aprobado`, `log`, `traza_tools`, `traza`, `thread_id` y los campos de contexto necesarios. La función **única** de adaptación vive en `src/state.py`. Ni CLI ni Streamlit reinterpretarán checkpoints ni duplicarán reglas de `done`.

**Semántica de evidencia agregada:** `evidencias` conserva cada resultado de tool sin alterarlo. `estado_evidencia` toma uno de `suficiente`, `parcial` o `insuficiente`. `limitaciones` contiene lo que no se ha podido verificar. La `abstencion` **global** es verdadera si no hay una respuesta sustantiva sustentada para el pedido; una respuesta parcialmente fundamentada puede conservar `abstencion=False` y `estado_evidencia="parcial"`, exponiendo expresamente sus limitaciones. Las abstenciones propias de cada tool RAG se mantienen **sin sobrescribir ni reinterpretar sus valores originales**.

**Escrituras y resets:** como el flujo es secuencial, `log` y `traza_tools` serán listas sustituidas de forma controlada por su propietario (sin reducers de concatenación que dificulten los resets). Los nodos añaden una sola vez sus eventos mediante helpers comunes; entre nuevos pedidos se reinician **todos** los campos de trabajo del turno (incluidos contadores, rutas, notas, evidencias, plan y aprobación), conservando solo `historial_turnos` acotado y `contexto_revision` cuando existe rechazo. No se reutilizan notas antiguas como si fueran pruebas de una petición nueva.

**Continuidad tras rechazo:** el nuevo pedido podrá expresar «cambia el distrito» o «hazlo sin imagen». Para entenderlo se conservará el pedido previo y el rechazo en `contexto_revision`/historial, pero **se volverán a obtener las evidencias necesarias**; no se reutiliza automáticamente una respuesta rechazada.

## 7. Supervisor, selección y fallback

La salida del supervisor será exactamente una de: `residuos`, `puntos`, `sintetizar`, `FINISH`. Toda salida inesperada, error del LLM o decisión inconsistente se sustituye por una ruta determinista mediante `src/agent_routing.py`.

La necesidad de especialistas se evalúa por el pedido (incluido contexto de revisión) y **se valida al terminar**: un pedido compuesto conserva pendiente al segundo especialista aunque el LLM proponga sintetizar antes. Si una intención no es concluyente, la regla segura exige aclaración o una respuesta abstentiva, no una herramienta arbitraria. El supervisor puede seleccionar un único especialista cuando basta para completar la consulta.

**Fallback obligatorio:**

1. Si falta la tarea requerida de residuos, devolver `residuos`.
2. Si falta la tarea requerida de puntos, devolver `puntos`.
3. Si las tareas requeridas están atendidas —incluidas aclaraciones y errores registrados— y no hay salida, devolver `sintetizar`.
4. Si existe respuesta y plan válido, devolver `FINISH`.
5. Al alcanzar `MAX_ITER`, terminar de forma segura con explicación de límite o síntesis restringida a evidencia presente, sin nuevos bucles y **sin** `done=True` automático. Nunca generar un FINISH que salte directamente a `aplicar` sin plan.

Se registrarán tanto la ruta solicitada por el modelo como la ruta ejecutada si hubo corrección. Las reglas de selección de dominio se implementarán una sola vez; los nodos consultan los flags, no reimplementan un clasificador de intenciones.

## 8. Ejecución de tools y preservación de evidencia

Tools reales y únicas: `consultar_rag`, `buscar_puntos_limpios_cercanos`, `hora_actual`, `describir_imagen`. La allowlist global continúa en `src.tools`; además, **cada especialista solo podrá ejecutar las tools autorizadas para su rol**.

El mini-bucle único de `src/agent_tool_loop.py` debe:

- validar las solicitudes devueltas por el LLM y ejecutarlas solo mediante la API central de tools;
- imponer en Python `max_steps` (rondas LLM) **y** los presupuestos separados de llamadas reales por especialista y por turno, incluidos los casos con varias llamadas simultáneas propuestas en una misma ronda;
- devolver para cada solicitud atendida una respuesta de tool coherente; las llamadas bloqueadas o excedidas no se ejecutan y quedan anotadas como tales;
- preservar en estructuras internas los resultados `dict`, especialmente `fuentes`, `abstencion` y `motivo` del RAG, y `fuente` (`api`/`corpus_local`) de la herramienta geográfica;
- convertir esos resultados a mensajes para el LLM solo mediante serialización controlada, **sin evaluarlos ni volver a parsear representaciones Python**;
- generar traza `{nodo, paso, tool, status, fuente, preview}` (con argumentos solo si no contienen datos sensibles), sin registrar secretos, imágenes ni payloads completos;
- diferenciar **error técnico**, **ausencia de resultados** y **abstención del RAG**.

La síntesis tiene prohibido presentar una clasificación sin evidencia de `consultar_rag`, una localización sin resultado geográfico válido o un estado «abierto ahora» derivado únicamente de `hora_actual`. Ante falta de datos debe pedir el elemento necesario o abstenerse y crear pasos de verificación, sin falsificar citas.

## 9. HITL y persistencia obligatorios

`build_graph()` compilará con `checkpointer=MemorySaver()` e `interrupt_before=["aplicar"]` (o alias compatible de la versión instalada). `get_app()` compartirá **una única instancia compilada por proceso**; los clientes identificarán cada conversación por `thread_id` único (`{"configurable": {"thread_id": tid}}`). `MemorySaver` persiste **solo durante la vida del proceso**, no tras un reinicio; no se afirmará persistencia permanente.

**Invariante:** `done=True` solamente puede escribirse en `aplicar` después de que el humano haya aprobado el plan. El LLM nunca decide `done`, la aprobación ni la reanudación.

Secuencia:

1. `procesar_turno(...)` valida mensaje y estado, invoca el grafo, consulta su snapshot y devuelve `pendiente_hitl=True` únicamente si `get_state(...).next` señala `aplicar`.
2. Durante la pausa no se admiten nuevos mensajes ordinarios, ni puede cerrarse por un `invoke()` no autorizado.
3. `aprobar_plan(...)` comprueba que hay una pausa real; usa `update_state(..., {"plan_aprobado": aprobado})` y solo entonces `invoke(None, config_hilo(thread_id))`.
4. **Aprobar:** `done=True`, conservar plan y trazas; el hilo queda cerrado para nuevas preguntas y el cliente ofrece nueva conversación.
5. **Rechazar:** `done=False`, plan no vigente, registro de rechazo y contexto de revisión conservados; después puede formularse otro turno **en el mismo hilo**.
6. Un `thread_id` nuevo no hereda datos de otro. El cliente no debe reutilizar IDs tras «Nueva conversación».

**Si ocurre un fallo técnico que impide obtener evidencia o producir una síntesis segura**, se devolverá un error identificable y recuperable sin fabricar plan, sin entrar en HITL ficticio ni marcar `done=True`. Los errores parciales no bloquearán la parte del pedido fundamentada cuando sea posible delimitarla.

**Si falla reanudar** no se mostrará un éxito ficticio. El estado visible reflejará el error y la posición real del checkpoint; no marcará `done=True` por anticipado. Un `aprobar_plan` sin pausa será operación inocua con mensaje explícito.

## 10. CLI, Streamlit y entrega previstos desde este diseño

La CLI mantiene `python main.py rag query|ask|index ...` y añade `python main.py agente demo` y `python main.py agente interactivo`, con opción de demo sin autoaprobar. El modo interactivo reconoce `aprobar` y `rechazar`; muestra respuesta, plan, fuentes, log, herramientas, límites y estado HITL. Un mensaje vacío no consume turno ni llama al grafo. El límite es por mensajes efectivos del usuario y se verifica antes de invocar LangGraph, no solo deshabilitando botones. `max_turns` se comprueba en el **orquestador** y se refleja en ambos clientes; la decisión HITL no cuenta como consulta nueva.

La futura Streamlit **solo** importará `procesar_turno`, `aprobar_plan`, `nuevo_thread_id` y `crear_estado_producto` y mostrará chat, herramientas, fuentes, flags, botones HITL y nueva conversación. La entrada estructurada ya admite `ubicacion` y `ruta_imagen` opcionales, por lo que no hará falta redefinir las firmas. Las imágenes se almacenarán solo en la ruta validada por la herramienta; la geolocalización será opt-in.

El informe `entregables/informe_decisiones.md` documentará diseño y selección de agentes, HITL aprobado y rechazado, cuatro tools, multimodal, trazas, tres fallos observados y siguientes pasos MLOps. La presentación de aproximadamente 10 minutos demostrará RAG, una abstención, un pedido con ambos especialistas, HITL, tools, persistencia de hilo y CLI/UI. El tema ReciclaTIA ya está confirmado como continuidad docente; no figura como pendiente.

## 11. Criterios de aceptación (obligatorios)

**Arquitectura y regresión**

- [ ] `src.rag.responder()` y las cuatro tools preservan firmas y tests; no se altera el índice, los embeddings ni el corpus.
- [ ] La suite previa de RAG/tools/CLI sigue pasando y el nuevo código usa configuración operativa únicamente de `config.py`.
- [ ] `main.py rag ...` conserva funcionamiento; `main.py agente ...` usa la misma API que Streamlit.
- [ ] El grafo compila con nodos y aristas exigidos, `MemorySaver`, `thread_id` e interrupción **antes** de `aplicar`.

**Especialistas y evidencia**

- [ ] Consulta solo residuos → **únicamente** `residuos` (no invocación gratuita de `puntos`).
- [ ] Consulta solo puntos/horarios → **únicamente** `puntos` (no invocación gratuita de `residuos`).
- [ ] Consulta compuesta → **ambos especialistas**, en secuencia; no se sintetiza antes de registrar ambos.
- [ ] Petición multimodal → descripción con visión + verificación RAG antes de recomendar destino.
- [ ] Sin coordenadas → no se llama la API espacial; con coordenadas voluntarias válidas → se consulta y distingue API/fallback.
- [ ] RAG in-corpus muestra fuentes reales; RAG fuera de corpus conserva abstención/motivo; fallos técnicos no se ocultan como abstenciones.
- [ ] Los resultados estructurados no se pierden al pasar por la tool; no hay fuentes inventadas, horarios atribuidos indebidamente ni inferencias visuales convertidas en hechos.

**Control y memoria**

- [ ] Supervisor inválido o fallido → fallback válido y trazado; no se reiteran especialistas completados.
- [ ] `MAX_ITER` impide bucles; `max_steps` limita rondas LLM y los topes de calls impiden exceder ejecuciones con múltiples tool_calls.
- [ ] Antes de aprobar: `pendiente_hitl=True`, `done=False` y `aplicar` no ejecutado.
- [ ] Aprobar → `done=True`; rechazar → `done=False`, elimina plan vigente y admite corrección en **el mismo hilo** sin arrastrar evidencia antigua.
- [ ] Nuevos mensajes se bloquean durante HITL; nuevos hilos están aislados; `MemorySaver` conserva estado entre invocaciones del mismo proceso.
- [ ] La CLI y Streamlit consumen el mismo contrato; `max_turns` se impone sin contar aprobaciones como turno.
- [ ] Mensaje vacío no invoca el grafo; los errores están saneados; trazas no contienen claves ni payloads sensibles.

**Ejecución verificable**

- [ ] Tests unitarios con mocks, sin red/cuota, para rutas individuales y compuestas, protección de tools, límites, estado, reanudación y rechazo.
- [ ] Tests de integración reales para LangGraph/checkpointer, CLI y herramientas; al menos una demostración completa con ambas especialidades.
- [ ] Antes de considerar terminada la fase, ejecutar `python -m pytest tests/ -q --tb=short` e informar resultados efectivos y pruebas omitidas.

## 12. Orden de desarrollo por archivo (sin implementaciones provisionales)

1. **`specs/06-langgraph.md`** — cerrar y guardar esta especificación.
2. **`config.py`** — incorporar de una vez todos los parámetros operativos del grafo, CLI y futura UI que deban estar centralizados.
3. **`src/gemini_client.py`** — añadir helper de credenciales sin romper el cliente actual (y su test correspondiente).
4. **`src/state.py`** — contrato definitivo de entrada, estado, fábrica y adaptación.
5. **`src/llm.py`** — único adaptador LangChain/Gemini.
6. **`src/agent_prompts.py`** — prompts de todos los roles.
7. **`src/agent_routing.py`** — reglas de selección/validación/fallback.
8. **`src/tools/__init__.py`** — retorno estructurado con compatibilidad y tests.
9. **`src/agent_tool_loop.py`** — ejecución común y presupuestos; tests de llamadas múltiples.
10. **`src/nodes.py`** — cinco nodos conectados a las dependencias anteriores.
11. **`src/graph.py`** — montar y compilar StateGraph, hilo e interrupt; tests estructurales.
12. **`src/agent.py`** — contratos públicos del turno/HITL; tests de aprobación/rechazo y hilos.
13. **`src/cli_agent.py`** — cliente CLI y pruebas de interacción.
14. **`main.py`** — extender parser y despacho, preservando comandos RAG; pruebas de regresión.
15. **`app.py`** — cliente Streamlit sin lógica de dominio ni orquestación; pruebas de interfaz donde proceda.
16. **`README.md` y `entregables/informe_decisiones.md`** — documentación y cierre, solamente sobre resultados demostrados.

Cada archivo se **presentará, explicará, revisará y dará por definitivo** antes de continuar. Las pruebas podrán añadirse junto al archivo que validan, pero no se adelantará el desarrollo de otros módulos ni se harán cambios remotos sin permiso. Si una prueba revela una incompatibilidad, se corrige y justifica explícitamente en el módulo propietario; no se creará una segunda implementación en otro archivo.
