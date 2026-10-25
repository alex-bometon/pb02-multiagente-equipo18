# ReciclaTIA — PB02 Multiagente con LangGraph

ReciclaTIA es un asistente académico para consultas sobre **residuos domésticos y puntos limpios de Madrid ciudad**. Integra un RAG completo basado en ChromaDB, un grafo multiagente LangGraph con supervisión humana (HITL), herramientas externas y una interfaz CLI y Streamlit.

**Alcance:** información documentada de Madrid ciudad. No geocodifica direcciones, no determina rutas reales, no puede garantizar la apertura actual de una instalación y no ejecuta acciones físicas. `Aprobar` acepta el plan y finaliza el hilo; no desencadena un segundo ciclo de herramientas.

## Requisitos

- Python 3.11 o 3.12 recomendado (el entorno del equipo debe verificarse).
- Dependencias declaradas en `requirements.txt`.
- Clave de Gemini en `.env`, no versionado.
- Corpus municipal CSV y HTML en `data/raw/` e índice ChromaDB local construido.
- Conexión a Internet para Gemini y, cuando sea posible, la API municipal.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y configura `GEMINI_API_KEY`. Nunca añadas `.env` al repositorio ni grabes la clave en la presentación. Gemini es un servicio externo con cuotas y posibles costes.

## Preparar el RAG (pipeline offline)

Desde la raíz del proyecto:

```bash
python -m src.ingest
python -m src.embed
python -m src.index
```

El índice ChromaDB se almacena en `output/chroma_db/`. No vuelvas a generar embeddings por rutina; el pipeline offline puede consumir cuota y requiere consistencia entre corpus, modelo y dimensión de embeddings.

## Ejecutar la CLI

```bash
python main.py --help
python main.py rag query "¿Dónde tiro una botella de plástico?"
python main.py rag ask "¿Dónde tiro una botella de plástico?"
python main.py agente demo "¿Dónde van las pilas?"
python main.py agente interactivo --max-turns 6 --max-steps 4
```

En `demo`, el HITL solicita `aprobar`, `rechazar` o `salir`. En `interactivo`, la misma conversación admite rechazo y nuevo mensaje; `aprobar` cierra el hilo y `/nuevo` crea un identificador distinto. Escribe `/ayuda` para consultar todos los comandos.

Para una búsqueda geográfica con coordenadas voluntarias:

```bash
python main.py agente demo \
  "Busca puntos limpios de proximidad cerca de mí" \
  --latitud 40.4168 --longitud -3.7038
```

Para multimodalidad, siempre con archivos permitidos dentro de `assets/`:

```bash
python main.py agente demo \
  "Identifica el objeto y dime dónde se recicla" \
  --imagen assets/capsula.jpg
```

## Ejecutar Streamlit

Una vez copiado `app.py` a la raíz:

```bash
python -m streamlit run app.py
```

Usa el chat para consultar residuos y servicios; habilita **Aportar coordenadas** solo cuando quieras realizar una búsqueda geográfica. Puedes adjuntar PNG o JPEG (máximo 5 MiB y 16 megapíxeles); se almacena temporalmente dentro de `assets/` para la herramienta y se borra al finalizar el turno.

**Ajustes de la barra lateral:** máximo de turnos al inicio de cada conversación, límite `max_steps` para el siguiente turno, ubicación voluntaria, mapa y depuración. No son controles para cambiar la clave Gemini, el modelo, el `TOP_K` del RAG ni límites globales no expuestos por el contrato del agente.

**HITL:** la aplicación ofrece **Aprobar/Rechazar** en barra lateral y en el área central. Mientras existe un plan pendiente, no acepta nuevas consultas. Tras aprobar, `done=True` y el hilo se cierra. Tras rechazar, `done=False` y puedes reformular dentro del mismo `thread_id`.

En el panel de depuración se observan `thread_id`, flags, supervisor, especialistas, herramientas y trazas. `MemorySaver` **no es persistencia en disco**; el estado del hilo se conserva solo durante la vida del proceso de servidor.

## Arquitectura

```text
CLI (main.py + src/cli_agent.py)       Streamlit (app.py)
                   \                    /
                    \                  /
                      src/agent.py
                            |
                    src/graph.py
                            |
             supervisor (src/nodes.py)
                       /         \
                 residuos        puntos
                       \         /
                        sintetizar
                            |
                  HITL (interrupt_before)
                            |
                          aplicar
                            |
                          done
```

- `src/state.py`: contratos públicos de estado, presupuesto y turnos.
- `src/agent_routing.py`: selección determinista y fallback del supervisor.
- `src/agent_tool_loop.py`: selección y ejecución supervisada de herramientas.
- `src/agent_prompts.py`: prompts separados por rol.
- `src/nodes.py`: supervisor, especialistas, síntesis y aplicación.
- `src/graph.py`: `StateGraph`, `MemorySaver` y punto HITL.
- `src/agent.py`: interfaz compartida por CLI y Streamlit (`procesar_turno`, `aprobar_plan`, `nuevo_thread_id`, `run_demo`).

## RAG completo como herramienta

El corpus se carga, limpia, fragmenta e indexa con Gemini embeddings y ChromaDB:

```text
src/load.py + clean.py → chunk.py / ingest.py
  → embed.py → index.py → retrieve.py
  → generate.py + generation_validation.py → rag.py
```

`src.rag.responder(pregunta) -> dict` sigue siendo utilizable sin interfaz, con `respuesta`, `abstencion`, `motivo`, `fuentes`, `chunks` y métricas. La herramienta `consultar_rag` lo expone al sistema LangGraph. La salida de Gemini se valida: no se aceptan citas fuera del contexto ni respuestas afirmativas sin soporte.

## Herramientas y permisos

| Herramienta | Rol autorizado | Función |
|---|---|---|
| `consultar_rag` | residuos y puntos | Información fundamentada en documentos |
| `describir_imagen` | residuos | Descripción visual con Gemini; no asigna contenedor por sí sola |
| `buscar_puntos_limpios_cercanos` | puntos | API municipal ArcGIS + fallback local; distancia Haversine |
| `hora_actual` | puntos | Hora de Madrid usando `Europe/Madrid` |

La allowlist está en `src/tools/__init__.py`. Python impone `MAX_ITER`, `max_steps`, `max_turns`, los presupuestos de herramientas y los permisos del rol. El modelo no decide la aprobación humana ni `done`.

Los resultados geográficos indican `fuente=api` o `fuente=corpus_local`. Una respuesta de respaldo no equivale a comprobación en tiempo real. La distancia es **geográfica en línea recta**, no caminando ni conduciendo; una hora actual no prueba por sí sola que el punto esté abierto.

## Pruebas

```bash
python -m py_compile app.py
python -m pytest tests/ -q --tb=short
python tests/evaluate_retrieve.py
python tests/evaluate_rag.py --ids 9 10 14
```

Los tests aislados no realizan llamadas externas, salvo los scripts de evaluación/integración que lo indiquen; comprobar la cuota antes de lanzarlos. Para una demostración end-to-end real, consultar la matriz manual de pruebas en `GUIA_PRUEBAS_Y_ENTREGA.md` (si se incorpora al repositorio).

## Limitaciones conocidas y trabajo futuro

- La generación JSON de Gemini puede incumplir el contrato de citas y provocar un fallo recuperable, sin inventar evidencia.
- La síntesis presenta horarios únicamente cuando los recupera y considera pertinentes; no garantiza apertura actual.
- `MemorySaver` no almacena conversaciones de forma permanente.
- El HITL confirma/rechaza un plan, **no ejecuta nuevas herramientas después de aprobar**; un segundo ciclo ejecutable sería una mejora de producto y requeriría ampliar la especificación y las pruebas.
- La clasificación documentada puede requerir aclaraciones ante materiales o subtipos ambiguos.

## Fuentes, atribuciones y licencia

El corpus contiene documentos municipales CSV/HTML en `data/raw/`. **Antes de entregar, completar esta sección con las URLs concretas, fecha de consulta/descarga y condiciones de reutilización efectivas de cada archivo**, sin inventar estos datos. La herramienta geográfica consulta:

`https://sigma.madrid.es/hosted/rest/services/RESIDUOS/PUNTOS_LIMPIOS/MapServer/5/query`

**Pendiente de confirmar:** aprobación docente del tema reciclaje (fuera de las cuatro temáticas orientativas del enunciado) y adaptación del requisito de PR por miembro a un proyecto individual. También está pendiente la licencia explícita del código.

## Informe y presentación

El entregable `entregables/informe_decisiones.md` documenta las decisiones de diseño, las cuatro herramientas, el HITL aprobado/rechazado, los aciertos, tres fallos y siguientes pasos MLOps. La presentación debe demostrar RAG, tools, grafo, HITL, CLI/Streamlit y, si la cuota lo permite, la herramienta multimodal. El enunciado admite un único vídeo de presentación por grupo.
