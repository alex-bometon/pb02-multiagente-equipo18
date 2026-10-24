# ReciclaTIA — PB02 Multiagente

Proyecto académico individual de AI Engineering. El objetivo es construir un
asistente sobre gestión de residuos domésticos de **Madrid ciudad**, con un RAG
independiente que posteriormente utilizará un sistema multiagente LangGraph.

> **Estado actual:** RAG completo y herramientas implementadas y probadas.
> Pendientes de integración: grafo multiagente LangGraph, HITL,
> CLI multiagente y Streamlit.

## Requisitos

- Python 3.11 o 3.12 recomendado por la configuración del proyecto.
- Dependencias de `requirements.txt`.
- Clave de la API de Google Gemini para generar embeddings y respuestas.
- Corpus local de CSV y HTML municipales en `data/raw/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y configura `GEMINI_API_KEY`. No subas este archivo a GitHub.
Gemini es un servicio propietario externo y puede estar sujeto a cuotas o
costes; **el funcionamiento no es íntegramente offline ni libre de proveedores**.

## Preparar el índice (pipeline offline)

Desde la raíz del repositorio:

```bash
python -m src.ingest
python -m src.embed
python -m src.index
```

El resultado se guarda en `output/chroma_db/`. Los ficheros del índice y los
embeddings generados son reproducibles y no deben añadirse al control de
versiones. Si cambias el modelo de embeddings o el corpus, regenera los
artefactos y el índice para evitar incompatibilidades.

## Consultar el RAG (sin interfaz)

```python
from src.rag import responder

resultado = responder("¿En qué contenedor tiro un envase de plástico?")
print(resultado["respuesta"])
print(resultado["fuentes"])
```

`responder()` devuelve `pregunta`, `respuesta`, `abstencion`, `motivo`,
`fuentes`, `chunks` y `metricas`. Las preguntas generales de catálogo y los
horarios de puntos limpios móviles pueden utilizar consultas estructuradas
sin generación Gemini. Para conocer horarios de móviles es necesaria una
parada, calle o distrito; no son horarios en tiempo real.

La versión actual **solo cubre Madrid ciudad**. No se debe extrapolar la
información del corpus a otros municipios.

## Organización del RAG

```text
src/
  load.py, clean.py, chunk.py, ingest.py  # Preparación del corpus
  artifacts.py, embed.py, index.py         # Embeddings y ChromaDB
  retrieve.py                             # Búsqueda semántica
  retrieve_structured.py                  # Catálogo y paradas móviles
  generate.py                             # Llamadas al modelo
  generation_validation.py               # Citas y contrato JSON
  rag.py                                  # API pública responder()
  rag_intents.py                          # Enrutamiento de consultas
  domain.py, input_validation.py,
  text_utils.py                           # Reglas compartidas
```

No hay dependencias de Streamlit dentro del RAG. El grafo LangGraph deberá
consumir `responder()` mediante una herramienta con allowlist, sin replicar
la lógica de recuperación o generación.

## Herramientas del agente

ReciclaTIA dispone de cuatro herramientas independientes, registradas mediante `@tool` de LangChain:

| Herramienta | Función |
|---|---|
| `consultar_rag` | Consultar el RAG completo de residuos de Madrid |
| `buscar_puntos_limpios_cercanos` | Consultar puntos limpios mediante la API municipal, con respaldo local en ChromaDB |
| `hora_actual` | Obtener la fecha y hora de Madrid |
| `describir_imagen` | Describir residuos visibles en imágenes JPEG o PNG ubicadas en `assets/` |

`src/tools/__init__.py` contiene la allowlist `TOOLS`, el registro `TOOL_BY_NAME` y la función `ejecutar_tool()`.

Las herramientas pueden ejecutarse independientemente de LangGraph. El grafo se encargará posteriormente de la selección, los límites de pasos y la trazabilidad.

Para comprobar el registro:

```bash
python -m pytest tests/test_tools_registry.py -v
```

Para ejecutar una herramienta sencilla:

```python
from src.tools import ejecutar_tool

print(ejecutar_tool("hora_actual"))
```

## Pruebas

```bash
python -m pytest tests/ -q --tb=short
python tests/evaluate_retrieve.py
python tests/evaluate_rag.py --ids 9 10 14
```

Los archivos de prueba grandes se han separado por responsabilidad.
Las pruebas de integración que requieren ChromaDB se omiten si el índice
local no está disponible. Las evaluaciones completas pueden llamar a Gemini
y consumir cuota; las estructuradas no necesitan generación.

## Requisitos del PB02 pendientes

El proyecto todavía necesita supervisor, al menos dos especialistas,
síntesis, aplicación supervisada por humano (HITL), `MemorySaver` y
`thread_id`, límites de pasos en Python, trazas y un cliente CLI con aprobación o
rechazo manual. La interfaz Streamlit se desarrollará después.

También quedan pendientes el informe de decisiones, la presentación y la
documentación definitiva de fuentes del corpus (URL, fecha de descarga,
licencia y derechos de reutilización). **No inventar estos datos**.

Consulta `docs/auditoria_tecnica.md` para el diagnóstico y el plan de cierre.

## Condiciones académicas y licencia

El enunciado PB02 exige aprobación docente para una temática distinta de
las cuatro propuestas. Debe confirmarse que reciclaje está autorizado y
cómo se adapta a un trabajo individual la exigencia de PRs revisadas por
miembros del equipo.

Todavía no se ha escogido una licencia de software. Para publicar como
software libre, habrá que añadir una licencia explícita y revisar las
licencias de las fuentes municipales y las dependencias.
