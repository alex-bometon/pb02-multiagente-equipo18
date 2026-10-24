
"""Configuración global de ReciclaTIA PB02.

Centraliza los parámetros operativos del proyecto.
Las reglas de negocio, algoritmos y esquemas permanecen
en sus respectivos módulos.

Las credenciales se cargan desde .env.
"""

from pathlib import Path


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CSV_DIR = RAW_DIR / "csv"
HTML_DIR = RAW_DIR / "html"

ASSETS_DIR = BASE_DIR / "assets"

OUTPUT_DIR = BASE_DIR / "output"
CHUNKS_JSON = OUTPUT_DIR / "chunks.json"
EMBEDDINGS_JSON = OUTPUT_DIR / "embeddings.json"


# =========================================================
# FRAGMENTACIÓN DEL CORPUS
# =========================================================

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 100


# =========================================================
# EMBEDDINGS GEMINI
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

EMBED_BATCH_SIZE = 50
MAX_CHUNKS_EMBED = None

EMBED_BATCH_PAUSE_SECONDS = 32
EMBED_MAX_RETRIES = 5
EMBED_RETRY_SECONDS = 60


# =========================================================
# ÍNDICE VECTORIAL CHROMADB
# =========================================================

CHROMA_DIR = OUTPUT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "residuos_madrid"
CHROMA_DISTANCE = "cosine"

INDEX_BATCH_SIZE = 100


# =========================================================
# RECUPERACIÓN Y CONTEXTO RAG
# =========================================================

TOP_K = 5
MAX_TOP_K_RETRIEVAL = 10

MAX_PREGUNTA_CARACTERES = 500
MAX_CONTEXTO_CARACTERES = 12_000

EJEMPLOS_POR_DESTINO = 2


# =========================================================
# GENERACIÓN GEMINI
# =========================================================

GENERATION_MODEL = "gemini-3.5-flash-lite"

GENERATION_TEMPERATURE = 0
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_SECONDS = 2


# =========================================================
# HERRAMIENTA MULTIMODAL
# =========================================================

IMAGE_MODEL = GENERATION_MODEL

IMAGE_MAX_BYTES = 5 * 1024 * 1024
IMAGE_MAX_PIXELS = 16_000_000

IMAGE_ALLOWED_EXTENSIONS = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
})

IMAGE_ALLOWED_FORMATS = frozenset({
    "JPEG",
    "PNG",
})


# =========================================================
# API MUNICIPAL DE PUNTOS LIMPIOS
# =========================================================

API_PUNTOS_LIMPIOS_URL = (
    "https://sigma.madrid.es/hosted/rest/services/"
    "RESIDUOS/PUNTOS_LIMPIOS/MapServer/5/query"
)

API_TIMEOUT_SECONDS = 12
API_RESULT_RECORD_COUNT = 2_000


# =========================================================
# BÚSQUEDA GEOGRÁFICA
# =========================================================

GEO_DEFAULT_RADIUS_METERS = 2_000
GEO_MIN_RADIUS_METERS = 1
GEO_MAX_RADIUS_METERS = 5_000

GEO_MAX_RESULTS = 10


# =========================================================
# FECHA Y HORA
# =========================================================

TIMEZONE = "Europe/Madrid"



# =========================================================
# LANGGRAPH — MODELOS Y GENERACIÓN DE AGENTES
# =========================================================

# Reutilizamos el modelo Gemini configurado para el RAG.
AGENT_MODEL = GENERATION_MODEL

# Temperatura de los agentes y del supervisor.
AGENT_TEMPERATURE = 0
SUPERVISOR_TEMPERATURE = 0


# =========================================================
# LANGGRAPH — LÍMITES DE ORQUESTACIÓN
# =========================================================

# Máximo de decisiones del supervisor por consulta.
MAX_ITER = 8

# Máximo de rondas LLM por especialista.
MAX_STEPS_DEFAULT = 4
MAX_STEPS_LIMIT = 8

# Presupuestos de llamadas reales a herramientas.
# Una ronda LLM puede solicitar varias herramientas.
MAX_TOOL_CALLS_PER_SPECIALIST = 8
MAX_TOOL_CALLS_PER_TURN = 12

# Límite de pasos internos del grafo.
GRAPH_RECURSION_LIMIT = 2 * MAX_ITER + 8


# =========================================================
# CONVERSACIÓN — CLI Y STREAMLIT
# =========================================================

# Límites de consultas por conversación.
MAX_TURNS_DEFAULT = 6
MAX_TURNS_LIMIT = 12

# Contexto de turnos anteriores.
MAX_HISTORY_TURNS = 6

# Reutilizamos la validación de longitud del RAG.
AGENT_MAX_MESSAGE_CHARS = MAX_PREGUNTA_CARACTERES


# =========================================================
# OBSERVABILIDAD Y TRAZABILIDAD
# =========================================================

# Longitud máxima de las previsualizaciones de herramientas.
TRACE_PREVIEW_MAX_CHARS = 160


# =========================================================
# HERRAMIENTAS PERMITIDAS POR ESPECIALISTA
# =========================================================

# Restricciones adicionales a la allowlist global de src.tools.
# La ejecución y validación corresponden al módulo de herramientas.
AGENT_TOOLS_BY_ROLE = {
    "residuos": (
        "consultar_rag",
        "describir_imagen",
    ),
    "puntos": (
        "consultar_rag",
        "buscar_puntos_limpios_cercanos",
        "hora_actual",
    ),
}
