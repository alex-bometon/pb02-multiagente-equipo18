from pathlib import Path

# =========================================================
# RUTAS
# =========================================================

# Raíz del proyecto:
# pg-rag-equipo_04/
BASE_DIR = Path(__file__).resolve().parent

# Carpeta general de datos
DATA_DIR = BASE_DIR / "data"

# Datos originales, sin modificar
RAW_DIR = DATA_DIR / "raw"

# Tipos de fuente
CSV_DIR = RAW_DIR / "csv"
HTML_DIR = RAW_DIR / "html"

# Salidas generadas por el pipeline
OUTPUT_DIR = BASE_DIR / "output"

CHUNKS_JSON = OUTPUT_DIR / "chunks.json"


# =========================================================
# CHUNKING
# =========================================================

# Valores iniciales para la primera versión.
# Seleccionados tras comparar distintas estrategias.
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 100


# =========================================================
# EMBEDDINGS
# =========================================================

EMBEDDINGS_JSON = OUTPUT_DIR / "embeddings.json"

EMBEDDING_MODEL = "gemini-embedding-2"

EMBEDDING_DIMENSIONS = 768

# Tamaño de los grupos procesados por nuestro código.
# No corresponde a la Batch API de Google.
EMBED_BATCH_SIZE = 50

# Durante las primeras pruebas se procesará
# únicamente una parte del corpus.
# None = procesar todos los chunks.
MAX_CHUNKS_EMBED = None

# Pausa entre lotes para respetar el límite del Free Tier.
# Con 50 embeddings por lote y un máximo de 100/minuto,
# dejamos margen respecto al límite.
EMBED_BATCH_PAUSE_SECONDS = 32

# Reintentos adicionales si la API devuelve 429.
EMBED_MAX_RETRIES = 5

# Espera tras alcanzar temporalmente la cuota.
EMBED_RETRY_SECONDS = 60


# =========================================================
# CHROMADB
# =========================================================

# Base de datos vectorial persistente.
CHROMA_DIR = OUTPUT_DIR / "chroma_db"

# Nombre de la colección utilizada por el RAG.
CHROMA_COLLECTION_NAME = "residuos_madrid"

# Métrica utilizada para comparar embeddings.
CHROMA_DISTANCE = "cosine"

# Número máximo de registros que intentaremos insertar
# en una misma operación.
INDEX_BATCH_SIZE = 100


# =========================================================
# RETRIEVAL
# =========================================================

# Número final de chunks que se entregan al RAG.
#
# Este valor fue seleccionado durante la evaluación inicial
# del retrieval comparando distintos valores de K.
TOP_K = 5


# ---------------------------------------------------------
# RECUPERACIÓN DE CANDIDATOS
# ---------------------------------------------------------

# Los siguientes parámetros se conservan para reproducir
# los experimentos de reranking híbrido.
#
# El flujo RAG final utiliza como baseline el retrieval
# semántico original, definido en src/rag.py.
RETRIEVAL_MIN_CANDIDATES = 50


# ---------------------------------------------------------
# RERANKING HÍBRIDO
# ---------------------------------------------------------

# Peso de la similitud semántica en el score final.
RETRIEVAL_SEMANTIC_WEIGHT = 0.70

# Peso de la coincidencia léxica en el score final.
RETRIEVAL_LEXICAL_WEIGHT = 0.30


# ---------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------

# Dataset canónico utilizado para evaluar tanto retrieval
# como el flujo RAG completo.
QUERIES_DIR = BASE_DIR / "queries"

EVAL_QUERIES_JSON = QUERIES_DIR / "eval_queries.json"


# ---------------------------------------------------------
# CACHÉ DE EVALUACIÓN
# ---------------------------------------------------------

# Artefactos temporales/reutilizables generados durante
# las evaluaciones.
#
# Se mantienen fuera de queries/ porque no forman parte
# del dataset de evaluación.
CACHE_DIR = OUTPUT_DIR / "cache"

# Caché persistente de embeddings de las preguntas.
#
# Evita volver a llamar a la API de embeddings cada vez
# que se repiten experimentos de retrieval o distintos K.
EVAL_QUERY_EMBEDDINGS_JSON = CACHE_DIR / "eval_query_embeddings.json"


# ---------------------------------------------------------
# RESULTADOS DE EVALUACIÓN
# ---------------------------------------------------------

# Resultados producidos por las evaluaciones end-to-end.
#
# Se mantienen separados tanto del dataset de entrada
# como de los artefactos principales del pipeline.
EVALUATION_RESULTS_DIR = BASE_DIR / "tests" / "evaluation"

# =========================================================
# GENERACIÓN
# =========================================================

# Modelo utilizado para generar la respuesta final del RAG.
GENERATION_MODEL = "gemini-3.5-flash-lite"
# gemini-3.5-flash
# gemini-3.5-flash-lite
# gemini-3.6-flash
# gemini-3.1-flash-lite
# gemini-2.5-flash

# Número máximo de intentos ante errores temporales de la API.
GENERATION_MAX_RETRIES = 3


# Espera inicial entre reintentos.
#
# Se aplica backoff exponencial:
#
#   intento 1 -> 2 segundos
#   intento 2 -> 4 segundos
#   intento 3 -> error definitivo
GENERATION_RETRY_SECONDS = 2
