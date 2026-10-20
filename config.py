"""Configuración global de ReciclaTIA PB02.

Todas las rutas parten de la raíz del proyecto. No guardar secretos aquí:
las credenciales se cargan desde .env mediante gemini_client.py.
"""

from pathlib import Path

# Rutas del proyecto y datos del corpus
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CSV_DIR = RAW_DIR / "csv"
HTML_DIR = RAW_DIR / "html"
OUTPUT_DIR = BASE_DIR / "output"
CHUNKS_JSON = OUTPUT_DIR / "chunks.json"

# Fragmentación del corpus
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 100

# Embeddings Gemini
EMBEDDINGS_JSON = OUTPUT_DIR / "embeddings.json"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
EMBED_BATCH_SIZE = 50
MAX_CHUNKS_EMBED = None
EMBED_BATCH_PAUSE_SECONDS = 32
EMBED_MAX_RETRIES = 5
EMBED_RETRY_SECONDS = 60

# Índice vectorial persistente
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "residuos_madrid"
CHROMA_DISTANCE = "cosine"
INDEX_BATCH_SIZE = 100

# Consulta y contexto del RAG
TOP_K = 5
MAX_TOP_K_RETRIEVAL = 10
MAX_PREGUNTA_CARACTERES = 500
MAX_CONTEXTO_CARACTERES = 12_000

# Parámetros de experimentos de reranking: NO usados en el flujo actual
RETRIEVAL_MIN_CANDIDATES = 50
RETRIEVAL_SEMANTIC_WEIGHT = 0.70
RETRIEVAL_LEXICAL_WEIGHT = 0.30

# Evaluación
QUERIES_DIR = BASE_DIR / "queries"
EVAL_QUERIES_JSON = QUERIES_DIR / "eval_queries.json"
CACHE_DIR = OUTPUT_DIR / "cache"
EVAL_QUERY_EMBEDDINGS_JSON = CACHE_DIR / "eval_query_embeddings.json"
EVALUATION_RESULTS_DIR = BASE_DIR / "tests" / "evaluation"

# Generación Gemini
GENERATION_MODEL = "gemini-3.5-flash-lite"
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_SECONDS = 2