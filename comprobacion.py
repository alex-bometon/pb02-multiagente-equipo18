
import hashlib
import json

from config import OUTPUT_DIR
from src.retrieve import buscar_chunks_relevantes

pregunta = "¿En qué contenedor tiro un envase de plástico?"

ruta = OUTPUT_DIR / "cache" / "pb02_embeddings_consultas.json"
cache = json.loads(ruta.read_text(encoding="utf-8"))

clave = hashlib.sha256(pregunta.encode("utf-8")).hexdigest()
vector = cache["embeddings"][clave]

chunks = buscar_chunks_relevantes(
    pregunta,
    top_k=5,
    query_embedding=vector,
)

for indice, chunk in enumerate(chunks, start=1):
    print(f"\n{'=' * 60}")
    print(f"CHUNK {indice}")
    print("Fuente:", chunk["metadata"].get("source"))
    print("Distancia:", chunk["distance"])
    print("Texto completo:")
    print(chunk["text"])
