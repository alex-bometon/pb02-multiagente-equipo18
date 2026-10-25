# Carga y guarda artefactos intermediarios del pipeline
# Se encarga de la persistencia
# chunks.json
# embeddings.json

import json
from pathlib import Path

from config import (
    CHUNKS_JSON,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_JSON
)


# =========================================================
# CHUNKS
# =========================================================

def guardar_chunks_json(
    chunks: list[dict],
    ruta: Path = CHUNKS_JSON,
) -> Path:
    """
    Guarda los chunks generados por el pipeline en formato JSON.

    El archivo conserva:
    - configuración de chunking;
    - fuentes utilizadas;
    - número total de chunks;
    - texto y metadatos de cada chunk.

    Devuelve la ruta del archivo generado.
    """

    fuentes = sorted({chunk["metadata"]["source"] for chunk in chunks})

    payload = {
        "schema_version": 1,
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "source_count": len(fuentes),
        "sources": fuentes,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, ensure_ascii=False, indent=2)

    return ruta


def cargar_chunks_json(
    ruta: Path = CHUNKS_JSON,
) -> list[dict]:
    """
    Carga los chunks previamente guardados.

    Esta función permitirá que la fase de embeddings
    trabaje directamente desde chunks.json sin tener
    que repetir load -> clean -> chunk.
    """

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {ruta}")

    with ruta.open("r", encoding="utf-8") as archivo:
        payload = json.load(archivo)

    if "chunks" not in payload:
        raise ValueError("El archivo no contiene la clave 'chunks'.")

    return payload["chunks"]


# =========================================================
# EMBEDDINGS
# =========================================================

def guardar_embeddings_json(
    items: list[dict],
    *,
    modelo: str,
    dimensiones: int,
    total_chunks_origen: int,
    ruta: Path = EMBEDDINGS_JSON,
) -> Path:
    """
    Guarda los embeddings generados junto con el texto
    y los metadatos correspondientes.

    Cada elemento mantiene la relación:

        texto <-> metadata <-> vector

    También se guarda información general necesaria
    para reproducir la generación de embeddings.
    """

    payload = {
        "schema_version": 1,
        "embedding_model": modelo,
        "embedding_dimensions": dimensiones,
        "embedding_input_format": "title: {title_or_none} | text: {text}",
        "total_chunks_origen": total_chunks_origen,
        "total_embeddings": len(items),
        "items": items,
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, ensure_ascii=False, indent=2)

    return ruta

def cargar_embeddings_json(ruta: Path = EMBEDDINGS_JSON) -> list[dict]:
    """
    Carga los embeddings previamente generados.

    Devuelve únicamente la lista de elementos que mantienen
    la relación:

        texto <-> metadata <-> vector

    Esto permite que index.py trabaje directamente desde
    embeddings.json sin volver a generar los embeddings.
    """

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de embeddings: {ruta}")

    with ruta.open("r", encoding="utf-8") as archivo:
        payload = json.load(archivo)

    if "items" not in payload:
        raise ValueError("El archivo no contiene la clave 'items'.")

    items = payload["items"]

    if not isinstance(items, list):
        raise ValueError("La clave 'items' debe contener una lista.")

    return items