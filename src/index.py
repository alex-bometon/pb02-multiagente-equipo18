# Se encarga de almacenar los embeddings en ChromaDB
# y construir el índice vectorial persistente.

import hashlib

import chromadb

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHROMA_DISTANCE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INDEX_BATCH_SIZE,
)

from src.artifacts import cargar_embeddings_json


# =========================================================
# CLIENTE DE CHROMADB
# =========================================================

def crear_cliente_chroma():
    """
    Crea un cliente persistente de ChromaDB.

    La base de datos se almacena dentro de output/chroma_db,
    por lo que podrá reutilizarse posteriormente desde
    retrieve.py sin volver a ejecutar la indexación.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


# =========================================================
# IDENTIFICADORES
# =========================================================

def generar_id_item(item: dict) -> str:
    """
    Genera un identificador determinista para un embedding.

    El ID se construye a partir de información estable del
    documento:

        path
        source_row
        chunk_index
        text

    Después se transforma mediante SHA-256.

    Esto permite que el mismo chunk produzca siempre
    el mismo identificador.
    """

    metadata = item["metadata"]

    identidad = "|".join(
        [
            str(metadata.get("path", "")),
            str(metadata.get("source_row", "")),
            str(metadata.get("chunk_index", "")),
            item["text"],
        ]
    )

    hash_id = hashlib.sha256(identidad.encode("utf-8")).hexdigest()

    return f"chunk_{hash_id}"


# =========================================================
# CREACIÓN DE LA COLECCIÓN
# =========================================================

def crear_coleccion(client):
    """
    Crea desde cero la colección utilizada por el RAG.

    Si ya existe una colección con el mismo nombre,
    se elimina antes de crearla de nuevo.

    De este modo, cada ejecución de index.py representa
    exactamente el contenido actual de embeddings.json
    y no conserva registros de versiones anteriores.
    """

    colecciones = client.list_collections()

    nombres = [coleccion.name for coleccion in colecciones]

    if CHROMA_COLLECTION_NAME in nombres:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)

    return client.create_collection(
        name=CHROMA_COLLECTION_NAME,

        # Los embeddings ya han sido generados por Gemini.
        # Chroma no debe generar otros embeddings.
        embedding_function=None,
        configuration={"hnsw": {"space": CHROMA_DISTANCE}},
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "distance_metric": CHROMA_DISTANCE,
        }
    )


# =========================================================
# INDEXACIÓN
# =========================================================

def indexar_embeddings(
    collection,
    items: list[dict],
    batch_size: int,
) -> None:
    """
    Inserta los elementos de embeddings.json en ChromaDB.

    Cada registro almacena:

        id
        documento
        embedding
        metadata

    Los elementos se insertan por lotes para que el código
    pueda seguir funcionando aunque el corpus crezca.
    """

    for inicio in range(0, len(items), batch_size):

        fin = inicio + batch_size
        lote = items[inicio:fin]

        ids = [generar_id_item(item) for item in lote]
        documentos = [item["text"] for item in lote]
        embeddings = [item["vector"] for item in lote]
        metadatas = [item["metadata"] for item in lote]

        collection.add(
            ids=ids,
            documents=documentos,
            embeddings=embeddings,
            metadatas=metadatas,
        )


# =========================================================
# EJECUCIÓN DE LA FASE DE INDEXACIÓN
# =========================================================

def ejecutar_indexacion():
    """
    Ejecuta la fase completa de indexación.

    Flujo:

        embeddings.json
            -> carga
            -> creación de ChromaDB
            -> creación de colección
            -> generación de IDs
            -> indexación por lotes
            -> validación del número de registros
    """

    # -----------------------------------------------------
    # CARGAR EMBEDDINGS
    # -----------------------------------------------------

    items = cargar_embeddings_json()

    if not items:
        raise ValueError("embeddings.json no contiene elementos.")

    print(f"Embeddings disponibles: {len(items)}")


    # -----------------------------------------------------
    # COMPROBAR IDS
    # -----------------------------------------------------

    ids = [generar_id_item(item) for item in items]

    if len(ids) != len(set(ids)):
        raise ValueError("Se han generado IDs duplicados.")


    # -----------------------------------------------------
    # CREAR CLIENTE
    # -----------------------------------------------------

    client = crear_cliente_chroma()


    # -----------------------------------------------------
    # CREAR COLECCIÓN
    # -----------------------------------------------------

    collection = crear_coleccion(client)


    # -----------------------------------------------------
    # DETERMINAR TAMAÑO DE LOTE
    # -----------------------------------------------------

    max_batch_size = (client.get_max_batch_size())

    batch_size = min(INDEX_BATCH_SIZE, max_batch_size)

    print(f"Tamaño de lote: {batch_size}")


    # -----------------------------------------------------
    # INDEXAR
    # -----------------------------------------------------

    indexar_embeddings(collection, items, batch_size)


    # -----------------------------------------------------
    # VALIDAR RESULTADO
    # -----------------------------------------------------

    total_indexados = collection.count()

    if total_indexados != len(items):
        raise ValueError(
            "El número de registros indexados no coincide "
            "con el número de embeddings de origen. "
            f"Embeddings: {len(items)} | "
            f"ChromaDB: {total_indexados}"
        )


    # -----------------------------------------------------
    # RESULTADO
    # -----------------------------------------------------

    print()
    print("Indexación completada correctamente.")
    print(f"Colección: {CHROMA_COLLECTION_NAME}")
    print(f"Registros indexados: {total_indexados}")
    print(f"Dimensiones: {EMBEDDING_DIMENSIONS}")
    print(f"Métrica: {CHROMA_DISTANCE}")
    print(f"Base de datos: {CHROMA_DIR}")

    return collection


# =========================================================
# EJECUCIÓN DIRECTA
# =========================================================

if __name__ == "__main__":
    ejecutar_indexacion()