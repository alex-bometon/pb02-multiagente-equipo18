# Orquesta la preparación del corpus

from src.load import cargar_corpus
from src.clean import limpiar_corpus
from src.chunk import crear_chunks
from src.artifacts import guardar_chunks_json


def ejecutar_ingesta() -> list[dict]:
    """
    Ejecuta el pipeline previo a embeddings:

    archivos originales
        -> carga
        -> limpieza
        -> chunking
        -> chunks.json
    """

    corpus = cargar_corpus()

    documentos = limpiar_corpus(corpus)

    chunks = crear_chunks(documentos)

    ruta = guardar_chunks_json(chunks)

    print(f"Archivos originales: {len(corpus)}")

    print(f"Documentos limpios: {len(documentos)}")

    print(f"Chunks generados: {len(chunks)}")

    print(f"Chunks guardados en: {ruta}")

    return chunks


if __name__ == "__main__":
    ejecutar_ingesta()