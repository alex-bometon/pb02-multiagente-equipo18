# Se encarga de producir los vectores, embeddings

import time

from google.genai import types, errors

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBED_BATCH_SIZE,
    EMBED_BATCH_PAUSE_SECONDS,
    EMBED_MAX_RETRIES,
    EMBED_RETRY_SECONDS,    
    MAX_CHUNKS_EMBED,
)

from src.artifacts import cargar_chunks_json, guardar_embeddings_json

from src.gemini_client import crear_cliente_gemini


# =========================================================
# PREPARACIÓN DEL TEXTO
# =========================================================

def preparar_documento_embedding(chunk: dict,) -> str:
    """
    Prepara un chunk para generar su embedding.

    Gemini Embedding 2 recomienda representar los
    documentos destinados a retrieval con la forma:

        title: {title} | text: {content}

    Si el documento no dispone de título, se utiliza
    'none', tal como indica la documentación de Google.
    """

    texto = chunk["text"]

    metadata = chunk.get("metadata", {})

    titulo = metadata.get("title")

    if not titulo:
        titulo = "none"

    return f"title: {titulo} | text: {texto}"


# =========================================================
# GENERACIÓN DE UN EMBEDDING
# =========================================================

def embeddear_documento(client, chunk: dict) -> list[float]:
    """
    Genera el embedding de un único chunk.

    Devuelve solamente el vector numérico.
    """

    contenido = preparar_documento_embedding(chunk)

    resultado = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contenido,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )

    if not resultado.embeddings:
        raise ValueError("Gemini no ha devuelto ningún embedding.")

    vector = resultado.embeddings[0].values

    if vector is None:
        raise ValueError("El embedding recibido no contiene valores.")

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Dimensión inesperada del embedding: {len(vector)}. "
            f"Se esperaban {EMBEDDING_DIMENSIONS}."
        )

    return vector


# =========================================================
# GENERACIÓN DE EMBEDDINGS POR LOTES
# =========================================================

def _embeddear_lote(client, contenidos):
    """
    Genera los embeddings de un lote.

    Si se alcanza temporalmente el límite del Free Tier,
    espera y vuelve a intentarlo.
    """

    for intento in range(EMBED_MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contenidos,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSIONS
                ),
            )

        except errors.ClientError as error:

            # Si no es un error de cuota 429,
            # dejamos que el error continúe normalmente.
            if error.code != 429:
                raise

            # Si hemos agotado todos los reintentos,
            # dejamos que se muestre el error.
            if intento == EMBED_MAX_RETRIES - 1:
                raise

            print()
            print("Límite temporal del Free Tier alcanzado (429).")
            print(f"Esperando {EMBED_RETRY_SECONDS} segundos antes de reintentar...")

            time.sleep(EMBED_RETRY_SECONDS)


def embeddear_documentos(client, chunks: list[dict]) -> list[list[float]]:
    """
    Genera los embeddings de varios chunks.

    Los chunks se procesan en grupos definidos por
    EMBED_BATCH_SIZE para evitar enviar todo el corpus
    en una única llamada.

    Mantiene el mismo orden de entrada:

        chunks[0] -> embeddings[0]
        chunks[1] -> embeddings[1]
        ...

    Devuelve una lista de vectores.
    """

    embeddings = []

    for inicio in range(0, len(chunks), EMBED_BATCH_SIZE):

        fin = inicio + EMBED_BATCH_SIZE

        lote = chunks[inicio:fin]

        contenidos = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=preparar_documento_embedding(chunk)
                    )
                ]
            )
            for chunk in lote
        ]

        resultado = _embeddear_lote(client, contenidos)

        # Comprobamos que Gemini devuelve exactamente
        # un embedding por cada chunk enviado.
        if len(resultado.embeddings) != len(lote):
            raise ValueError(
                "Número de embeddings inesperado. "
                f"Se enviaron {len(lote)} chunks "
                f"y se recibieron {len(resultado.embeddings)} embeddings."
            )

        for embedding in resultado.embeddings:

            vector = embedding.values

            if vector is None:
                raise ValueError(
                    "Uno de los embeddings no contiene valores."
                )

            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Dimensión inesperada del embedding: {len(vector)}. "
                    f"Se esperaban {EMBEDDING_DIMENSIONS}."
                )

            embeddings.append(vector)

        # Cada lote de 50 consume 50 operaciones de la cuota.
        # Introducimos una pausa para mantenernos por debajo
        # del límite de 100 embeddings/minuto del Free Tier.
        if fin < len(chunks):
            time.sleep(EMBED_BATCH_PAUSE_SECONDS)

    return embeddings


# =========================================================
# EJECUCIÓN DE LA FASE DE EMBEDDINGS
# =========================================================

def ejecutar_embeddings() -> list[dict]:
    """
    Ejecuta la fase completa de generación de embeddings.

    Flujo:

        chunks.json
            -> selección de chunks
            -> generación de embeddings
            -> asociación texto/metadata/vector
            -> embeddings.json
    """


    # CARGAR CHUNKS

    chunks = cargar_chunks_json()

    total_chunks_origen = len(chunks)

    if MAX_CHUNKS_EMBED is not None:
        chunks_a_procesar = chunks[:MAX_CHUNKS_EMBED]
    else:
        chunks_a_procesar = chunks

    print(f"Chunks disponibles: {total_chunks_origen}")

    print(f"Chunks que se procesarán: {len(chunks_a_procesar)}")


    # GENERAR EMBEDDINGS

    client = crear_cliente_gemini()

    try:
        vectores = embeddear_documentos(client, chunks_a_procesar)

    finally:
        client.close()


    # VALIDAR RESULTADO

    if len(vectores) != len(chunks_a_procesar):
        raise ValueError(
            "El número total de embeddings no coincide "
            "con el número de chunks procesados."
        )


    # ASOCIAR CADA CHUNK CON SU VECTOR

    items = []

    for chunk, vector in zip(chunks_a_procesar, vectores):

        items.append(
            {
                "text": chunk["text"],
                "vector": vector,
                "metadata": chunk["metadata"],
            }
        )


    # GUARDAR EMBEDDINGS.JSON

    ruta = guardar_embeddings_json(
        items,
        modelo=EMBEDDING_MODEL,
        dimensiones=EMBEDDING_DIMENSIONS,
        total_chunks_origen=total_chunks_origen,
    )

    print()
    print("Embeddings generados correctamente.")
    print(f"Modelo: {EMBEDDING_MODEL}")
    print(f"Dimensiones: {EMBEDDING_DIMENSIONS}")
    print(f"Embeddings generados: {len(items)}")
    print(f"Archivo generado: {ruta}")

    return items


# =========================================================
# EJECUCIÓN DIRECTA
# =========================================================

if __name__ == "__main__":
    ejecutar_embeddings()