from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# =========================================================
# CREACIÓN DEL SPLITTER
# =========================================================

def crear_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Crea el divisor de texto utilizado para los documentos
    textuales largos del corpus.

    Se utiliza RecursiveCharacterTextSplitter porque intenta
    mantener juntas las unidades semánticas antes de recurrir
    a cortes más pequeños.

    Orden de prioridad:
        1. saltos de línea
        2. frases
        3. palabras
        4. caracteres

    CHUNK_SIZE y CHUNK_OVERLAP se encuentran en config.py
    para poder modificarlos fácilmente durante la evaluación.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


# =========================================================
# CHUNK DE DOCUMENTO ESTRUCTURADO
# =========================================================

def _chunk_documento_estructurado(documento: dict) -> list[dict]:
    """
    Convierte un documento estructurado procedente de CSV
    en un único chunk.

    Los registros CSV ya han sido transformados en clean.py
    en unidades semánticas completas.

    Ejemplo:

        Contenedor de ropa y residuos textiles.
        Dirección: ...
        Distrito: ...
        Barrio: ...

    Dividir este contenido no aportaría ninguna ventaja y
    podría separar información que pertenece a una misma
    entidad.
    """

    texto = documento["text"]
    metadata = documento["metadata"].copy()

    metadata.update(
        {
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_size": len(texto),
        }
    )

    return [
        {
            "text": texto,
            "metadata": metadata,
        }
    ]


# =========================================================
# CHUNK DE DOCUMENTACIÓN HTML
# =========================================================

def _chunk_documentacion(
    documento: dict,
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    Divide un documento textual largo en fragmentos.

    Esta función se utiliza principalmente con las páginas
    HTML del Ayuntamiento que clean.py ya ha convertido
    previamente en texto limpio.
    """

    texto = documento["text"]

    partes = splitter.split_text(texto)

    chunks = []

    total_chunks = len(partes)

    for indice, parte in enumerate(partes):

        parte = parte.strip()

        if not parte:
            continue

        metadata = documento["metadata"].copy()

        metadata.update(
            {
                "chunk_index": indice,
                "chunk_count": total_chunks,
                "chunk_size": len(parte),
            }
        )

        chunks.append(
            {
                "text": parte,
                "metadata": metadata,
            }
        )

    return chunks


# =========================================================
# CHUNK DE UN DOCUMENTO
# =========================================================

def crear_chunks_documento(
    documento: dict,
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    Decide cómo procesar un documento según su origen.

    - Documentación municipal procedente de HTML:
        se divide mediante RecursiveCharacterTextSplitter.

    - Registros procedentes de CSV:
        se mantienen completos como un único chunk.
    """

    tipo = documento["metadata"].get("document_type")

    if tipo == "documentacion_municipal":
        return _chunk_documentacion(documento, splitter)

    return _chunk_documento_estructurado(documento)


# =========================================================
# CHUNKING DEL CORPUS COMPLETO
# =========================================================

def crear_chunks(documentos: list[dict]) -> list[dict]:
    """
    Genera los chunks de todo el corpus limpio.

    Parameters
    ----------
    documentos : list[dict]
        Salida de limpiar_corpus().

    Returns
    -------
    list[dict]
        Lista de chunks preparados para la fase
        de embeddings.
    """

    splitter = crear_text_splitter()

    chunks = []

    for documento in documentos:

        chunks_documento = crear_chunks_documento(documento, splitter)

        chunks.extend(chunks_documento)

    return chunks