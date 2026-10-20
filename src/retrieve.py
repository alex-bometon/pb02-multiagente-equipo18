"""Recuperación semántica del corpus de ReciclaTIA mediante ChromaDB.

Aplica filtros conservadores cuando la consulta menciona explícitamente
una modalidad de punto limpio, un distrito o el catálogo de residuos.
Delega las consultas estructuradas a un módulo especializado.
No genera respuestas, no aplica reranking y no modifica el índice.
"""

import re
from math import isfinite
from numbers import Real

from google.genai import types

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_PREGUNTA_CARACTERES,
    MAX_TOP_K_RETRIEVAL as MAX_TOP_K,
    TOP_K,
)
from src.gemini_client import crear_cliente_gemini
from src.index import crear_cliente_chroma
from src.input_validation import validar_pregunta as _validar_pregunta
from src import retrieve_structured as _structured
from src.text_utils import (
    normalizar_distrito as _normalizar_distrito,
    normalizar_texto as _normalizar_texto,
)


FUENTE_CATALOGO = _structured.FUENTE_CATALOGO

MODALIDADES = {
    "fijo": "punto_limpio_fijo",
    "movil": "punto_limpio_movil",
    "movil_24h": "punto_limpio_movil_24h",
    "proximidad": "punto_limpio_proximidad",
}


# ---------------------------------------------------------
# VALIDACIÓN DE ENTRADAS Y EMBEDDINGS
# ---------------------------------------------------------


def _validar_pregunta(pregunta: str) -> str:
    """Comprueba que la pregunta es válida y elimina espacios externos."""
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta debe ser un texto no vacío.")

    pregunta = pregunta.strip()
    if len(pregunta) > MAX_PREGUNTA_CARACTERES:
        raise ValueError(
            "La pregunta no puede superar "
            f"{MAX_PREGUNTA_CARACTERES} caracteres."
        )
    return pregunta


def _validar_embedding(vector: list[float]) -> list[float]:
    """Comprueba la dimensión y los valores numéricos del embedding."""
    if not isinstance(vector, list):
        raise ValueError("El embedding debe ser una lista de números.")

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"El embedding debe tener {EMBEDDING_DIMENSIONS} dimensiones; "
            f"se recibieron {len(vector)}."
        )

    if any(
        isinstance(valor, bool)
        or not isinstance(valor, Real)
        or not isfinite(float(valor))
        for valor in vector
    ):
        raise ValueError("El embedding debe contener números finitos.")

    return [float(valor) for valor in vector]


def _generar_embedding_pregunta(pregunta: str) -> list[float]:
    """Genera un embedding con Gemini usando el modelo del índice."""
    cliente = crear_cliente_gemini()

    try:
        try:
            respuesta = cliente.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=pregunta,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "No se pudo generar el embedding con Gemini."
            ) from exc
    finally:
        cliente.close()

    if not respuesta.embeddings or respuesta.embeddings[0].values is None:
        raise RuntimeError("Gemini no devolvió un embedding válido.")

    return _validar_embedding(list(respuesta.embeddings[0].values))


# ---------------------------------------------------------
# ACCESO A CHROMADB
# ---------------------------------------------------------


def _abrir_coleccion():
    """Abre la colección persistente sin crear un índice nuevo."""
    if not CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encuentra la base vectorial en {CHROMA_DIR}. "
            "Ejecuta primero: python -m src.index"
        )

    try:
        coleccion = crear_cliente_chroma().get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=None,
        )
        total = coleccion.count()
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo abrir la colección '{CHROMA_COLLECTION_NAME}'."
        ) from exc

    if total == 0:
        raise ValueError("La colección de ChromaDB está vacía.")

    return coleccion, total


# ---------------------------------------------------------
# IDENTIFICACIÓN CONSERVADORA DE INTENCIÓN
# ---------------------------------------------------------


def _detectar_modalidad(pregunta: str) -> str | None:
    """Identifica una modalidad inequívoca de punto limpio."""
    if not re.search(r"\bpuntos?\s+limpios?\b", pregunta):
        return None

    detectadas = set()
    if re.search(r"\bmovil(?:es)?\b", pregunta):
        if re.search(r"\b24\s*(?:horas?|h)\b", pregunta):
            detectadas.add("movil_24h")
        else:
            detectadas.add("movil")

    if re.search(r"\bfijos?\b", pregunta):
        detectadas.add("fijo")

    if re.search(r"\bproximidad\b", pregunta):
        detectadas.add("proximidad")

    if len(detectadas) != 1:
        return None

    return MODALIDADES[detectadas.pop()]


def _es_busqueda_catalogo_semantico(pregunta: str) -> bool:
    """Reconoce consultas generales sobre el catálogo de residuos."""
    if "punto limpio" in pregunta or "contenedor" in pregunta:
        return False

    return bool(re.search(
        r"\b(?:tipos|categorias|clases|catalogo|listado|relacion)"
        r"\s+de\s+residuos\b",
        pregunta,
    ))


def _distritos_del_indice(coleccion) -> set[str]:
    """Obtiene del propio índice los distritos presentes en metadatos."""
    try:
        resultado = coleccion.get(include=["metadatas"])
        metadatos = resultado.get("metadatas") or []
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron consultar los distritos del índice."
        ) from exc

    return {
        str(metadata["district"]).strip()
        for metadata in metadatos
        if metadata and metadata.get("district")
    }


def _detectar_distritos(
    pregunta: str,
    coleccion,
    modalidad: str | None,
) -> list[str]:
    """Detecta un distrito y conserva todas sus variantes en el índice."""
    if modalidad is None and "distrito" not in pregunta:
        return []

    pregunta_normalizada = _normalizar_distrito(pregunta)
    encontrados = []

    for distrito in _distritos_del_indice(coleccion):
        nombre = re.escape(_normalizar_distrito(distrito))
        patrones = [
            rf"\bdistrito\s+(?:de\s+)?{nombre}(?!\w)",
        ]
        if modalidad is not None:
            patrones.append(
                rf"\b(?:de|del|en|por)\s+(?:el\s+)?{nombre}(?!\w)"
            )

        coincide = any(
            re.search(patron, pregunta_normalizada)
            for patron in patrones
        )
        if "distrito" in pregunta_normalizada:
            coincide = coincide or bool(
                re.search(rf"(?<!\w){nombre}(?!\w)", pregunta_normalizada)
            )

        if coincide:
            encontrados.append(distrito)

    return sorted(encontrados)


def _construir_filtro(pregunta: str, coleccion) -> dict | None:
    """Construye un filtro sin restringir consultas ambiguas."""
    normalizada = _normalizar_texto(pregunta)
    modalidad = _detectar_modalidad(normalizada)

    # Varias modalidades explícitas implican una petición ambigua.
    menciona_fijo = bool(re.search(r"\bfijos?\b", normalizada))
    menciona_movil = bool(re.search(r"\bmovil(?:es)?\b", normalizada))
    menciona_proximidad = "proximidad" in normalizada
    if (
        "punto limpio" in normalizada
        and sum((menciona_fijo, menciona_movil, menciona_proximidad)) > 1
    ):
        return None

    distritos = _detectar_distritos(normalizada, coleccion, modalidad)
    distritos_distintos = {
        _normalizar_distrito(distrito) for distrito in distritos
    }
    if len(distritos_distintos) > 1:
        return None

    condiciones = []
    if modalidad is not None:
        condiciones.append({"document_type": modalidad})
    if distritos:
        valor_distrito = (
            distritos[0]
            if len(distritos) == 1
            else {"$in": distritos}
        )
        condiciones.append({"district": valor_distrito})
    if not condiciones and _es_busqueda_catalogo_semantico(normalizada):
        condiciones.append({"source": FUENTE_CATALOGO})

    if not condiciones:
        return None
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}


def _filtro_tiene_registros(coleccion, filtro: dict) -> bool:
    """Evita generar embeddings cuando un filtro no tiene resultados."""
    try:
        resultado = coleccion.get(
            where=filtro,
            limit=1,
            include=["metadatas"],
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo validar el filtro en ChromaDB."
        ) from exc

    return bool(resultado.get("ids"))


# ---------------------------------------------------------
# API PÚBLICA: RETRIEVAL Y CONSULTA DEL CATÁLOGO
# ---------------------------------------------------------


def buscar_chunks_relevantes(
    pregunta: str,
    top_k: int = TOP_K,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Recupera chunks mediante similitud coseno y filtros explícitos.

    Args:
        pregunta: Consulta del usuario, hasta 500 caracteres.
        top_k: Número de resultados entre 1 y 10.
        query_embedding: Embedding opcional de 768 dimensiones.
            Si se proporciona, no se llama a Gemini.

    Returns:
        Chunks ordenados por distancia ascendente, cada uno con
        las claves ``text``, ``distance`` y ``metadata``. Devuelve
        una lista vacía si un filtro explícito carece de registros.
    """
    pregunta = _validar_pregunta(pregunta)

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k debe ser un número entero.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k debe estar entre 1 y {MAX_TOP_K}.")

    if query_embedding is not None:
        query_embedding = _validar_embedding(query_embedding)

    coleccion, total = _abrir_coleccion()
    filtro = _construir_filtro(pregunta, coleccion)

    if filtro is not None and not _filtro_tiene_registros(
        coleccion, filtro
    ):
        return []

    if query_embedding is None:
        query_embedding = _generar_embedding_pregunta(pregunta)

    parametros = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if filtro is not None:
        parametros["where"] = filtro

    try:
        resultado = coleccion.query(**parametros)
    except Exception as exc:
        raise RuntimeError("Falló la consulta vectorial a ChromaDB.") from exc

    try:
        documentos = resultado["documents"][0]
        metadatos = resultado["metadatas"][0]
        distancias = resultado["distances"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "ChromaDB devolvió una respuesta incompleta."
        ) from exc

    if not (len(documentos) == len(metadatos) == len(distancias)):
        raise RuntimeError("ChromaDB devolvió resultados inconsistentes.")

    chunks = [
        {
            "text": texto,
            "distance": float(distancia),
            "metadata": metadata or {},
        }
        for texto, metadata, distancia in zip(
            documentos, metadatos, distancias
        )
    ]
    return sorted(chunks, key=lambda chunk: chunk["distance"])




# Compatibilidad de la API pública: implementaciones especializadas.
FUENTE_MOVILES = _structured.FUENTE_MOVILES
TIPO_MOVIL = _structured.TIPO_MOVIL
EJEMPLOS_POR_DESTINO = _structured.EJEMPLOS_POR_DESTINO
_coincide_direccion = _structured._coincide_direccion


def listar_destinos_residuos() -> list[str]:
    """Consulta los destinos registrados en el catálogo municipal."""
    return _structured.listar_destinos_residuos(_abrir_coleccion)


def consultar_catalogo_residuos() -> dict:
    """Consulta las relaciones y destinos del catálogo municipal."""
    return _structured.consultar_catalogo_residuos(_abrir_coleccion)


def consultar_paradas_moviles(
    *,
    distrito: str | None = None,
    direccion: str | None = None,
    dia: str | None = None,
) -> list[dict]:
    """Consulta las paradas móviles ordinarias por ubicación y día."""
    return _structured.consultar_paradas_moviles(
        distrito=distrito,
        direccion=direccion,
        dia=dia,
        abrir_coleccion=_abrir_coleccion,
        distritos_del_indice=_distritos_del_indice,
    )
