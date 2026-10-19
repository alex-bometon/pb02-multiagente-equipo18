"""Recuperación semántica del corpus de ReciclaTIA mediante ChromaDB.

Aplica filtros conservadores cuando la consulta menciona explícitamente
una modalidad de punto limpio, un distrito o el catálogo de residuos.
También permite consultar estructuradamente el catálogo y las paradas.
No genera respuestas, no aplica reranking y no modifica el índice.
"""

import re
import unicodedata
from math import isfinite
from numbers import Real

from google.genai import types

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TOP_K,
)
from src.gemini_client import crear_cliente_gemini
from src.index import crear_cliente_chroma


MAX_PREGUNTA_CARACTERES = 500
MAX_TOP_K = 10
FUENTE_CATALOGO = "tipos_residuos.csv"

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


def _normalizar_texto(texto: str) -> str:
    """Normaliza tildes, mayúsculas y espacios solo para detección."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return " ".join(texto.split())


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


def _es_consulta_catalogo(pregunta: str) -> bool:
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


def _normalizar_distrito(nombre: str) -> str:
    """Unifica tildes, mayúsculas y separadores de los distritos."""
    normalizado = _normalizar_texto(nombre)
    return re.sub(r"[\s-]+", " ", normalizado).strip()


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
    if not condiciones and _es_consulta_catalogo(normalizada):
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


def listar_destinos_residuos() -> list[str]:
    """Lista los lugares de depósito registrados en tipos_residuos.csv.

    Los valores no constituyen una taxonomía exhaustiva de residuos.
    Esta función consulta exclusivamente los metadatos de ChromaDB,
    sin solicitar embeddings a Gemini ni generar respuestas.
    """
    coleccion, _ = _abrir_coleccion()

    try:
        resultado = coleccion.get(
            where={"source": FUENTE_CATALOGO},
            include=["metadatas"],
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo consultar el catálogo de residuos."
        ) from exc

    destinos = {
        str(metadata["destino"]).strip()
        for metadata in (resultado.get("metadatas") or [])
        if metadata and metadata.get("destino")
        and str(metadata["destino"]).strip()
    }
    return sorted(destinos, key=str.casefold)


# ---------------------------------------------------------
# CONSULTAS ESTRUCTURADAS DEL CORPUS
# ---------------------------------------------------------


FUENTE_MOVILES = "puntos_limpios_moviles.csv"
TIPO_MOVIL = MODALIDADES["movil"]
EJEMPLOS_POR_DESTINO = 2


def _obtener_registros(coleccion, filtro: dict) -> list[tuple]:
    """Consulta documentos y metadatos con sus posiciones alineadas."""
    try:
        resultado = coleccion.get(
            where=filtro,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Falló la consulta estructurada a ChromaDB."
        ) from exc

    if not isinstance(resultado, dict):
        raise RuntimeError("ChromaDB devolvió una respuesta incompleta.")

    documentos = resultado.get("documents")
    metadatos = resultado.get("metadatas")
    if not isinstance(documentos, list) or not isinstance(
        metadatos, list
    ):
        raise RuntimeError("ChromaDB devolvió una respuesta incompleta.")
    if len(documentos) != len(metadatos):
        raise RuntimeError("ChromaDB devolvió registros inconsistentes.")

    return list(zip(documentos, metadatos))


def _extraer_campo(texto: str, campo: str) -> str | None:
    """Obtiene el valor de un campo situado al comienzo de una línea."""
    patron = rf"(?m)^\s*{re.escape(campo)}:\s*(.*?)\s*$"
    coincidencia = re.search(patron, texto)
    if not coincidencia:
        return None
    return coincidencia.group(1).strip() or None


def _orden_fila(metadata: dict) -> tuple:
    """Permite ordenar por número de fila cuando esté disponible."""
    fila = metadata.get("source_row")
    if isinstance(fila, int) and not isinstance(fila, bool) and fila > 0:
        return (0, fila)
    return (1, 0)


def consultar_catalogo_residuos() -> dict:
    """Agrupa las relaciones residuo-destino del catálogo de Madrid.

    Devuelve todos los recuentos, y hasta dos ejemplos literales por
    destino. Los destinos no son una taxonomía oficial exhaustiva.
    Los ejemplos se recortan en cantidad, nunca en su texto original.
    No solicita embeddings y no altera el índice.
    """
    coleccion, _ = _abrir_coleccion()
    registros = _obtener_registros(
        coleccion, {"source": FUENTE_CATALOGO}
    )
    agrupados = {}
    filas_vistas = set()
    descartados = 0
    duplicados = 0
    rutas = set()

    for texto, metadata in sorted(
        registros,
        key=lambda item: _orden_fila(
            item[1] if isinstance(item[1], dict) else {}
        ),
    ):
        if not isinstance(texto, str) or not isinstance(metadata, dict):
            descartados += 1
            continue
        origen = metadata.get("source")
        ruta = metadata.get("path")
        destino = metadata.get("destino")
        residuo = _extraer_campo(texto, "Residuo")
        destino_texto = _extraer_campo(texto, "Lugar de depósito")
        if not all(
            isinstance(valor, str) and valor.strip()
            for valor in (
                origen, ruta, destino, residuo, destino_texto
            )
        ) or origen != FUENTE_CATALOGO:
            descartados += 1
            continue
        if _normalizar_texto(destino) != _normalizar_texto(
            destino_texto
        ):
            descartados += 1
            continue

        fila = metadata.get("source_row")
        fila_valida = (
            isinstance(fila, int)
            and not isinstance(fila, bool)
            and fila > 0
        )
        # source_row se reinicia para cada grupo de destino en el CSV.
        # Una fila solo es duplicada dentro del mismo destino.
        clave = destino.strip()
        identificador_fila = (_normalizar_texto(clave), fila)
        if fila_valida and identificador_fila in filas_vistas:
            duplicados += 1
            continue
        if fila_valida:
            filas_vistas.add(identificador_fila)

        rutas.add(ruta.strip())
        grupo = agrupados.setdefault(
            clave, {"destino": clave, "cantidad": 0, "ejemplos": []}
        )
        grupo["cantidad"] += 1
        if len(grupo["ejemplos"]) < EJEMPLOS_POR_DESTINO:
            ejemplo = {"residuo": residuo}
            if fila_valida:
                ejemplo["source_row"] = fila
            grupo["ejemplos"].append(ejemplo)

    if len(rutas) > 1:
        raise RuntimeError(
            "El catálogo tiene rutas de procedencia inconsistentes."
        )

    destinos = sorted(
        agrupados.values(), key=lambda x: x["destino"].casefold()
    )
    return {
        "source": FUENTE_CATALOGO,
        "path": next(iter(rutas), None),
        "total_relaciones": sum(item["cantidad"] for item in destinos),
        "total_destinos": len(destinos),
        "destinos": destinos,
        "registros_descartados": descartados,
        "registros_duplicados": duplicados,
        "ejemplos_por_destino": EJEMPLOS_POR_DESTINO,
    }


def _validar_criterio(valor: str | None, nombre: str) -> str | None:
    """Valida un criterio opcional de consulta estructurada."""
    if valor is None:
        return None
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{nombre} debe ser un texto no vacío.")
    if len(valor.strip()) > MAX_PREGUNTA_CARACTERES:
        raise ValueError(
            f"{nombre} no puede superar "
            f"{MAX_PREGUNTA_CARACTERES} caracteres."
        )
    return valor.strip()


def _normalizar_direccion(direccion: str) -> str:
    """Normaliza signos y tildes para comparar calles sin inventarlas."""
    texto = _normalizar_texto(direccion)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", texto).split())


def _coincide_direccion(direccion: str, busqueda: str) -> bool:
    """Exige que la búsqueda aparezca con límites de palabras."""
    normalizada = _normalizar_direccion(direccion)
    buscada = _normalizar_direccion(busqueda)
    if not buscada:
        return False
    return bool(re.search(
        rf"(?<!\w){re.escape(buscada)}(?!\w)", normalizada
    ))


def consultar_paradas_moviles(
    *,
    distrito: str | None = None,
    direccion: str | None = None,
    dia: str | None = None,
) -> list[dict]:
    """Consulta paradas móviles ordinarias con su horario particular.

    Exige distrito o dirección; un día aislado no identifica una parada.
    Conserva todas las coincidencias sin seleccionar arbitrariamente una.
    Los móviles 24 horas no están incluidos.
    """
    distrito = _validar_criterio(distrito, "distrito")
    direccion = _validar_criterio(direccion, "direccion")
    dia = _validar_criterio(dia, "dia")
    if distrito is None and direccion is None:
        raise ValueError(
            "Es necesario indicar un distrito o una dirección."
        )

    coleccion, _ = _abrir_coleccion()
    filtro: dict = {"document_type": TIPO_MOVIL}
    if distrito is not None:
        nombre = _normalizar_distrito(distrito)
        variantes = sorted(
            valor for valor in _distritos_del_indice(coleccion)
            if _normalizar_distrito(valor) == nombre
        )
        if not variantes:
            return []
        distrito_filtro = (
            variantes[0] if len(variantes) == 1
            else {"$in": variantes}
        )
        filtro = {
            "$and": [
                {"document_type": TIPO_MOVIL},
                {"district": distrito_filtro},
            ]
        }

    registros = _obtener_registros(coleccion, filtro)
    resultados = []
    filas_vistas = set()
    for texto, metadata in registros:
        if not isinstance(texto, str) or not isinstance(metadata, dict):
            continue
        if (
            metadata.get("document_type") != TIPO_MOVIL
            or metadata.get("source") != FUENTE_MOVILES
            or not isinstance(metadata.get("path"), str)
            or not metadata["path"].strip()
        ):
            continue

        calle = _extraer_campo(texto, "Dirección")
        horario = _extraer_campo(texto, "Horario")
        turno = _extraer_campo(texto, "Turno")
        dia_texto = _extraer_campo(texto, "Día")
        distrito_texto = _extraer_campo(texto, "Distrito")
        if not all((calle, horario, dia_texto, distrito_texto)):
            continue
        dia_texto = dia_texto.rstrip(".").strip()
        distrito_texto = distrito_texto.rstrip(".").strip()

        distrito_meta = metadata.get("district")
        dia_meta = metadata.get("day")
        if not isinstance(distrito_meta, str) or not isinstance(
            dia_meta, str
        ):
            continue
        if (
            _normalizar_distrito(distrito_meta)
            != _normalizar_distrito(distrito_texto)
            or _normalizar_texto(dia_meta)
            != _normalizar_texto(dia_texto)
        ):
            continue
        if distrito is not None and (
            _normalizar_distrito(distrito_meta)
            != _normalizar_distrito(distrito)
        ):
            continue
        if dia is not None and (
            _normalizar_texto(dia_meta) != _normalizar_texto(dia)
        ):
            continue
        if direccion is not None and not _coincide_direccion(
            calle, direccion
        ):
            continue

        fila = metadata.get("source_row")
        fila_valida = (
            isinstance(fila, int)
            and not isinstance(fila, bool)
            and fila > 0
        )
        if fila_valida and fila in filas_vistas:
            continue
        if fila_valida:
            filas_vistas.add(fila)

        registro = {
            "direccion": calle,
            "distrito": distrito_texto,
            "dia": dia_texto,
            "turno": turno,
            "horario": horario,
            "source": FUENTE_MOVILES,
            "path": metadata["path"].strip(),
            "text": texto,
            "metadata": dict(metadata),
        }
        if fila_valida:
            registro["source_row"] = fila
        resultados.append(registro)

    return sorted(
        resultados,
        key=lambda r: (
            _normalizar_distrito(r["distrito"]),
            _normalizar_direccion(r["direccion"]),
            _normalizar_texto(r["dia"]),
            r.get("source_row", 0),
        ),
    )
