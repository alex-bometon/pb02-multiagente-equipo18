"""Consultas estructuradas sobre el corpus persistido de Madrid.

No ejecuta embeddings, LLM ni la búsqueda vectorial.
La conexión a Chroma se inyecta para conservar el contrato de retrieval
sin ocultar dependencias ni acoplar esta capa a la orquestación.
"""

import re
from typing import Callable

from config import MAX_PREGUNTA_CARACTERES
from src.text_utils import normalizar_distrito, normalizar_texto


FUENTE_CATALOGO = "tipos_residuos.csv"
FUENTE_MOVILES = "puntos_limpios_moviles.csv"
TIPO_MOVIL = "punto_limpio_movil"
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


def consultar_catalogo_residuos(abrir_coleccion: Callable) -> dict:
    """Agrupa las relaciones residuo-destino del catálogo de Madrid.

    Devuelve todos los recuentos, y hasta dos ejemplos literales por
    destino. Los destinos no son una taxonomía oficial exhaustiva.
    Los ejemplos se recortan en cantidad, nunca en su texto original.
    No solicita embeddings y no altera el índice.
    """
    coleccion, _ = abrir_coleccion()
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
        if normalizar_texto(destino) != normalizar_texto(
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
        identificador_fila = (normalizar_texto(clave), fila)
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
    texto = normalizar_texto(direccion)
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
    abrir_coleccion: Callable,
    distritos_del_indice: Callable,
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

    coleccion, _ = abrir_coleccion()
    filtro: dict = {"document_type": TIPO_MOVIL}
    if distrito is not None:
        nombre = normalizar_distrito(distrito)
        variantes = sorted(
            valor for valor in distritos_del_indice(coleccion)
            if normalizar_distrito(valor) == nombre
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
            normalizar_distrito(distrito_meta)
            != normalizar_distrito(distrito_texto)
            or normalizar_texto(dia_meta)
            != normalizar_texto(dia_texto)
        ):
            continue
        if distrito is not None and (
            normalizar_distrito(distrito_meta)
            != normalizar_distrito(distrito)
        ):
            continue
        if dia is not None and (
            normalizar_texto(dia_meta) != normalizar_texto(dia)
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
            normalizar_distrito(r["distrito"]),
            _normalizar_direccion(r["direccion"]),
            normalizar_texto(r["dia"]),
            r.get("source_row", 0),
        ),
    )


def listar_destinos_residuos(abrir_coleccion: Callable) -> list[str]:
    """Lista los lugares de depósito registrados en tipos_residuos.csv.

    Los valores no constituyen una taxonomía exhaustiva de residuos.
    Esta función consulta exclusivamente los metadatos de ChromaDB,
    sin solicitar embeddings a Gemini ni generar respuestas.
    """
    coleccion, _ = abrir_coleccion()

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
