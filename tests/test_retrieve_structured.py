"""Pruebas de consultas estructuradas y puntos limpios móviles."""

from unittest.mock import Mock

import pytest

from src import retrieve


# ---------------------------------------------------------


def _registro_catalogo(residuo, destino, fila, *, ruta=None):
    """Construye un registro con la estructura real del CSV indexado."""
    if ruta is None:
        ruta = "data/raw/csv/tipos_residuos.csv"
    texto = (
        f"Residuo: {residuo}\n"
        f"Lugar de depósito: {destino}"
    )
    metadata = {
        "source": "tipos_residuos.csv",
        "path": ruta,
        "destino": destino,
        "source_row": fila,
        "document_type": "residuo_destino",
    }
    return texto, metadata


@pytest.fixture

def coleccion_catalogo(monkeypatch):
    """Simula consultas estructuradas sin embeddings ni API externa."""
    simulada = Mock()
    registros = [
        _registro_catalogo("Restos de comida", "ORGÁNICA", 4),
        _registro_catalogo("Botellas de plástico", "AMARILLO", 2),
        _registro_catalogo("Botes de detergente", "AMARILLO", 3),
        _registro_catalogo("Restos de fruta", "ORGÁNICA", 5),
        _registro_catalogo("Envases metálicos", "AMARILLO", 6),
    ]
    simulada.get.return_value = {
        "documents": [texto for texto, _ in registros],
        "metadatas": [metadata for _, metadata in registros],
    }
    monkeypatch.setattr(
        retrieve,
        "_abrir_coleccion",
        Mock(return_value=(simulada, 712)),
    )
    generar = Mock(side_effect=AssertionError("No debe usarse Gemini"))
    monkeypatch.setattr(retrieve, "_generar_embedding_pregunta", generar)
    return simulada, generar


def test_catalogo_estructurado_agrupa_y_conserva_procedencia(
    coleccion_catalogo,
):
    coleccion, generar = coleccion_catalogo
    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["source"] == "tipos_residuos.csv"
    assert resultado["path"] == "data/raw/csv/tipos_residuos.csv"
    assert resultado["total_relaciones"] == 5
    assert resultado["total_destinos"] == 2
    assert resultado["registros_descartados"] == 0
    assert resultado["registros_duplicados"] == 0
    assert resultado["ejemplos_por_destino"] == 2
    assert [grupo["destino"] for grupo in resultado["destinos"]] == [
        "AMARILLO", "ORGÁNICA"
    ]
    assert [grupo["cantidad"] for grupo in resultado["destinos"]] == [
        3, 2
    ]
    assert resultado["destinos"][0]["ejemplos"] == [
        {"residuo": "Botellas de plástico", "source_row": 2},
        {"residuo": "Botes de detergente", "source_row": 3},
    ]
    coleccion.get.assert_called_once_with(
        where={"source": "tipos_residuos.csv"},
        include=["documents", "metadatas"],
    )
    coleccion.query.assert_not_called()
    generar.assert_not_called()


def test_catalogo_estructurado_vacio_no_inventa_destinos(
    coleccion_catalogo,
):
    coleccion, _ = coleccion_catalogo
    coleccion.get.return_value = {
        "documents": [],
        "metadatas": [],
    }

    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["total_relaciones"] == 0
    assert resultado["total_destinos"] == 0
    assert resultado["destinos"] == []
    assert resultado["path"] is None


def test_catalogo_descarta_filas_duplicadas(coleccion_catalogo):
    coleccion, _ = coleccion_catalogo
    texto, metadata = _registro_catalogo("Pilas", "PUNTO LIMPIO", 8)
    coleccion.get.return_value = {
        "documents": [texto, texto],
        "metadatas": [metadata, dict(metadata)],
    }

    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["total_relaciones"] == 1
    assert resultado["registros_duplicados"] == 1
    assert resultado["registros_descartados"] == 0


@pytest.mark.parametrize(
    "defecto",
    [
        "texto_invalido",
        "metadata_invalida",
        "sin_residuo",
        "sin_ruta",
        "sin_destino",
        "destino_inconsistente",
        "fuente_incorrecta",
    ],
)
def test_catalogo_descarta_registros_incompletos(
    coleccion_catalogo, defecto
):
    coleccion, _ = coleccion_catalogo
    texto, metadata = _registro_catalogo("Pilas", "PUNTO LIMPIO", 2)
    metadata = dict(metadata)
    if defecto == "texto_invalido":
        texto = None
    elif defecto == "metadata_invalida":
        metadata = None
    elif defecto == "sin_residuo":
        texto = "Lugar de depósito: PUNTO LIMPIO"
    elif defecto == "sin_ruta":
        metadata["path"] = ""
    elif defecto == "sin_destino":
        metadata["destino"] = ""
    elif defecto == "destino_inconsistente":
        metadata["destino"] = "CONTENEDOR AMARILLO"
    elif defecto == "fuente_incorrecta":
        metadata["source"] = "otra_fuente.csv"

    coleccion.get.return_value = {
        "documents": [texto],
        "metadatas": [metadata],
    }

    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["total_relaciones"] == 0
    assert resultado["registros_descartados"] == 1


def test_catalogo_admite_registro_sin_numero_fila(coleccion_catalogo):
    coleccion, _ = coleccion_catalogo
    texto, metadata = _registro_catalogo("Pilas", "PUNTO LIMPIO", 2)
    metadata.pop("source_row")
    coleccion.get.return_value = {
        "documents": [texto],
        "metadatas": [metadata],
    }

    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["total_relaciones"] == 1
    assert resultado["destinos"][0]["ejemplos"] == [
        {"residuo": "Pilas"}
    ]


def test_catalogo_rechaza_rutas_de_procedencia_incompatibles(
    coleccion_catalogo,
):
    coleccion, _ = coleccion_catalogo
    primero = _registro_catalogo("Pilas", "PUNTO LIMPIO", 2)
    segundo = _registro_catalogo(
        "Restos", "RESTO", 3, ruta="data/otro_catalogo.csv"
    )
    coleccion.get.return_value = {
        "documents": [primero[0], segundo[0]],
        "metadatas": [primero[1], segundo[1]],
    }

    with pytest.raises(RuntimeError, match="procedencia"):
        retrieve.consultar_catalogo_residuos()


@pytest.mark.parametrize(
    "respuesta",
    [
        None,
        {},
        {"documents": [], "metadatas": None},
        {"documents": ["texto"], "metadatas": []},
    ],
)
def test_consulta_estructurada_detecta_respuesta_chroma_invalida(
    coleccion_catalogo, respuesta
):
    coleccion, _ = coleccion_catalogo
    coleccion.get.return_value = respuesta

    with pytest.raises(RuntimeError, match="ChromaDB"):
        retrieve.consultar_catalogo_residuos()


def test_catalogo_propaga_fallo_de_chroma(coleccion_catalogo):
    coleccion, _ = coleccion_catalogo
    coleccion.get.side_effect = ConnectionError("Sin acceso")

    with pytest.raises(RuntimeError, match="ChromaDB"):
        retrieve.consultar_catalogo_residuos()


# ---------------------------------------------------------
# REGRESIÓN: PARADAS Y HORARIOS DE PUNTOS LIMPIOS MÓVILES
# ---------------------------------------------------------


def _registro_movil(
    calle,
    distrito,
    dia,
    horario,
    fila,
    *,
    tipo="punto_limpio_movil",
    fuente="puntos_limpios_moviles.csv",
):
    """Reproduce los campos y la puntuación del CSV de móviles."""
    texto = (
        "Punto limpio móvil.\n"
        f"Dirección: {calle}.\n"
        f"Distrito: {distrito}.\n"
        f"Día: {dia}.\n"
        "Turno: MAÑANA.\n"
        f"Horario: {horario}."
    )
    metadata = {
        "source": fuente,
        "path": "data/raw/csv/puntos_limpios_moviles.csv",
        "document_type": tipo,
        "district": distrito,
        "day": dia,
        "source_row": fila,
    }
    return texto, metadata


@pytest.fixture

def coleccion_moviles(monkeypatch):
    """Crea paradas reales en forma y variantes de distrito."""
    simulada = Mock()
    registros = [
        _registro_movil(
            "CALLE ARRIETA, 15", "CENTRO", "MARTES",
            "12:15 - 14:15", 2,
        ),
        _registro_movil(
            "CALLE BARQUILLO, 7", "Centro", "SABADO",
            "07:45 - 09:45", 3,
        ),
        _registro_movil(
            "CALLE LUNA, 4", "ARGANZUELA", "LUNES",
            "10:00 - 12:00", 4,
        ),
        _registro_movil(
            "AVENIDA DEL NORTE, 3", "CENTRO", "MIERCOLES",
            "24 HORAS", 5, tipo="punto_limpio_movil_24h",
            fuente="puntos_limpios_moviles_24h.csv",
        ),
    ]

    def get_simulado(**parametros):
        if "where" not in parametros:
            return {"metadatas": [item[1] for item in registros]}
        return {
            "documents": [item[0] for item in registros],
            "metadatas": [item[1] for item in registros],
        }

    simulada.get.side_effect = get_simulado
    monkeypatch.setattr(
        retrieve,
        "_abrir_coleccion",
        Mock(return_value=(simulada, 712)),
    )
    generar = Mock(side_effect=AssertionError("No debe usarse Gemini"))
    monkeypatch.setattr(retrieve, "_generar_embedding_pregunta", generar)
    return simulada, generar


@pytest.mark.parametrize(
    "criterios",
    [
        {},
        {"dia": "martes"},
        {"distrito": ""},
        {"direccion": " "},
        {"dia": 22, "distrito": "Centro"},
        {"distrito": 7},
        {"distrito": "a" * 501},
    ],
)
def test_moviles_rechaza_criterios_insuficientes_o_invalidos(
    coleccion_moviles, criterios
):
    coleccion, generar = coleccion_moviles

    with pytest.raises(ValueError):
        retrieve.consultar_paradas_moviles(**criterios)

    coleccion.get.assert_not_called()
    generar.assert_not_called()


def test_moviles_agrupa_variantes_de_distrito_sin_mezclar_modalidades(
    coleccion_moviles,
):
    coleccion, generar = coleccion_moviles

    paradas = retrieve.consultar_paradas_moviles(distrito="centro")

    assert len(paradas) == 2
    assert [item["direccion"] for item in paradas] == [
        "CALLE ARRIETA, 15.", "CALLE BARQUILLO, 7."
    ]
    assert [item["dia"] for item in paradas] == ["MARTES", "SABADO"]
    assert [item["horario"] for item in paradas] == [
        "12:15 - 14:15.", "07:45 - 09:45."
    ]
    assert all(
        item["source"] == "puntos_limpios_moviles.csv"
        and item["metadata"]["document_type"] == "punto_limpio_movil"
        for item in paradas
    )
    assert all(item["path"] for item in paradas)
    assert [item["source_row"] for item in paradas] == [2, 3]
    filtro = coleccion.get.call_args.kwargs["where"]
    assert filtro == {
        "$and": [
            {"document_type": "punto_limpio_movil"},
            {"district": {"$in": ["CENTRO", "Centro"]}},
        ]
    }
    coleccion.query.assert_not_called()
    generar.assert_not_called()


def test_moviles_filtra_direccion_y_dia_sin_distrito(coleccion_moviles):
    coleccion, _ = coleccion_moviles

    resultado = retrieve.consultar_paradas_moviles(
        direccion="calle arrieta 15", dia="martes"
    )

    assert len(resultado) == 1
    assert resultado[0]["distrito"] == "CENTRO"
    assert resultado[0]["dia"] == "MARTES"
    assert resultado[0]["horario"] == "12:15 - 14:15."
    assert coleccion.get.call_args.kwargs["where"] == {
        "document_type": "punto_limpio_movil"
    }


def test_moviles_direccion_parcial_respetando_palabras(coleccion_moviles):
    resultado = retrieve.consultar_paradas_moviles(direccion="Arrieta")
    assert len(resultado) == 1
    assert "ARRIETA" in resultado[0]["direccion"]

    no_coincidencia = retrieve.consultar_paradas_moviles(
        direccion="rieta"
    )
    assert no_coincidencia == []


def test_moviles_no_inventa_coincidencias_desconocidas(coleccion_moviles):
    assert retrieve.consultar_paradas_moviles(distrito="No Existe") == []
    assert retrieve.consultar_paradas_moviles(
        distrito="Centro", direccion="Calle Falsa 100"
    ) == []


def test_moviles_filtra_dia_sin_suponer_parada_unica(coleccion_moviles):
    resultado = retrieve.consultar_paradas_moviles(
        distrito="Centro", dia="sábado"
    )

    assert len(resultado) == 1
    assert resultado[0]["direccion"] == "CALLE BARQUILLO, 7."


def test_moviles_descarta_registros_con_metadatos_incompatibles(
    coleccion_moviles,
):
    coleccion, _ = coleccion_moviles
    correcto = _registro_movil(
        "CALLE ARRIETA, 15", "CENTRO", "MARTES",
        "12:15 - 14:15", 2,
    )
    distrito_invalido = _registro_movil(
        "CALLE INVENTADA, 5", "CENTRO", "MARTES",
        "12:15 - 14:15", 3,
    )
    distrito_invalido[1]["district"] = "RETIRO"
    dia_invalido = _registro_movil(
        "CALLE INVENTADA, 6", "CENTRO", "MARTES",
        "12:15 - 14:15", 4,
    )
    dia_invalido[1]["day"] = "VIERNES"
    fuente_invalida = _registro_movil(
        "CALLE INVENTADA, 7", "CENTRO", "MARTES",
        "12:15 - 14:15", 5, fuente="otra.csv",
    )
    registros = [
        correcto, distrito_invalido, dia_invalido, fuente_invalida
    ]

    def get_simulado(**parametros):
        if "where" not in parametros:
            return {"metadatas": [item[1] for item in registros]}
        return {
            "documents": [item[0] for item in registros],
            "metadatas": [item[1] for item in registros],
        }

    coleccion.get.side_effect = get_simulado
    resultado = retrieve.consultar_paradas_moviles(distrito="Centro")

    assert len(resultado) == 1
    assert resultado[0]["source_row"] == 2


def test_moviles_descarta_filas_duplicadas(coleccion_moviles):
    coleccion, _ = coleccion_moviles
    repetida = _registro_movil(
        "CALLE ARRIETA, 15", "CENTRO", "MARTES",
        "12:15 - 14:15", 2,
    )
    coleccion.get.side_effect = lambda **kwargs: {
        "metadatas": [repetida[1], dict(repetida[1])],
        "documents": [repetida[0], repetida[0]],
    }

    resultado = retrieve.consultar_paradas_moviles(direccion="Arrieta")

    assert len(resultado) == 1


def test_moviles_propaga_error_consulta_chroma(coleccion_moviles):
    coleccion, _ = coleccion_moviles
    coleccion.get.side_effect = RuntimeError("Error original")

    with pytest.raises(RuntimeError, match="ChromaDB"):
        retrieve.consultar_paradas_moviles(direccion="Arrieta")


@pytest.mark.parametrize(
    ("entrada", "busqueda", "esperado"),
    [
        ("CALLE ARRIETA, 15.", "calle arrieta 15", True),
        ("CALLE ARRIETA, 15.", "Arrieta", True),
        ("CALLE ARRIETA, 15.", "rieta", False),
        ("CALLE ARRIETA, 15.", "Arrieta 1", False),
    ],
)
def test_moviles_coincidencia_direccion_con_limites_de_palabra(
    entrada, busqueda, esperado
):
    assert retrieve._coincide_direccion(entrada, busqueda) is esperado


def test_catalogo_conserva_filas_iguales_de_destinos_distintos(
    coleccion_catalogo,
):
    """Una fila del CSV puede repetirse en otra categoría de destino."""
    coleccion, _ = coleccion_catalogo
    registros = [
        _registro_catalogo("Pañales", "RESTO", 2),
        _registro_catalogo("Fruta", "ORGÁNICA", 2),
    ]
    coleccion.get.return_value = {
        "documents": [texto for texto, _ in registros],
        "metadatas": [metadata for _, metadata in registros],
    }

    resultado = retrieve.consultar_catalogo_residuos()

    assert resultado["total_relaciones"] == 2
    assert resultado["total_destinos"] == 2
    assert resultado["registros_duplicados"] == 0
