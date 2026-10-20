"""Pruebas de recuperación semántica sin llamadas reales a Gemini."""

from unittest.mock import Mock

import pytest

from config import EMBEDDING_DIMENSIONS
from src import retrieve


@pytest.fixture
def vector():
    """Devuelve un embedding válido para las pruebas."""
    return [0.1] * EMBEDDING_DIMENSIONS


@pytest.fixture
def coleccion():
    """Simula una colección ChromaDB con tres resultados."""
    simulada = Mock()
    simulada.query.return_value = {
        "documents": [["Texto A", "Texto B", "Texto C"]],
        "metadatas": [[
            {"source": "a.csv"},
            {"source": "b.html"},
            {"source": "c.csv"},
        ]],
        "distances": [[0.3, 0.1, 0.2]],
    }
    return simulada


@pytest.fixture
def retrieval_simulado(monkeypatch, coleccion, vector):
    """Aísla las pruebas de Gemini y del índice real."""
    abrir = Mock(return_value=(coleccion, 712))
    generar = Mock(return_value=vector)
    monkeypatch.setattr(retrieve, "_abrir_coleccion", abrir)
    monkeypatch.setattr(retrieve, "_generar_embedding_pregunta", generar)
    return abrir, generar


@pytest.mark.parametrize("pregunta", ["", "   ", None, 123])
def test_rechaza_preguntas_vacias_o_invalidas(pregunta):
    with pytest.raises(ValueError):
        retrieve.buscar_chunks_relevantes(pregunta)


def test_rechaza_pregunta_demasiado_larga():
    with pytest.raises(ValueError, match="500"):
        retrieve.buscar_chunks_relevantes("a" * 501)


@pytest.mark.parametrize("top_k", [0, -1, 11, 2.5, "3", True])
def test_rechaza_top_k_invalido(top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieve.buscar_chunks_relevantes("¿Dónde reciclo?", top_k=top_k)


@pytest.mark.parametrize(
    "vector_invalido",
    [
        [],
        [0.1] * (EMBEDDING_DIMENSIONS - 1),
        [float("nan")] * EMBEDDING_DIMENSIONS,
        [float("inf")] * EMBEDDING_DIMENSIONS,
        [True] * EMBEDDING_DIMENSIONS,
        "no es un vector",
    ],
)
def test_rechaza_embeddings_invalidos(vector_invalido):
    with pytest.raises(ValueError):
        retrieve.buscar_chunks_relevantes(
            "¿Dónde reciclo?", query_embedding=vector_invalido
        )


def test_vector_precalculado_no_llama_a_gemini(
    monkeypatch, coleccion, vector
):
    abrir = Mock(return_value=(coleccion, 712))
    generar = Mock()
    monkeypatch.setattr(retrieve, "_abrir_coleccion", abrir)
    monkeypatch.setattr(retrieve, "_generar_embedding_pregunta", generar)

    resultados = retrieve.buscar_chunks_relevantes(
        "¿Dónde reciclo una bombilla?",
        top_k=3,
        query_embedding=vector,
    )

    generar.assert_not_called()
    assert len(resultados) == 3
    coleccion.query.assert_called_once_with(
        query_embeddings=[vector],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )


def test_genera_embedding_si_no_se_proporciona(
    retrieval_simulado, coleccion
):
    _, generar = retrieval_simulado

    retrieve.buscar_chunks_relevantes("  ¿Dónde reciclo?  ", top_k=3)

    generar.assert_called_once_with("¿Dónde reciclo?")
    assert coleccion.query.call_count == 1


def test_devuelve_resultados_ordenados_y_con_fuentes(
    retrieval_simulado, vector
):
    resultados = retrieve.buscar_chunks_relevantes(
        "Consulta", top_k=3, query_embedding=vector
    )

    assert [item["distance"] for item in resultados] == [0.1, 0.2, 0.3]
    assert [item["metadata"]["source"] for item in resultados] == [
        "b.html", "c.csv", "a.csv"
    ]
    assert resultados[0]["text"] == "Texto B"


def test_limita_resultados_a_los_disponibles(
    retrieval_simulado, coleccion, vector
):
    abrir, _ = retrieval_simulado
    abrir.return_value = (coleccion, 2)

    retrieve.buscar_chunks_relevantes(
        "Consulta", top_k=5, query_embedding=vector
    )

    assert coleccion.query.call_args.kwargs["n_results"] == 2


def test_propaga_error_de_consulta_chroma(
    retrieval_simulado, coleccion, vector
):
    coleccion.query.side_effect = ConnectionError("Fallo simulado")

    with pytest.raises(RuntimeError, match="ChromaDB"):
        retrieve.buscar_chunks_relevantes(
            "Consulta", query_embedding=vector
        )


def test_detecta_respuesta_incompleta(
    retrieval_simulado, coleccion, vector
):
    coleccion.query.return_value = {
        "documents": [["Texto A"]],
        "metadatas": [[{"source": "a.csv"}]],
    }

    with pytest.raises(RuntimeError, match="incompleta"):
        retrieve.buscar_chunks_relevantes(
            "Consulta", query_embedding=vector
        )


def test_detecta_respuesta_inconsistente(
    retrieval_simulado, coleccion, vector
):
    coleccion.query.return_value = {
        "documents": [["Texto A", "Texto B"]],
        "metadatas": [[{"source": "a.csv"}]],
        "distances": [[0.1, 0.2]],
    }

    with pytest.raises(RuntimeError, match="inconsistentes"):
        retrieve.buscar_chunks_relevantes(
            "Consulta", query_embedding=vector
        )


def test_rechaza_indice_inexistente(monkeypatch, tmp_path, vector):
    monkeypatch.setattr(retrieve, "CHROMA_DIR", tmp_path / "no_existe")

    with pytest.raises(FileNotFoundError):
        retrieve._abrir_coleccion()


def test_integracion_local_sin_gemini():
    """Comprueba el índice real usando un embedding ya almacenado."""
    from config import CHROMA_DIR, EMBEDDINGS_JSON
    from src.artifacts import cargar_embeddings_json

    if not CHROMA_DIR.is_dir() or not EMBEDDINGS_JSON.is_file():
        pytest.skip("No están disponibles los artefactos locales del RAG.")

    vector_real = cargar_embeddings_json()[0]["vector"]
    resultados = retrieve.buscar_chunks_relevantes(
        "Comprobación local",
        top_k=3,
        query_embedding=vector_real,
    )

    assert len(resultados) == 3
    assert all(item["text"] for item in resultados)
    assert all("source" in item["metadata"] for item in resultados)

# ---------------------------------------------------------
# REGRESIÓN: FILTROS DE MODALIDAD Y DISTRITO
# ---------------------------------------------------------


@pytest.fixture
def coleccion_con_distritos():
    """Simula las variantes de distrito existentes en el corpus."""
    simulada = Mock()
    simulada.get.return_value = {
        "metadatas": [
            {"district": "ARGANZUELA"},
            {"district": "Arganzuela"},
            {"district": "HORTALEZA"},
            {"district": "Hortaleza"},
            {"district": "Fuencarral-El Pardo"},
            {"district": "Fuencarral - El Pardo"},
            {"district": "FUENCARRAL-EL PARDO"},
        ],
        "ids": ["item-1"],
    }
    return simulada


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        ("ARGANZUELA", "arganzuela"),
        ("Arganzuela", "arganzuela"),
        ("Hortaleza", "hortaleza"),
        ("Fuencarral-El Pardo", "fuencarral el pardo"),
        ("Fuencarral - El Pardo", "fuencarral el pardo"),
        ("FUENCARRAL-EL PARDO", "fuencarral el pardo"),
    ],
)
def test_normaliza_variantes_distrito(nombre, esperado):
    assert retrieve._normalizar_distrito(nombre) == esperado


@pytest.mark.parametrize(
    ("pregunta", "modalidad"),
    [
        ("¿Dónde hay un punto limpio fijo?", "punto_limpio_fijo"),
        ("¿Qué horario tiene el punto limpio móvil?", "punto_limpio_movil"),
        (
            "¿Dónde hay un punto limpio móvil 24 horas?",
            "punto_limpio_movil_24h",
        ),
        (
            "¿Dónde hay un punto limpio de proximidad?",
            "punto_limpio_proximidad",
        ),
    ],
)
def test_detecta_modalidades_inequivocas(pregunta, modalidad):
    pregunta = retrieve._normalizar_texto(pregunta)
    assert retrieve._detectar_modalidad(pregunta) == modalidad


def test_arganzuela_conserva_las_dos_variantes(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Dónde hay un punto limpio fijo en el distrito de Arganzuela?",
        coleccion_con_distritos,
    )

    assert filtro == {
        "$and": [
            {"document_type": "punto_limpio_fijo"},
            {"district": {"$in": ["ARGANZUELA", "Arganzuela"]}},
        ],
    }


def test_hortaleza_conserva_las_dos_variantes(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Cuál es la dirección del punto limpio fijo de Hortaleza?",
        coleccion_con_distritos,
    )

    assert filtro == {
        "$and": [
            {"document_type": "punto_limpio_fijo"},
            {"district": {"$in": ["HORTALEZA", "Hortaleza"]}},
        ],
    }


def test_fuencarral_reune_variantes_de_guiones(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Dónde hay un punto limpio fijo en Fuencarral-El Pardo?",
        coleccion_con_distritos,
    )

    assert filtro == {
        "$and": [
            {"document_type": "punto_limpio_fijo"},
            {
                "district": {
                    "$in": [
                        "FUENCARRAL-EL PARDO",
                        "Fuencarral - El Pardo",
                        "Fuencarral-El Pardo",
                    ]
                }
            },
        ],
    }


def test_rechaza_filtro_con_dos_distritos(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Hay puntos limpios fijos en el distrito de Arganzuela o "
        "Hortaleza?",
        coleccion_con_distritos,
    )

    assert filtro is None


def test_no_filtra_si_se_piden_dos_modalidades(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Dónde hay puntos limpios fijos o móviles?",
        coleccion_con_distritos,
    )

    assert filtro is None


def test_no_filtra_puntos_limpios_genericos(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Dónde puedo llevar una bombilla a un punto limpio?",
        coleccion_con_distritos,
    )

    assert filtro is None


def test_catalogo_se_filtra_por_fuente(coleccion_con_distritos):
    filtro = retrieve._construir_filtro(
        "¿Qué tipos de residuos reconoce el Ayuntamiento de Madrid?",
        coleccion_con_distritos,
    )

    assert filtro == {"source": "tipos_residuos.csv"}


def test_retirar_residuo_no_activa_filtro_catalogo(
    coleccion_con_distritos,
):
    filtro = retrieve._construir_filtro(
        "¿Dónde tiro los restos de comida y las cáscaras de fruta?",
        coleccion_con_distritos,
    )

    assert filtro is None


def test_retrieval_envia_filtro_arganzuela_a_chroma(
    retrieval_simulado,
    coleccion,
    coleccion_con_distritos,
    vector,
):
    coleccion.get.return_value = coleccion_con_distritos.get.return_value

    retrieve.buscar_chunks_relevantes(
        "¿Dónde hay un punto limpio fijo en el distrito de Arganzuela?",
        top_k=5,
        query_embedding=vector,
    )

    assert coleccion.query.call_args.kwargs["where"] == {
        "$and": [
            {"document_type": "punto_limpio_fijo"},
            {"district": {"$in": ["ARGANZUELA", "Arganzuela"]}},
        ],
    }
    assert coleccion.query.call_count == 1


def test_filtro_sin_registros_no_invoca_gemini_ni_query(
    retrieval_simulado,
    coleccion,
    coleccion_con_distritos,
):
    _, generar = retrieval_simulado

    def simular_get(**parametros):
        if "where" in parametros:
            return {"ids": []}
        return coleccion_con_distritos.get.return_value

    coleccion.get.side_effect = simular_get

    resultado = retrieve.buscar_chunks_relevantes(
        "¿Dónde hay un punto limpio fijo en Arganzuela?",
    )

    assert resultado == []
    generar.assert_not_called()
    coleccion.query.assert_not_called()


def test_catalogo_devuelve_destinos_sin_duplicados(
    monkeypatch,
    coleccion,
):
    coleccion.get.return_value = {
        "metadatas": [
            {"destino": "PUNTO LIMPIO FIJO"},
            {"destino": "CONTENEDOR RESTO"},
            {"destino": "PUNTO LIMPIO FIJO"},
            {"destino": ""},
            {"source": "tipos_residuos.csv"},
        ]
    }
    monkeypatch.setattr(
        retrieve,
        "_abrir_coleccion",
        Mock(return_value=(coleccion, 712)),
    )

    destinos = retrieve.listar_destinos_residuos()

    assert destinos == ["CONTENEDOR RESTO", "PUNTO LIMPIO FIJO"]
    coleccion.get.assert_called_once_with(
        where={"source": "tipos_residuos.csv"},
        include=["metadatas"],
    )


# ---------------------------------------------------------
# INTEGRACIÓN LOCAL SIN CONSUMIR GEMINI
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("distrito", "consulta"),
    [
        (
            "ARGANZUELA",
            "¿Dónde hay un punto limpio fijo en el distrito de Arganzuela?",
        ),
        (
            "HORTALEZA",
            "¿Cuál es la dirección del punto limpio fijo de Hortaleza?",
        ),
    ],
)
def test_integracion_filtra_punto_fijo_y_distrito(
    distrito,
    consulta,
):
    """Comprueba la modalidad y el distrito del registro real."""
    from config import CHROMA_DIR, EMBEDDINGS_JSON
    from src.artifacts import cargar_embeddings_json

    if not CHROMA_DIR.is_dir() or not EMBEDDINGS_JSON.is_file():
        pytest.skip("No están disponibles los artefactos locales del RAG.")

    items = cargar_embeddings_json()
    registro = next(
        (
            item for item in items
            if item["metadata"].get("document_type") == "punto_limpio_fijo"
            and item["metadata"].get("district") == distrito
        ),
        None,
    )
    assert registro is not None, f"Falta el punto fijo de {distrito}."

    resultados = retrieve.buscar_chunks_relevantes(
        consulta,
        top_k=5,
        query_embedding=registro["vector"],
    )

    assert len(resultados) == 1
    assert resultados[0]["metadata"]["source"] == (
        "puntos_limpios_fijos.csv"
    )
    assert resultados[0]["metadata"]["document_type"] == (
        "punto_limpio_fijo"
    )
    assert resultados[0]["metadata"]["district"] == distrito
