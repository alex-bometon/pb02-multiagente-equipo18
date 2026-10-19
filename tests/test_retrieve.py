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