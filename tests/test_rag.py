"""Pruebas unitarias de la orquestación RAG, sin ChromaDB ni Gemini.

Se simulan exclusivamente las dos dependencias de ``src.rag``:
retrieval y generación. La integración real se prueba por separado.
"""

from unittest.mock import Mock, call

import pytest

from config import GENERATION_MODEL, TOP_K
from src import rag


@pytest.fixture
def chunks():
    """Devuelve resultados representativos de buscar_chunks_relevantes."""
    return [
        {
            "text": "Envases de plástico: contenedor amarillo.",
            "distance": 0.21,
            "metadata": {
                "source": "tipos_residuos.csv",
                "path": "data/raw/csv/tipos_residuos.csv",
            },
        },
        {
            "text": "Papel y cartón: contenedor azul.",
            "distance": 0.28,
            "metadata": {
                "source": "fraccion_papel.html",
                "path": "data/raw/html/fraccion_papel.html",
            },
        },
    ]


@pytest.fixture
def respuesta_generada():
    """Simula el contrato de salida validado por generate.py."""
    return {
        "respuesta": "Los envases van al contenedor amarillo. [C1]",
        "abstencion": False,
        "motivo": None,
        "fuentes": [
            {
                "chunk_id": "C1",
                "source": "tipos_residuos.csv",
                "path": "data/raw/csv/tipos_residuos.csv",
            },
        ],
    }


@pytest.fixture
def dependencias(monkeypatch, chunks, respuesta_generada):
    """Intercepta el retrieval y la generación para evitar llamadas API."""
    recuperar = Mock(return_value=chunks)
    generar = Mock(return_value=respuesta_generada)
    monkeypatch.setattr(rag, "buscar_chunks_relevantes", recuperar)
    monkeypatch.setattr(rag, "generar_respuesta", generar)
    return recuperar, generar


def test_respuesta_expone_contrato_completo(
    dependencias, chunks, respuesta_generada
):
    """El resultado une ambos contratos sin perder sus campos."""
    resultado = rag.responder("¿Dónde tiro un envase?")

    assert set(resultado) == {
        "pregunta",
        "respuesta",
        "abstencion",
        "motivo",
        "fuentes",
        "chunks",
        "metricas",
    }
    assert resultado["pregunta"] == "¿Dónde tiro un envase?"
    for campo, valor in respuesta_generada.items():
        assert resultado[campo] == valor
    assert resultado["chunks"] is chunks


def test_llama_una_vez_a_cada_componente(dependencias, chunks):
    """La orquestación no repite retrieval ni generación."""
    recuperar, generar = dependencias
    pregunta = "¿Dónde tiro un envase?"

    rag.responder(pregunta)

    recuperar.assert_called_once_with(
        pregunta, top_k=TOP_K, query_embedding=None
    )
    generar.assert_called_once_with(pregunta, chunks, client=None)


def test_top_k_por_defecto_procede_de_config(dependencias):
    """Respeta el valor centralizado, sin otra constante local."""
    recuperar, _ = dependencias

    resultado = rag.responder("Consulta")

    assert resultado["metricas"]["top_k"] == TOP_K
    assert recuperar.call_args.kwargs["top_k"] == TOP_K


@pytest.mark.parametrize("top_k", sorted({1, 2, TOP_K}))
def test_acepta_top_k_dentro_del_rango(dependencias, top_k):
    """Propaga el K solicitado cuando es válido."""
    recuperar, _ = dependencias

    resultado = rag.responder("Consulta", top_k=top_k)

    assert recuperar.call_args.kwargs["top_k"] == top_k
    assert resultado["metricas"]["top_k"] == top_k


@pytest.mark.parametrize(
    "top_k",
    [0, -1, TOP_K + 1, True, False, None, 2.5, "2", [], {}],
)
def test_rechaza_top_k_invalido_sin_invocar_dependencias(
    dependencias, top_k
):
    """Se detiene antes de abrir ChromaDB o crear un cliente Gemini."""
    recuperar, generar = dependencias

    with pytest.raises(ValueError, match="top_k"):
        rag.responder("Consulta", top_k=top_k)

    recuperar.assert_not_called()
    generar.assert_not_called()


def test_reutiliza_el_embedding_precalculado(dependencias):
    """Pasa al retrieval el vector recibido sin modificarlo."""
    recuperar, _ = dependencias
    vector = [0.12, 0.34, 0.56]

    rag.responder("Consulta", query_embedding=vector)

    assert recuperar.call_args.kwargs["query_embedding"] is vector


def test_no_inventa_embedding_si_no_se_proporciona(dependencias):
    """Deja a retrieval la decisión de generar el vector."""
    recuperar, _ = dependencias

    rag.responder("Consulta")

    assert recuperar.call_args.kwargs["query_embedding"] is None


def test_reutiliza_cliente_gemini_inyectado(dependencias):
    """La propiedad y el cierre del cliente corresponden al llamador."""
    _, generar = dependencias
    cliente = Mock()

    rag.responder("Consulta", client=cliente)

    assert generar.call_args.kwargs["client"] is cliente
    cliente.close.assert_not_called()


def test_sin_cliente_inyectado_delega_su_creacion(dependencias):
    """El orquestador no instancia ni cierra clientes directamente."""
    _, generar = dependencias

    rag.responder("Consulta")

    assert generar.call_args.kwargs["client"] is None


def test_devuelve_pregunta_recortada_sin_alterar_la_original(
    dependencias
):
    """La salida se normaliza, aunque retrieval recibe la entrada."""
    recuperar, generar = dependencias
    pregunta = "  ¿Dónde tiro un envase?  "

    resultado = rag.responder(pregunta)

    assert resultado["pregunta"] == "¿Dónde tiro un envase?"
    assert recuperar.call_args.args[0] == pregunta
    assert generar.call_args.args[0] == pregunta


def test_preserva_los_chunks_y_su_orden(dependencias, chunks):
    """No ordena, recorta ni modifica los documentos recuperados."""
    _, generar = dependencias
    originales = [fragmento.copy() for fragmento in chunks]

    resultado = rag.responder("Consulta")

    assert resultado["chunks"] is chunks
    assert generar.call_args.args[1] is chunks
    assert chunks == originales


def test_reutiliza_fuentes_validadas_por_el_generador(
    dependencias, respuesta_generada
):
    """No reconstruye fuentes a partir de metadatos sin validar."""
    _, generar = dependencias
    fuentes = [
        {"chunk_id": "C2", "source": "fraccion_papel.html", "path": "b"},
        {"chunk_id": "C1", "source": "tipos_residuos.csv", "path": "a"},
    ]
    generar.return_value = {**respuesta_generada, "fuentes": fuentes}

    resultado = rag.responder("Consulta")

    assert resultado["fuentes"] is fuentes
    assert resultado["fuentes"] == fuentes


def test_sin_chunks_propaga_abstencion_de_generacion(dependencias):
    """La ausencia de resultados queda registrada sin crear fuentes."""
    recuperar, generar = dependencias
    recuperar.return_value = []
    generar.return_value = {
        "respuesta": "No hay información suficiente.",
        "abstencion": True,
        "motivo": "sin_contexto",
        "fuentes": [],
    }

    resultado = rag.responder("Consulta")

    generar.assert_called_once_with("Consulta", [], client=None)
    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "sin_contexto"
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    assert resultado["metricas"]["chunks_recuperados"] == 0


@pytest.mark.parametrize(
    ("motivo", "texto"),
    [
        ("evidencia_insuficiente", "No puedo responder."),
        ("fuera_de_dominio", "No puedo responder."),
        ("ambiguedad", "¿Se trata de un envase?"),
    ],
)
def test_conserva_los_motivos_y_textos_de_abstencion(
    dependencias, motivo, texto
):
    """No transforma las abstenciones en respuestas afirmativas."""
    _, generar = dependencias
    generar.return_value = {
        "respuesta": texto,
        "abstencion": True,
        "motivo": motivo,
        "fuentes": [],
    }

    resultado = rag.responder("Consulta")

    assert resultado["respuesta"] == texto
    assert resultado["abstencion"] is True
    assert resultado["motivo"] == motivo
    assert resultado["fuentes"] == []


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Pregunta inválida"),
        FileNotFoundError("No existe ChromaDB"),
        RuntimeError("Fallo del índice"),
    ],
)
def test_propaga_error_de_retrieval_sin_generar(
    dependencias, error
):
    """Los fallos técnicos no se presentan como abstenciones."""
    recuperar, generar = dependencias
    recuperar.side_effect = error

    with pytest.raises(type(error)) as capturado:
        rag.responder("Consulta")

    assert capturado.value is error
    generar.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Gemini no está disponible"),
        ValueError("Contexto inválido"),
    ],
)
def test_propaga_error_de_generacion(dependencias, error):
    """No oculta errores de API o de validación como falta de datos."""
    recuperar, generar = dependencias
    generar.side_effect = error

    with pytest.raises(type(error)) as capturado:
        rag.responder("Consulta")

    assert capturado.value is error
    recuperar.assert_called_once()
    generar.assert_called_once()


def test_no_oculta_contrato_incompleto_de_generacion(dependencias):
    """Una salida con campos ausentes no produce éxito silencioso."""
    _, generar = dependencias
    generar.return_value = {"respuesta": "Texto sin más campos"}

    with pytest.raises(KeyError, match="abstencion"):
        rag.responder("Consulta")


def test_metricas_identifican_modelo_y_documentos(
    dependencias, chunks
):
    """La telemetría contiene el modelo configurado y el total real."""
    resultado = rag.responder("Consulta", top_k=TOP_K)
    metricas = resultado["metricas"]

    assert set(metricas) == {
        "top_k",
        "chunks_recuperados",
        "modelo_configurado",
        "retrieval_ms",
        "generacion_ms",
        "total_ms",
    }
    assert metricas["chunks_recuperados"] == len(chunks)
    assert metricas["modelo_configurado"] == GENERATION_MODEL


def test_tiempos_miden_cada_etapa_sin_solaparse(
    monkeypatch, dependencias
):
    """Usa tres marcas temporales y expresa su diferencia en ms."""
    reloj = Mock(side_effect=[100.0, 100.012, 100.045])
    monkeypatch.setattr(rag, "perf_counter", reloj)

    resultado = rag.responder("Consulta")
    metricas = resultado["metricas"]

    assert reloj.call_count == 3
    assert metricas["retrieval_ms"] == pytest.approx(12.0, abs=0.01)
    assert metricas["generacion_ms"] == pytest.approx(33.0, abs=0.01)
    assert metricas["total_ms"] == pytest.approx(45.0, abs=0.01)


def test_tiempos_reales_son_numericos_y_no_negativos(dependencias):
    """Todas las duraciones se registran en milisegundos."""
    metricas = rag.responder("Consulta")["metricas"]

    for clave in ("retrieval_ms", "generacion_ms", "total_ms"):
        assert isinstance(metricas[clave], float)
        assert metricas[clave] >= 0

    assert metricas["total_ms"] == pytest.approx(
        metricas["retrieval_ms"] + metricas["generacion_ms"],
        abs=0.02,
    )


def test_dos_consultas_no_comparten_estado(dependencias):
    """Cada ejecución conserva su propia pregunta y sus llamadas."""
    recuperar, generar = dependencias

    primera = rag.responder("Consulta uno")
    segunda = rag.responder("Consulta dos")

    assert primera is not segunda
    assert primera["pregunta"] == "Consulta uno"
    assert segunda["pregunta"] == "Consulta dos"
    assert recuperar.call_args_list == [
        call("Consulta uno", top_k=TOP_K, query_embedding=None),
        call("Consulta dos", top_k=TOP_K, query_embedding=None),
    ]
    assert generar.call_count == 2


def test_no_se_llama_a_generacion_con_top_k_invalido(dependencias):
    """La validación temprana evita operaciones innecesarias."""
    recuperar, generar = dependencias

    with pytest.raises(ValueError):
        rag.responder("Consulta", top_k=TOP_K + 10)

    recuperar.assert_not_called()
    generar.assert_not_called()