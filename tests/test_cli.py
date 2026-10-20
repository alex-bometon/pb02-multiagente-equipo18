"""Contratos de la CLI; sin dependencias externas ni llamadas a Gemini."""

import sys
from types import ModuleType

import pytest

from main import main


def _simular_modulo(monkeypatch, nombre, **funciones):
    modulo = ModuleType(nombre)
    for clave, funcion in funciones.items():
        setattr(modulo, clave, funcion)
    monkeypatch.setitem(sys.modules, nombre, modulo)


def test_help_muestra_comandos(capsys):
    with pytest.raises(SystemExit) as error:
        main(["rag", "--help"])
    assert error.value.code == 0
    salida = capsys.readouterr().out
    assert "query" in salida
    assert "ask" in salida
    assert "index" in salida


def test_query_muestra_fuente_distancia_y_texto(monkeypatch, capsys):
    llamadas = []

    def recuperar(pregunta, top_k):
        llamadas.append((pregunta, top_k))
        return [{
            "text": "Va al amarillo",
            "distance": 0.125,
            "metadata": {
                "source": "residuos.csv", "path": "data/residuos.csv"
            },
        }]

    _simular_modulo(
        monkeypatch, "src.retrieve", buscar_chunks_relevantes=recuperar
    )
    assert main(["rag", "query", "plástico", "--top-k", "3"]) == 0
    assert llamadas == [("plástico", 3)]
    salida = capsys.readouterr().out
    assert "0.1250" in salida
    assert "residuos.csv" in salida
    assert "Va al amarillo" in salida


def test_ask_muestra_respuesta_fuentes_y_metricas(monkeypatch, capsys):
    def responder(pregunta, top_k):
        assert (pregunta, top_k) == ("¿Y el vidrio?", 5)
        return {
            "respuesta": "Contenedor verde [C1]",
            "abstencion": False,
            "motivo": None,
            "fuentes": [{
                "chunk_id": "C1", "source": "vidrio.html", "path": "data/x"
            }],
            "chunks": [{"text": "Envases de vidrio", "metadata": {}}],
            "metricas": {
                "chunks_recuperados": 1, "total_ms": 200,
                "modelo_configurado": "gemini-test",
            },
        }

    _simular_modulo(
        monkeypatch, "src.rag", responder=responder
    )
    assert main(["rag", "ask", "¿Y el vidrio?", "--contexto"]) == 0
    salida = capsys.readouterr().out
    assert "Contenedor verde [C1]" in salida
    assert "vidrio.html" in salida
    assert "200 ms" in salida
    assert "Envases de vidrio" in salida


def test_index_no_modifica_sin_autorizacion(capsys):
    assert main(["rag", "index"]) == 2
    assert "No se ha modificado" in capsys.readouterr().err


def test_index_usa_embeddings_existentes(monkeypatch):
    llamadas = []
    _simular_modulo(
        monkeypatch, "src.index",
        ejecutar_indexacion=lambda: llamadas.append("index"),
    )
    assert main(["rag", "index", "--yes"]) == 0
    assert llamadas == ["index"]


def test_index_full_respeta_orden(monkeypatch):
    llamadas = []
    for modulo, nombre, etapa in (
        ("src.ingest", "ejecutar_ingesta", "ingest"),
        ("src.embed", "ejecutar_embeddings", "embed"),
        ("src.index", "ejecutar_indexacion", "index"),
    ):
        _simular_modulo(
            monkeypatch, modulo,
            **{nombre: lambda paso=etapa: llamadas.append(paso)},
        )
    assert main(["rag", "index", "--full", "--yes"]) == 0
    assert llamadas == ["ingest", "embed", "index"]


def test_error_controlado_no_imprime_traceback(monkeypatch, capsys):
    def fallar(pregunta, top_k):
        raise ValueError("top_k incorrecto")

    _simular_modulo(
        monkeypatch, "src.retrieve", buscar_chunks_relevantes=fallar
    )
    assert main(["rag", "query", "vidrio"]) == 1
    assert "top_k incorrecto" in capsys.readouterr().err
