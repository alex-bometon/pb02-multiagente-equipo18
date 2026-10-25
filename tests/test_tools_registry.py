"""Pruebas de la allowlist y ejecución de herramientas."""

from unittest.mock import Mock

import pytest

from src.tools import TOOLS, TOOL_BY_NAME, ejecutar_tool


def test_registro_de_herramientas():
    nombres = {
        "consultar_rag",
        "buscar_puntos_limpios_cercanos",
        "hora_actual",
        "describir_imagen",
    }

    assert {tool.name for tool in TOOLS} == nombres
    assert set(TOOL_BY_NAME) == nombres


def test_herramienta_no_autorizada():
    resultado = ejecutar_tool("herramienta_inventada", {})

    assert resultado == "Error: herramienta no permitida."


def test_ejecucion_autorizada(monkeypatch):
    herramienta = Mock()
    herramienta.args = {}
    herramienta.invoke.return_value = "Hora simulada"

    monkeypatch.setitem(
        TOOL_BY_NAME, "hora_actual", herramienta
    )

    resultado = ejecutar_tool("hora_actual", {})

    assert resultado == "Hora simulada"
    herramienta.invoke.assert_called_once_with({})


def test_argumentos_no_permitidos(monkeypatch):
    herramienta = Mock()
    herramienta.args = {}

    monkeypatch.setitem(
        TOOL_BY_NAME, "hora_actual", herramienta
    )

    resultado = ejecutar_tool(
        "hora_actual", {"argumento_extra": "valor"}
    )

    assert resultado == "Error: argumentos no permitidos."
    herramienta.invoke.assert_not_called()


@pytest.mark.parametrize("args", ["texto", 123, [], True])
def test_argumentos_invalidos(args):
    resultado = ejecutar_tool("hora_actual", args)

    assert resultado == "Error: argumentos inválidos."


def test_error_controlado(monkeypatch):
    herramienta = Mock()
    herramienta.args = {}
    herramienta.invoke.side_effect = RuntimeError(
        "GEMINI_API_KEY=secreto"
    )

    monkeypatch.setitem(
        TOOL_BY_NAME, "hora_actual", herramienta
    )

    resultado = ejecutar_tool("hora_actual", {})

    assert resultado.startswith("Error:")
    assert "secreto" not in resultado


def test_argumento_obligatorio_ausente():
    resultado = ejecutar_tool("consultar_rag", {})

    assert resultado.startswith("Error:")