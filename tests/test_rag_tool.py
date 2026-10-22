
"""Pruebas unitarias de la herramienta RAG de ReciclaTIA."""

from unittest.mock import patch

from src.tools.rag_tool import consultar_rag


def test_consultar_rag_delega_y_devuelve_resultado():
    """Comprueba que la tool utiliza el RAG sin alterar su respuesta."""

    pregunta = "¿Dónde tiro una botella de plástico?"

    respuesta_simulada = {
        "pregunta": pregunta,
        "respuesta": "Contenedor amarillo",
        "abstencion": False,
        "motivo": None,
        "fuentes": [],
        "chunks": [],
        "metricas": {},
    }

    with patch(
        "src.tools.rag_tool.responder",
        return_value=respuesta_simulada,
    ) as mock_responder:

        resultado = consultar_rag.invoke({"pregunta": pregunta})

        mock_responder.assert_called_once_with(pregunta)
        assert resultado == respuesta_simulada
