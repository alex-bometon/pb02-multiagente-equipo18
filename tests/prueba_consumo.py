"""Mide el consumo real de Gemini en una consulta de ReciclaTIA.

Esta prueba no se ejecuta automáticamente con pytest para evitar
consumir cuota durante las pruebas habituales del proyecto.
"""

from unittest.mock import patch

from google.genai.models import Models

from src.agent import run_demo


def test_consumo_gemini():
    """Verifica el número de solicitudes de una consulta sencilla."""
    llamadas = {
        "embedding": 0,
        "generacion": 0,
    }

    tokens = {
        "entrada": 0,
        "salida": 0,
        "total": 0,
    }

    embedding_original = Models.embed_content
    generacion_original = Models.generate_content

    def contar_embedding(cliente, *args, **kwargs):
        """Registra cada invocación al modelo de embeddings."""
        llamadas["embedding"] += 1

        if llamadas["embedding"] > 1:
            raise AssertionError("Embedding repetido innecesariamente.")

        return embedding_original(cliente, *args, **kwargs)

    def contar_generacion(cliente, *args, **kwargs):
        """Registra la generación y sus tokens de uso."""
        llamadas["generacion"] += 1

        if llamadas["generacion"] > 1:
            raise AssertionError("Generación repetida innecesariamente.")

        respuesta = generacion_original(
            cliente,
            *args,
            **kwargs,
        )

        uso = getattr(respuesta, "usage_metadata", None)

        if uso is not None:
            tokens["entrada"] += getattr(uso, "prompt_token_count", 0) or 0
            tokens["salida"] += getattr(uso, "candidates_token_count", 0) or 0
            tokens["total"] += getattr(uso, "total_token_count", 0) or 0

        return respuesta

    def bloquear_llm(*args, **kwargs):
        """Impide llamadas redundantes desde los agentes."""
        raise AssertionError("Llamada LLM innecesaria desde un agente.")

    with (
        patch.object(
            Models,
            "embed_content",
            contar_embedding,
        ),
        patch.object(
            Models,
            "generate_content",
            contar_generacion,
        ),
        patch(
            "src.nodes.get_llm",
            side_effect=bloquear_llm,
        ),
        patch(
            "src.agent_tool_loop.get_llm",
            side_effect=bloquear_llm,
        ),
    ):
        estado = run_demo(auto_aprobar=False)

    print("\n=== SOLICITUDES AL SDK ===")
    print("Embeddings:", llamadas["embedding"])
    print("Generaciones:", llamadas["generacion"])
    print("Total:", sum(llamadas.values()))

    print("\n=== TOKENS DE GENERACIÓN ===")
    print("Entrada:", tokens["entrada"])
    print("Salida:", tokens["salida"])
    print("Total:", tokens["total"])

    print("\n=== RESULTADO DEL AGENTE ===")
    print("Respuesta:", estado["respuesta"])
    print("HITL:", estado["pendiente_hitl"])
    print("DONE:", estado["done"])
    print("Error:", estado["error"])

    print("\n=== TRAZABILIDAD ===")
    print("Fuentes:", estado["fuentes"])
    print("Herramientas:", estado["traza_tools"])

    assert estado["error"] is None, estado["error"]
    assert estado["pendiente_hitl"] is True
    assert estado["done"] is False

    assert llamadas == {
        "embedding": 1,
        "generacion": 1,
    }

    assert sum(
        traza.get("tool") == "consultar_rag"
        and traza.get("status") == "ok"
        for traza in estado["traza_tools"]
    ) == 1

    assert any(
        evento["nodo"] == "supervisor"
        and evento["accion"] == "ruta_directa"
        for evento in estado["log"]
    )
