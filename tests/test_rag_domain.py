"""Pruebas de dominio y reconocimiento de teléfonos móviles."""

from unittest.mock import Mock, call

import pytest

from config import GENERATION_MODEL, TOP_K
from src import rag


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué servicios de reparación de móviles hay cerca de Gran Vía?",
        "¿Dónde hay talleres de reparación de teléfonos?",
        "Busco un servicio técnico para smartphones en Madrid.",
        "¿Cuánto cuesta reparar mi móvil?",
        "¿Hay tiendas que reparan teléfonos?",
        "¿Qué técnicos reparan móviles cerca de aquí?",
        "Necesito un servicio técnico de teléfonos.",
        "¿Dónde encuentro reparación de móviles?",
        "¿Hay talleres para arreglar mi smartphone?",
        "¿Qué servicios de reparación de teléfonos hay en Barcelona?",
        "¿Dónde se reparan móviles por Gran Vía?",
        "¿Dónde hay tiendas de reparación de celulares?",
        "¿Qué precio tiene reparar el móvil?",
        "¿Dónde arreglan teléfonos?",
    ],
)
def test_reparacion_telefonos_rechaza_servicio_sin_acceso_externo(
    pregunta, consultas_estructuradas, dependencias,
):
    """Los servicios técnicos no pertenecen al dominio municipal RAG."""
    catalogo, paradas = consultas_estructuradas
    recuperar, generar = dependencias
    cliente = Mock()
    embedding = [0.01, 0.02, 0.03]

    resultado = rag.responder(
        pregunta, query_embedding=embedding, client=cliente,
    )

    assert resultado["pregunta"] == pregunta.strip()
    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["respuesta"] == rag.MENSAJE_ABSTENCION
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    assert resultado["metricas"]["chunks_recuperados"] == 0
    assert resultado["metricas"]["top_k"] == TOP_K
    catalogo.assert_not_called()
    paradas.assert_not_called()
    recuperar.assert_not_called()
    generar.assert_not_called()
    cliente.close.assert_not_called()
    cliente.models.generate_content.assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Dónde reciclo un teléfono móvil roto?",
        "¿Dónde tiro mi móvil que no se puede reparar?",
        "¿Qué hago con un móvil para reciclarlo?",
        "¿Se puede reutilizar un teléfono que no se puede reparar?",
        "¿Dónde depositar teléfonos móviles irreparables?",
        "¿Dónde se deposita un móvil para su recogida?",
        "¿Cómo reciclar teléfonos reparados?",
        "¿Dónde hay una tienda de reutilización de móviles?",
    ],
)
def test_residuos_electronicos_no_se_confunden_con_reparacion(
    pregunta, consultas_estructuradas, dependencias,
):
    """Reciclar o desechar teléfonos conserva la vía semántica RAG."""
    catalogo, paradas = consultas_estructuradas
    recuperar, generar = dependencias

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is False
    recuperar.assert_called_once_with(
        pregunta, top_k=TOP_K, query_embedding=None,
    )
    generar.assert_called_once()
    catalogo.assert_not_called()
    paradas.assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué horario tiene el punto limpio móvil?",
        "¿A qué hora pasa el punto limpio móvil en Madrid?",
    ],
)
def test_reparacion_no_intercepta_horarios_de_puntos_limpios(
    pregunta, consultas_estructuradas, dependencias,
):
    """La palabra «móvil» no convierte un punto limpio en teléfono."""
    catalogo, paradas = consultas_estructuradas
    recuperar, generar = dependencias

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "ambiguedad"
    assert "distrito o calle" in resultado["respuesta"]
    catalogo.assert_not_called()
    paradas.assert_not_called()
    recuperar.assert_not_called()
    generar.assert_not_called()


def test_servicios_reparacion_priorizan_abstencion_antes_de_retrieval(
    consultas_estructuradas, dependencias,
):
    """El resultado no depende de K, embeddings ni cliente inyectado."""
    catalogo, paradas = consultas_estructuradas
    recuperar, generar = dependencias
    cliente = Mock()

    resultado = rag.responder(
        "¿Dónde hay talleres de reparación de móviles?",
        top_k=1,
        query_embedding=[0.5, 0.6],
        client=cliente,
    )

    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["metricas"]["top_k"] == 1
    assert resultado["metricas"]["total_ms"] >= 0
    catalogo.assert_not_called()
    paradas.assert_not_called()
    recuperar.assert_not_called()
    generar.assert_not_called()
    cliente.close.assert_not_called()


def test_servicio_reparacion_celular_singular_fuera_de_dominio(
    consultas_estructuradas, dependencias,
):
    """Documenta el caso límite conocido sin alterar el módulo probado."""
    catalogo, paradas = consultas_estructuradas
    recuperar, generar = dependencias

    resultado = rag.responder("¿Dónde puedo arreglar un celular?")

    assert resultado["motivo"] == "fuera_de_dominio"
    catalogo.assert_not_called()
    paradas.assert_not_called()
    recuperar.assert_not_called()
    generar.assert_not_called()
