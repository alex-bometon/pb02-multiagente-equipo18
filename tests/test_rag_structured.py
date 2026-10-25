"""Pruebas de consultas estructuradas del RAG municipal."""

from unittest.mock import Mock, call

import pytest

from config import GENERATION_MODEL, TOP_K
from src import rag


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué tipos de residuos reconoce el Ayuntamiento de Madrid?",
        "¿Cuáles son los tipos de residuos?",
        "Muéstrame el catálogo de residuos",
        "¿Qué categorías de residuos hay?",
        "Dame un listado de residuos",
    ],
)
def test_catalogo_general_usa_lectura_estructurada(
    pregunta, consultas_estructuradas, dependencias,
):
    """No responde una pregunta global con cinco filas semánticas."""
    recuperar, generar = dependencias
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(pregunta)

    catalogo.assert_called_once_with()
    moviles.assert_not_called()
    recuperar.assert_not_called()
    generar.assert_not_called()
    assert resultado["abstencion"] is False
    assert resultado["motivo"] is None
    assert "5 relaciones" in resultado["respuesta"]
    assert "2 destinos" in resultado["respuesta"]
    assert "CONTENEDOR AMARILLO" in resultado["respuesta"]
    assert "CONTENEDOR AZUL" in resultado["respuesta"]
    assert "no constituyen" in resultado["respuesta"].lower()
    assert resultado["respuesta"].count("[C1]") == 1


def test_catalogo_conserva_fuente_y_marca_resumen_derivado(
    consultas_estructuradas, dependencias,
):
    """Una agregación es evidencia derivada, no un chunk inventado."""
    resultado = rag.responder("¿Qué tipos de residuos hay?")

    assert resultado["fuentes"] == [{
        "chunk_id": "C1",
        "source": "tipos_residuos.csv",
        "path": "data/raw/csv/tipos_residuos.csv",
    }]
    assert len(resultado["chunks"]) == 1
    metadata = resultado["chunks"][0]["metadata"]
    assert metadata["document_type"] == "resumen_catalogo_derivado"
    assert metadata["derived"] is True
    assert metadata["total_relaciones"] == 5
    assert metadata["total_destinos"] == 2
    assert "3 relaciones" in resultado["chunks"][0]["text"]
    assert resultado["metricas"]["chunks_recuperados"] == 1
    assert resultado["metricas"]["modelo_configurado"] == GENERATION_MODEL


@pytest.mark.parametrize("destinos,ruta", [([], "data/catalogo.csv"),
                                            ([], None)])
def test_catalogo_vacio_se_abstiene_sin_inventar_fuentes(
    consultas_estructuradas, dependencias, destinos, ruta,
):
    """La ausencia de registros nunca produce una lista ficticia."""
    catalogo, _ = consultas_estructuradas
    catalogo.return_value = {
        "source": "tipos_residuos.csv",
        "path": ruta,
        "total_relaciones": 0,
        "total_destinos": 0,
        "destinos": destinos,
    }

    resultado = rag.responder("¿Qué tipos de residuos hay?")

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "sin_contexto"
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_catalogo_propaga_error_de_chroma_sin_generar(
    consultas_estructuradas, dependencias,
):
    """Un fallo técnico no es evidencia insuficiente."""
    catalogo, _ = consultas_estructuradas
    catalogo.side_effect = RuntimeError("Error de ChromaDB")

    with pytest.raises(RuntimeError, match="ChromaDB"):
        rag.responder("¿Qué tipos de residuos reconoce Madrid?")

    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Dónde tiro un envase de plástico?",
        "¿Qué tipos de residuos van al contenedor amarillo?",
        "¿Dónde depositar este tipo de residuo?",
    ],
)
def test_consulta_concreta_conserva_retrieval_semantico(
    pregunta, consultas_estructuradas, dependencias,
):
    """La vía agregada no intercepta preguntas de destino concreto."""
    recuperar, generar = dependencias
    catalogo, moviles = consultas_estructuradas

    rag.responder(pregunta)

    recuperar.assert_called_once()
    generar.assert_called_once()
    catalogo.assert_not_called()
    moviles.assert_not_called()


def test_catalogo_de_otra_ciudad_no_usa_datos_de_madrid(
    consultas_estructuradas, dependencias,
):
    """La ruta estructurada respeta el dominio territorial."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder("¿Qué tipos de residuos hay en Barcelona?")

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["fuentes"] == []
    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué horario tiene el punto limpio móvil?",
        "¿A qué hora pasa el punto limpio móvil?",
        "¿Qué días visita el punto limpio móvil?",
    ],
)
def test_horario_movil_sin_ubicacion_pide_aclaracion(
    pregunta, consultas_estructuradas, dependencias,
):
    """No inventa horarios de una parada no identificada."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "ambiguedad"
    assert "distrito o calle" in resultado["respuesta"]
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    assert resultado["metricas"]["chunks_recuperados"] == 0
    moviles.assert_not_called()
    catalogo.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_por_distrito_con_varias_paradas_aclara(
    consultas_estructuradas, dependencias,
):
    """No asigna una hora aleatoria a todo un distrito."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil de Centro?"
    )

    moviles.assert_called_once_with(
        distrito="centro", direccion=None, dia=None
    )
    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "ambiguedad"
    assert "2 registros" in resultado["respuesta"]
    assert resultado["chunks"] == []
    assert resultado["fuentes"] == []
    catalogo.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_por_distrito_con_una_parada_responde(
    consultas_estructuradas, paradas_estructuradas, dependencias,
):
    """Una parada inequívoca sí permite dar horario y dirección."""
    _, moviles = consultas_estructuradas
    moviles.return_value = paradas_estructuradas[:1]

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil de Centro?"
    )

    assert resultado["abstencion"] is False
    assert "CALLE ARRIETA, 15" in resultado["respuesta"]
    assert "12:15 - 14:15" in resultado["respuesta"]
    assert "MARTES" in resultado["respuesta"]
    assert "[C1]" in resultado["respuesta"]
    assert len(resultado["fuentes"]) == 1
    assert resultado["chunks"][0]["metadata"]["day"] == "MARTES"
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_por_calle_y_dia_transmite_filtros(
    consultas_estructuradas, paradas_estructuradas, dependencias,
):
    """La búsqueda identifica calle y día sin perder el número."""
    _, moviles = consultas_estructuradas
    moviles.return_value = paradas_estructuradas[:1]

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil de "
        "la calle Arrieta, 15 el martes?"
    )

    moviles.assert_called_once_with(
        distrito=None, direccion="calle arrieta, 15", dia="martes"
    )
    assert resultado["abstencion"] is False
    assert "CALLE ARRIETA, 15" in resultado["respuesta"]
    assert "12:15 - 14:15" in resultado["respuesta"]
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horarios_plurales_enumeran_paradas_con_citas(
    consultas_estructuradas, paradas_estructuradas, dependencias,
):
    """Cada franja conserva su dirección y referencia original."""
    _, moviles = consultas_estructuradas

    resultado = rag.responder(
        "¿Cuáles son los horarios de los puntos limpios móviles de Centro?"
    )

    moviles.assert_called_once_with(
        distrito="centro", direccion=None, dia=None
    )
    assert resultado["abstencion"] is False
    assert "CALLE ARRIETA, 15" in resultado["respuesta"]
    assert "CALLE BARQUILLO, 7" in resultado["respuesta"]
    assert "12:15 - 14:15" in resultado["respuesta"]
    assert "07:45 - 09:45" in resultado["respuesta"]
    assert "[C1]" in resultado["respuesta"]
    assert "[C2]" in resultado["respuesta"]
    assert len(resultado["fuentes"]) == 2
    assert len(resultado["chunks"]) == 2
    assert resultado["fuentes"][0]["source"] == (
        "puntos_limpios_moviles.csv"
    )
    assert resultado["chunks"][1]["text"] == (
        paradas_estructuradas[1]["text"]
    )
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horarios_recortados_advierten_que_no_son_exhaustivos(
    consultas_estructuradas, dependencias,
):
    """TOP_K limita filas mostradas sin ocultar coincidencias."""
    resultado = rag.responder(
        "¿Cuáles son los horarios de los puntos limpios móviles de Centro?",
        top_k=1,
    )

    assert resultado["abstencion"] is False
    assert "2 registros" in resultado["respuesta"]
    assert "Se muestran 1" in resultado["respuesta"]
    assert "CALLE ARRIETA, 15" in resultado["respuesta"]
    assert "CALLE BARQUILLO, 7" not in resultado["respuesta"]
    assert resultado["metricas"]["top_k"] == 1
    assert resultado["metricas"]["chunks_recuperados"] == 1
    assert len(resultado["fuentes"]) == 1
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_movil_con_dia_acota_busqueda(
    consultas_estructuradas, paradas_estructuradas,
):
    """Extrae la condición del día sin incluirla en el distrito."""
    _, moviles = consultas_estructuradas
    moviles.return_value = paradas_estructuradas[:1]

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil "
        "del distrito de Centro el martes?"
    )

    moviles.assert_called_once_with(
        distrito="centro", direccion=None, dia="martes"
    )
    assert resultado["abstencion"] is False
    assert "MARTES" in resultado["respuesta"]


def test_horario_movil_sin_registros_no_inventa(
    consultas_estructuradas, dependencias,
):
    """Una búsqueda sin coincidencias debe abstenerse sin fuentes."""
    _, moviles = consultas_estructuradas
    moviles.return_value = []

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil de Centro?"
    )

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "sin_contexto"
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_movil_propaga_error_de_chroma_sin_llamar_gemini(
    consultas_estructuradas, dependencias,
):
    """Un error de lectura del índice se distingue de la abstención."""
    _, moviles = consultas_estructuradas
    moviles.side_effect = RuntimeError("Fallo de ChromaDB")

    with pytest.raises(RuntimeError, match="ChromaDB"):
        rag.responder("¿Qué horario tiene el punto limpio móvil de Centro?")

    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué horario tiene el punto limpio móvil de Barcelona?",
        "¿Qué horario tiene el punto limpio móvil de Bilbao?",
        "¿A qué hora pasa el punto limpio móvil de Málaga?",
        "¿Qué días visita el punto limpio móvil de Sevilla?",
        "¿Qué horario tiene el punto limpio móvil del distrito de Barcelona?",
        "¿Qué horario tiene el punto limpio móvil en la ciudad de Barcelona?",
        "¿Qué horario tiene el punto limpio móvil para Toledo?",
        "¿Cuáles son los horarios de los puntos limpios móviles de Málaga?",
        "¿Qué horario tiene el punto limpio móvil de VALENCIA?",
        "¿Qué horario tiene el punto limpio móvil de Zaragoza?",
    ],
)
def test_horario_movil_de_otra_ciudad_no_consulta_madrid(
    pregunta, consultas_estructuradas, dependencias,
):
    """Rechaza ciudades conocidas en distintas formulaciones."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["fuentes"] == []
    assert resultado["chunks"] == []
    assert resultado["metricas"]["chunks_recuperados"] == 0
    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_horario_movil_en_otra_ciudad_abstiene_sin_acceso_externo(
    consultas_estructuradas, dependencias,
):
    """El caso inequívoco 'en Barcelona' queda fuera del dominio."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil en Barcelona?"
    )

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["fuentes"] == []
    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué horario tiene el punto limpio móvil en Madrid?",
        "¿Qué horario tiene el punto limpio móvil de Madrid?",
    ],
)
def test_horario_movil_en_madrid_sin_parada_pide_aclaracion(
    pregunta, consultas_estructuradas, dependencias,
):
    """Una mención a Madrid no debe descartarse como externa."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "ambiguedad"
    assert "distrito o calle" in resultado["respuesta"]
    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_calle_con_nombre_de_ciudad_no_es_fuera_de_dominio(
    consultas_estructuradas, dependencias,
):
    """Una calle homónima de una ciudad no es ubicación externa."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(
        "¿Qué horario tiene el punto limpio móvil de la calle Barcelona?"
    )

    assert resultado["motivo"] is None
    moviles.assert_called_once_with(
        distrito=None, direccion="calle barcelona", dia=None
    )
    assert resultado["fuentes"]
    catalogo.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué tipos de residuos hay en Bilbao?",
        "¿Qué categorías de residuos hay en Valencia?",
        "¿Qué tipos de residuos hay para Toledo?",
    ],
)
def test_catalogo_externo_no_usa_filas_de_madrid(
    pregunta, consultas_estructuradas, dependencias,
):
    """La ruta agregada no cita el catálogo de otra ciudad."""
    catalogo, moviles = consultas_estructuradas

    resultado = rag.responder(pregunta)

    assert resultado["abstencion"] is True
    assert resultado["motivo"] == "fuera_de_dominio"
    assert resultado["fuentes"] == []
    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué horario tiene el punto limpio móvil 24h en Centro?",
        "¿Qué es un punto limpio móvil?",
        "¿Qué horario tiene el punto limpio fijo de Centro?",
    ],
)
def test_otras_modalidades_y_preguntas_conceptuales_son_semanticas(
    pregunta, consultas_estructuradas, dependencias,
):
    """La ruta móvil ordinaria no intercepta consultas distintas."""
    catalogo, moviles = consultas_estructuradas

    rag.responder(pregunta)

    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_called_once()
    dependencias[1].assert_called_once()


@pytest.mark.parametrize("pregunta", ["", "   ", None, 123, "a" * 501])
def test_valida_preguntas_antes_de_consultar_dependencias(
    pregunta, consultas_estructuradas, dependencias,
):
    """No se elude la validación por las nuevas rutas."""
    catalogo, moviles = consultas_estructuradas

    with pytest.raises(ValueError, match="pregunta"):
        rag.responder(pregunta)

    catalogo.assert_not_called()
    moviles.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()


def test_consultas_estructuradas_mantienen_metricas_y_no_usan_cliente(
    consultas_estructuradas, dependencias,
):
    """La vía estructurada conserva firma y telemetría pública."""
    cliente = Mock()
    vector = [0.1] * 768

    resultado = rag.responder(
        "¿Qué tipos de residuos hay?", client=cliente,
        query_embedding=vector,
    )

    assert set(resultado) == {
        "pregunta", "respuesta", "abstencion", "motivo",
        "fuentes", "chunks", "metricas",
    }
    assert resultado["metricas"]["top_k"] == TOP_K
    for campo in ("retrieval_ms", "generacion_ms", "total_ms"):
        assert isinstance(resultado["metricas"][campo], float)
        assert resultado["metricas"][campo] >= 0
    cliente.models.generate_content.assert_not_called()
    cliente.close.assert_not_called()
    dependencias[0].assert_not_called()
    dependencias[1].assert_not_called()
