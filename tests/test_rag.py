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

# ---------------------------------------------------------
# CONSULTAS ESTRUCTURADAS: CATÁLOGO Y PUNTOS LIMPIOS MÓVILES
# ---------------------------------------------------------


@pytest.fixture
def catalogo_estructurado():
    """Proporciona el contrato agregado de consultar_catalogo_residuos."""
    return {
        "source": "tipos_residuos.csv",
        "path": "data/raw/csv/tipos_residuos.csv",
        "total_relaciones": 5,
        "total_destinos": 2,
        "destinos": [
            {
                "destino": "CONTENEDOR AMARILLO",
                "cantidad": 3,
                "ejemplos": [{"residuo": "Envases de plástico"}],
            },
            {
                "destino": "CONTENEDOR AZUL",
                "cantidad": 2,
                "ejemplos": [{"residuo": "Papel"}],
            },
        ],
        "registros_descartados": 0,
        "registros_duplicados": 0,
    }


@pytest.fixture
def paradas_estructuradas():
    """Representa dos paradas móviles ordinarias documentadas."""
    fuente = "puntos_limpios_moviles.csv"
    ruta = "data/raw/csv/puntos_limpios_moviles.csv"
    entradas = [
        ("CALLE ARRIETA, 15", "MARTES", "12:15 - 14:15", 2),
        ("CALLE BARQUILLO, 7", "SABADO", "07:45 - 09:45", 3),
    ]
    paradas = []
    for direccion, dia, horario, fila in entradas:
        texto = (
            "Punto limpio móvil.\n"
            f"Dirección: {direccion}.\n"
            "Distrito: CENTRO.\n"
            f"Día: {dia}.\n"
            f"Horario: {horario}."
        )
        metadata = {
            "source": fuente,
            "path": ruta,
            "document_type": "punto_limpio_movil",
            "district": "CENTRO",
            "day": dia,
            "source_row": fila,
        }
        paradas.append({
            "direccion": direccion,
            "distrito": "CENTRO",
            "dia": dia,
            "turno": "MAÑANA",
            "horario": horario,
            "source": fuente,
            "path": ruta,
            "text": texto,
            "metadata": metadata,
            "source_row": fila,
        })
    return paradas


@pytest.fixture
def consultas_estructuradas(
    monkeypatch, dependencias, catalogo_estructurado,
    paradas_estructuradas,
):
    """Simula lectores estructurados y deja vigilada la vía semántica."""
    catalogo = Mock(return_value=catalogo_estructurado)
    moviles = Mock(return_value=paradas_estructuradas)
    monkeypatch.setattr(rag, "consultar_catalogo_residuos", catalogo)
    monkeypatch.setattr(rag, "consultar_paradas_moviles", moviles)
    return catalogo, moviles


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
