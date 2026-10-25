"""Fixtures compartidas de las pruebas del orquestador RAG."""

from unittest.mock import Mock

import pytest

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
