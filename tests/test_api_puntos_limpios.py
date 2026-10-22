
"""Pruebas unitarias de la herramienta geográfica de ReciclaTIA."""

from unittest.mock import patch

import pytest

from src.tools.api_puntos_limpios import (
    buscar_puntos_limpios_cercanos,
)


LATITUD = 40.4168
LONGITUD = -3.7038


def _registro_api(identificador: int, latitud: float) -> dict:
    """Construye un registro ArcGIS simulado."""
    return {
        "attributes": {
            "OBJECTID_1": identificador,
            "TIPO_DATO": "Fijo",
            "Direccion": f"Calle de prueba {identificador}",
            "NOMBRE_DIS": "CENTRO",
        },
        "geometry": {
            "x": LONGITUD,
            "y": latitud,
        },
    }


def test_busqueda_api_correcta():
    """Comprueba ordenación, límite y procedencia de la API."""

    registros = [
        _registro_api(i, LATITUD + i * 0.0001)
        for i in range(12, 0, -1)
    ]

    with (
        patch(
            "src.tools.api_puntos_limpios._consultar_api_puntos_limpios",
            return_value=registros,
        ) as mock_api,
        patch(
            "src.tools.api_puntos_limpios.consultar_puntos_limpios_locales"
        ) as mock_local,
    ):
        resultado = buscar_puntos_limpios_cercanos.invoke({
            "latitud": LATITUD,
            "longitud": LONGITUD,
            "radio_metros": 2000,
            "tipo": "todos",
        })

    assert resultado["fuente"] == "api"
    assert resultado["total_encontrados"] == 12
    assert len(resultado["puntos"]) == 10

    distancias = [
        punto["distancia_metros"]
        for punto in resultado["puntos"]
    ]

    assert distancias == sorted(distancias)
    assert resultado["puntos"][0]["id"] == 1

    mock_api.assert_called_once()
    mock_local.assert_not_called()


def test_fallback_si_falla_api():
    """Comprueba el respaldo local ante un timeout."""

    registros_locales = [
        {
            "tipo": "Fijo",
            "direccion": "Punto cercano",
            "latitud": LATITUD + 0.001,
            "longitud": LONGITUD,
            "fuente": "corpus_local",
        },
        {
            "tipo": "Fijo",
            "direccion": "Punto lejano",
            "latitud": LATITUD + 0.05,
            "longitud": LONGITUD,
            "fuente": "corpus_local",
        },
    ]

    with (
        patch(
            "src.tools.api_puntos_limpios._consultar_api_puntos_limpios",
            side_effect=RuntimeError("Timeout simulado"),
        ),
        patch(
            "src.tools.api_puntos_limpios.consultar_puntos_limpios_locales",
            return_value=registros_locales,
        ) as mock_local,
    ):
        resultado = buscar_puntos_limpios_cercanos.invoke({
            "latitud": LATITUD,
            "longitud": LONGITUD,
            "radio_metros": 2000,
            "tipo": "fijo",
        })

    assert resultado["fuente"] == "corpus_local"
    assert resultado["total_encontrados"] == 1
    assert len(resultado["puntos"]) == 1
    assert resultado["puntos"][0]["direccion"] == "Punto cercano"
    assert resultado["aviso"] is not None

    mock_local.assert_called_once()


def test_api_sin_resultados():
    """Una consulta vacía válida no activa el fallback."""

    with (
        patch(
            "src.tools.api_puntos_limpios._consultar_api_puntos_limpios",
            return_value=[],
        ),
        patch(
            "src.tools.api_puntos_limpios.consultar_puntos_limpios_locales"
        ) as mock_local,
    ):
        resultado = buscar_puntos_limpios_cercanos.invoke({
            "latitud": LATITUD,
            "longitud": LONGITUD,
        })

    assert resultado["fuente"] == "api"
    assert resultado["total_encontrados"] == 0
    assert resultado["puntos"] == []

    mock_local.assert_not_called()


@pytest.mark.parametrize(
    "parametros",
    [
        {"latitud": 95, "longitud": LONGITUD},
        {"latitud": LATITUD, "longitud": 181},
        {
            "latitud": LATITUD,
            "longitud": LONGITUD,
            "radio_metros": 0,
        },
        {
            "latitud": LATITUD,
            "longitud": LONGITUD,
            "tipo": "desconocido",
        },
    ],
)
def test_parametros_invalidos(parametros):
    """Rechaza entradas inválidas antes de acceder a la red."""

    with (
        patch(
            "src.tools.api_puntos_limpios.requests.get"
        ) as mock_http,
        patch(
            "src.tools.api_puntos_limpios.consultar_puntos_limpios_locales"
        ) as mock_local,
    ):
        with pytest.raises(ValueError):
            buscar_puntos_limpios_cercanos.invoke(parametros)

    mock_http.assert_not_called()
    mock_local.assert_not_called()
