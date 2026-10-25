"""Pruebas unitarias de la herramienta multimodal de ReciclaTIA."""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.tools.describir_imagen import describir_imagen


@pytest.fixture
def imagen_prueba(tmp_path, monkeypatch):
    """Crea una imagen JPEG temporal dentro de assets/."""

    directorio = tmp_path / "assets"
    directorio.mkdir()

    ruta = directorio / "jamon.jpg"

    Image.new("RGB", (100, 100), "white").save(
        ruta, format="JPEG"
    )

    # Aislar los tests de los archivos reales del proyecto.
    monkeypatch.setattr(
        "src.tools.describir_imagen.BASE_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "src.tools.describir_imagen.ASSETS_DIR",
        directorio,
    )

    return ruta


def test_describir_imagen_registro():
    """Comprueba que LangChain reconoce la herramienta."""

    assert describir_imagen.name == "describir_imagen"
    assert describir_imagen.args["ruta_imagen"]["type"] == "string"


def test_describir_imagen_correcta(imagen_prueba):
    """Comprueba el envío de imagen y la respuesta de Gemini."""

    cliente = Mock()
    cliente.models.generate_content.return_value.text = (
        "Se observa una bandeja de plástico."
    )

    with patch(
        "src.tools.describir_imagen.crear_cliente_gemini",
        return_value=cliente,
    ) as mock_crear_cliente:

        resultado = describir_imagen.invoke({
            "ruta_imagen": "assets/jamon.jpg"
        })

    assert resultado == "Se observa una bandeja de plástico."

    mock_crear_cliente.assert_called_once_with()
    cliente.models.generate_content.assert_called_once()
    cliente.close.assert_called_once()

    # Comprobar que se enviaron los bytes de la imagen.
    argumentos = cliente.models.generate_content.call_args.kwargs
    contenido_imagen = argumentos["contents"][0]

    assert contenido_imagen.inline_data.mime_type == "image/jpeg"
    assert contenido_imagen.inline_data.data == imagen_prueba.read_bytes()


def test_describir_imagen_ruta_no_permitida(imagen_prueba):
    """Impide acceder a archivos fuera de assets/."""

    with patch(
        "src.tools.describir_imagen.crear_cliente_gemini"
    ) as mock_cliente:

        with pytest.raises(ValueError, match="dentro de assets"):
            describir_imagen.invoke({
                "ruta_imagen": "../archivo.jpeg"
            })

    mock_cliente.assert_not_called()


def test_describir_imagen_formato_invalido(imagen_prueba):
    """Rechaza archivos cuyo contenido no es JPEG ni PNG."""

    # Sobrescribir con contenido GIF, manteniendo el nombre .jpeg.
    Image.new("RGB", (100, 100), "white").save(
        imagen_prueba, format="GIF"
    )

    with patch(
        "src.tools.describir_imagen.crear_cliente_gemini"
    ) as mock_cliente:

        with pytest.raises(ValueError, match="contenido no es JPEG ni PNG"):
            describir_imagen.invoke({
                "ruta_imagen": "assets/jamon.jpg"
            })

    mock_cliente.assert_not_called()


def test_describir_imagen_respuesta_vacia(imagen_prueba):
    """Comprueba que Gemini no puede devolver una descripción vacía."""

    cliente = Mock()
    cliente.models.generate_content.return_value.text = ""

    with patch(
        "src.tools.describir_imagen.crear_cliente_gemini",
        return_value=cliente,
    ):

        with pytest.raises(RuntimeError, match="descripción"):
            describir_imagen.invoke({
                "ruta_imagen": "assets/jamon.jpg"
            })

    cliente.close.assert_called_once()
