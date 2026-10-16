# Localiza archivos
# Abre archivos
# interpreta CSV como DataFrame
# conserva HTML originales como texto

from pathlib import Path

import pandas as pd

from config import BASE_DIR, CSV_DIR, HTML_DIR


# =========================================================
# UTILIDADES
# =========================================================

def _leer_texto_con_fallback(path: Path) -> tuple[str, str]:
    """
    Lee un archivo de texto probando varias codificaciones.

    Devuelve:
        - contenido original del archivo
        - codificación utilizada
    """

    codificaciones = ("utf-8-sig", "utf-8", "latin-1")

    ultimo_error = None

    for encoding in codificaciones:
        try:
            contenido = path.read_text(encoding=encoding)

            return contenido, encoding

        except UnicodeDecodeError as error:
            ultimo_error = error

    raise ValueError(
        f"No se pudo leer el archivo '{path.name}'. "
        f"Último error: {ultimo_error}"
    )


# =========================================================
# CARGA DE CSV
# =========================================================

def cargar_csv(path: Path) -> dict:
    """
    Carga un CSV manteniendo su estructura tabular original.

    Todos los CSV del corpus utilizan ';' como separador.

    Las columnas se cargan como texto para conservar
    valores como códigos con ceros iniciales.
    """

    codificaciones = ("utf-8-sig", "latin-1")

    ultimo_error = None

    for encoding in codificaciones:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )

            return {
                "source": path.name,
                "path": str(path.relative_to(BASE_DIR)),
                "format": "csv",
                "encoding": encoding,
                "content": df
            }

        except UnicodeDecodeError as error:
            ultimo_error = error

        except pd.errors.ParserError as error:
            ultimo_error = error

    raise ValueError(
        f"No se pudo cargar el CSV '{path.name}'. "
        f"Último error: {ultimo_error}"
    )


# =========================================================
# CARGA DE HTML
# =========================================================

def cargar_html(path: Path) -> dict:
    """
    Carga el código HTML original.

    La extracción y limpieza del contenido útil
    se realizará posteriormente en clean.py.
    """

    contenido, encoding = _leer_texto_con_fallback(path)

    return {
        "source": path.name,
        "path": str(path.relative_to(BASE_DIR)),
        "format": "html",
        "encoding": encoding,
        "content": contenido
    }


# =========================================================
# CARGA DEL CORPUS
# =========================================================

def cargar_corpus() -> list[dict]:
    """
    Carga todos los CSV y HTML existentes en data/raw/.
    """

    documentos = []

    for path in sorted(CSV_DIR.glob("*.csv")):
        documentos.append(cargar_csv(path))

    for path in sorted(HTML_DIR.glob("*.html")):
        documentos.append(cargar_html(path))

    return documentos