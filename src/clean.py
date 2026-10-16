# Funciones para la transformación necesaria previa a chunks
# Transforma HTML a str con código HTML
# Transforma CSV a DataFrame

import re

from bs4 import BeautifulSoup


# =========================================================
# UTILIDADES GENERALES
# =========================================================

def _limpiar_texto(valor) -> str:
    """
    Normaliza un valor textual sin modificar su significado.

    - Convierte el valor a texto.
    - Sustituye espacios no separables.
    - Elimina espacios, tabulaciones y saltos repetidos.
    - Convierte valores vacíos habituales en cadena vacía.
    """

    if valor is None:
        return ""

    texto = str(valor)

    # Espacio especial frecuente en datos HTML y CSV.
    texto = texto.replace("\xa0", " ")

    # Unifica espacios, tabulaciones y saltos de línea.
    texto = re.sub(r"\s+", " ", texto).strip()

    # Valores utilizados para representar ausencia de datos.
    if texto.upper() in {"", "NULL", "NAN", "NONE"}:
        return ""

    return texto


def _a_float(valor):
    """
    Convierte una coordenada textual a float.

    Los CSV del Ayuntamiento utilizan tanto:
        40,39283578
    como:
        40.39419901357015

    Por eso sustituimos la coma decimal por punto antes
    de realizar la conversión.

    Si no puede convertirse, devuelve None.
    """

    texto = _limpiar_texto(valor)

    if not texto:
        return None

    texto = texto.replace(",", ".")

    try:
        return float(texto)

    except ValueError:
        return None


def _crear_documento(
    texto: str,
    documento: dict,
    tipo: str,
    **metadata,
) -> dict:
    """
    Crea la estructura documental común que utilizarán
    posteriormente chunk.py, embed.py e index.py.

    La estructura resultante es:

    {
        "text": "...",
        "metadata": {
            ...
        }
    }
    """

    metadata_final = {
        "source": documento["source"],
        "path": documento["path"],
        "format": documento["format"],
        "document_type": tipo,
    }

    # Solo incorporamos metadatos que tengan valor.
    # Esto evita guardar None o cadenas vacías.
    for clave, valor in metadata.items():
        if valor is not None and valor != "":
            metadata_final[clave] = valor

    return {
        "text": texto.strip(),
        "metadata": metadata_final,
    }


# =========================================================
# LIMPIEZA DE HTML
# =========================================================

def _obtener_titulo_html(soup, main, source: str) -> str:
    """
    Obtiene el título principal del documento HTML.
    """

    titulo_elemento = main.find(["h1", "h2", "h3"])

    if titulo_elemento is not None:
        return _limpiar_texto(" ".join(titulo_elemento.stripped_strings))

    if soup.title is not None:
        titulo = _limpiar_texto(soup.title.get_text())

        # Eliminamos el sufijo común del portal.
        titulo = titulo.replace(" - Ayuntamiento de Madrid", "")

        return titulo

    return source


def _extraer_texto_html(contenedor) -> str:
    """
    Extrae texto semánticamente útil del contenido HTML.

    Conservamos:
        - títulos
        - subtítulos
        - párrafos
        - elementos de listas

    Eliminamos elementos propios de la interfaz y evitamos
    duplicados provocados por listas que contienen párrafos.
    """

    # Elementos que nunca aportan información útil al RAG.
    for etiqueta in contenedor.find_all(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "button",
        ]
    ):
        etiqueta.decompose()

    etiquetas_texto = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]

    lineas = []

    for elemento in contenedor.find_all(etiquetas_texto):

        # Algunos <li> contienen otros bloques de texto.
        # Si los procesáramos completos y luego también sus
        # hijos, introduciríamos texto duplicado.
        if elemento.name == "li":

            # Lista dentro de otra lista.
            if elemento.find("li") is not None:
                continue

            # Si contiene párrafos con texto real,
            # dejamos que esos <p> se procesen por separado.
            parrafos = [
                parrafo for parrafo in elemento.find_all("p")
                if _limpiar_texto(" ".join(parrafo.stripped_strings))
            ]

            if parrafos:
                continue

        texto = _limpiar_texto(" ".join(elemento.stripped_strings))

        if not texto:
            continue

        # Elementos propios de la interfaz del portal.
        if texto in {"Volver", "Escuchar", "Ver listado", "Mapa"}:
            continue

        if texto.lower().startswith("total:"):
            continue

        # Evitamos repeticiones consecutivas.
        if lineas and texto == lineas[-1]:
            continue

        lineas.append(texto)

    return "\n".join(lineas)


def limpiar_html(documento: dict) -> list[dict]:
    """
    Limpia un documento HTML del Ayuntamiento de Madrid.

    Busca el contenido principal dentro de:

        <main id="readspeaker">

    En la mayoría de documentos existe además un bloque:

        <div class="detalle">

    que contiene la información específica de la página.
    """

    soup = BeautifulSoup(documento["content"], "html.parser")

    # Contenedor principal común a nuestros HTML.
    main = soup.find("main", id="readspeaker")

    if main is None:
        raise ValueError(
            f"No se encontró <main id='readspeaker'> "
            f"en '{documento['source']}'."
        )

    titulo = _obtener_titulo_html(soup, main, documento["source"])

    # En la mayoría de páginas el contenido útil está
    # dentro del bloque .detalle.
    detalle = main.find("div", class_="detalle")

    if detalle is not None:
        contenedor = detalle

    else:
        # recogida_muebles_enseres.html tiene una estructura
        # distinta y no utiliza .detalle.
        #
        # En ese caso seleccionamos el último bloque .nofluid,
        # que contiene el cuerpo de la página.
        bloques_nofluid = main.find_all("div", class_="nofluid")

        if bloques_nofluid:
            contenedor = bloques_nofluid[-1]
        else:
            contenedor = main

    contenido = _extraer_texto_html(contenedor)

    if not contenido:
        raise ValueError(
            f"No se pudo extraer texto útil de '{documento['source']}'."
        )

    # Algunas páginas incluyen el título dentro del bloque
    # extraído y otras no.
    #
    # Lo añadimos únicamente si todavía no está al comienzo.
    if not contenido.casefold().startswith(titulo.casefold()):
        contenido = f"{titulo}\n{contenido}"

    return [
        _crear_documento(
            contenido,
            documento,
            "documentacion_municipal",
            title=titulo,
        )
    ]


# =========================================================
# CSV: TIPOS DE RESIDUOS
# =========================================================

def limpiar_tipos_residuos(documento: dict) -> list[dict]:
    """
    Transforma tipos_residuos.csv.

    Este CSV tiene una estructura especial:

        columna -> lugar de depósito
        celda   -> residuo

    Por tanto, cada celda no vacía se convierte en una
    relación independiente residuo -> destino.
    """

    df = documento["content"]

    documentos = []

    for columna in df.columns:

        destino = _limpiar_texto(columna)

        for indice, valor in df[columna].items():

            residuo = _limpiar_texto(valor)

            if not residuo:
                continue

            texto = f"Residuo: {residuo}\nLugar de depósito: {destino}"

            documentos.append(
                _crear_documento(
                    texto,
                    documento,
                    "residuo_destino",
                    destino=destino,

                    # +2 porque:
                    # fila 1 = cabecera CSV
                    # índice 0 de pandas = fila 2 del CSV
                    source_row=int(indice) + 2,
                )
            )

    return documentos


# =========================================================
# CSV: CONTENEDORES DE ACEITE
# =========================================================

def limpiar_contenedores_aceite(documento: dict) -> list[dict]:
    """
    Convierte cada contenedor de aceite vegetal
    en un documento independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        direccion = _limpiar_texto(fila["DIRECCIÓN COMPLETA AMPLIADA"])

        if not direccion:
            direccion = _limpiar_texto(fila["DIRECCION COMPLETA"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        ubicacion = _limpiar_texto(fila["TIPO  SITUADO"])

        partes = [
            "Contenedor de aceite vegetal usado.",
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
        ]

        if ubicacion:
            partes.append(f"Tipo de ubicación: {ubicacion}.")

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "contenedor_aceite",
                district=distrito,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: CONTENEDORES ORDINARIOS --> No estará en la versión V1 pero se mantiene para futuro
# =========================================================

def limpiar_contenedores_ordinarios(documento: dict,) -> list[dict]:
    """
    Convierte los contenedores ordinarios de Madrid
    en documentos independientes.

    Aunque el archivo se llama
    contenedores_papel_carton_todos.csv,
    contiene diferentes tipos de contenedores:
        - Papel-Cartón
        - Vidrio
        - Orgánica
        - Resto
        - PMB
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        tipo_original = _limpiar_texto(fila["Tipo Contenedor"])

        # Existen dos filas sin tipo de contenedor.
        if not tipo_original:
            continue

        # Desarrollamos la abreviatura para mejorar
        # la recuperación semántica.
        if tipo_original == "PMB":
            tipo_texto = "Plásticos, metales y briks (PMB)"
        else:
            tipo_texto = tipo_original

        direccion = _limpiar_texto(fila["Dirección"])

        distrito = _limpiar_texto(fila["Distrito"])

        barrio = _limpiar_texto(fila["Barrio"])

        partes = [
            f"Contenedor de tipo: {tipo_texto}.",
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
            f"Barrio: {barrio}.",
        ]

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "contenedor_ordinario",
                container_type=tipo_original,
                district=distrito,
                neighborhood=barrio,
                latitude=_a_float(fila["Latitud"]),
                longitude=_a_float(fila["Longitud"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: CONTENEDORES DE PILAS
# =========================================================

def limpiar_contenedores_pilas(documento: dict,) -> list[dict]:
    """
    Convierte las marquesinas que disponen de
    contenedor de pilas en documentos independientes.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        tiene_pilas = _limpiar_texto(fila["CONT PILAS"])

        if tiene_pilas.upper() != "SI":
            continue

        direccion = _limpiar_texto(fila["Direccion_completa"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        texto = (
            "Contenedor de pilas en marquesina de autobús.\n"
            f"Dirección: {direccion}.\n"
            f"Distrito: {distrito}."
        )

        documentos.append(
            _crear_documento(
                texto,
                documento,
                "contenedor_pilas",
                district=distrito,
                latitude=_a_float(fila["Latitud"]),
                longitude=_a_float(fila["Longitud"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: CONTENEDORES DE ROPA
# =========================================================

def limpiar_contenedores_ropa(documento: dict,) -> list[dict]:
    """
    Convierte cada contenedor de ropa y residuos
    textiles en un documento independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        direccion = _limpiar_texto(fila["DIRECCIÓN COMPLETA AMPLIADA"])

        if not direccion:
            direccion = _limpiar_texto(fila["DIRECCION COMPLETA"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        barrio = _limpiar_texto(fila["BARRIO"])

        texto = (
            "Contenedor de ropa y residuos textiles.\n"
            f"Dirección: {direccion}.\n"
            f"Distrito: {distrito}.\n"
            f"Barrio: {barrio}."
        )

        documentos.append(
            _crear_documento(
                texto,
                documento,
                "contenedor_ropa",
                district=distrito,
                neighborhood=barrio,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: PUNTOS LIMPIOS FIJOS
# =========================================================

def limpiar_puntos_limpios_fijos(documento: dict,) -> list[dict]:
    """
    Convierte cada punto limpio fijo en un documento
    independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        nombre = _limpiar_texto(fila["NOMBRE"])

        direccion = " ".join(
            parte for parte in [
                _limpiar_texto(fila["CLASE-VIAL"]),
                _limpiar_texto(fila["NOMBRE-VIA"]),
                _limpiar_texto(fila["NUM"]),
            ]
            if parte
        )

        distrito = _limpiar_texto(fila["DISTRITO"])

        barrio = _limpiar_texto(fila["BARRIO"])

        horario = _limpiar_texto(fila["HORARIO"])

        transporte = _limpiar_texto(fila["TRANSPORTE"])

        descripcion = _limpiar_texto(fila["DESCRIPCION"])

        partes = [
            nombre,
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
            f"Barrio: {barrio}.",
        ]

        if horario:
            partes.append(f"Horario: {horario}")

        if transporte:
            partes.append(f"Transporte público: {transporte}")

        if descripcion:
            partes.append(descripcion)

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "punto_limpio_fijo",
                district=distrito,
                neighborhood=barrio,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: PUNTOS LIMPIOS MÓVILES
# =========================================================

def limpiar_puntos_limpios_moviles(documento: dict,) -> list[dict]:
    """
    Convierte cada parada y horario de punto limpio
    móvil en un documento independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        direccion = _limpiar_texto(fila["DIRECCIÓN_COMPLETA"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        dia = _limpiar_texto(fila["DÍA_SEMANA"])

        turno = _limpiar_texto(fila["TURNO"])

        hora_inicio = _limpiar_texto(fila["HORA_INICIO"])

        hora_final = _limpiar_texto(fila["HORA_FINAL"])

        partes = [
            "Punto limpio móvil.",
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
            f"Día: {dia}.",
        ]

        if turno:
            partes.append(f"Turno: {turno}.")

        if hora_inicio and hora_final:
            partes.append(f"Horario: {hora_inicio} - {hora_final}.")

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "punto_limpio_movil",
                district=distrito,
                day=dia,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: PUNTOS LIMPIOS MÓVILES 24 HORAS
# =========================================================

def limpiar_puntos_limpios_moviles_24h(documento: dict,) -> list[dict]:
    """
    Convierte cada ubicación de punto limpio móvil
    24 horas en un documento independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        direccion = _limpiar_texto(fila["DIRECCIÓN COMPLETA AMPLIADA"])

        if not direccion:
            direccion = _limpiar_texto(fila["DIRECCIÓN COMPLETA"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        barrio = _limpiar_texto(fila["BARRIO"])

        dia = _limpiar_texto(fila["DÍA_SEMANA"])

        ubicacion = _limpiar_texto(fila["UBICACIÓN"])

        partes = [
            "Punto limpio móvil 24 horas.",
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
            f"Barrio: {barrio}.",
            f"Día de ubicación: {dia}.",
        ]

        if ubicacion:
            partes.append(f"Ubicación: {ubicacion}.")

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "punto_limpio_movil_24h",
                district=distrito,
                neighborhood=barrio,
                day=dia,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# CSV: PUNTOS LIMPIOS DE PROXIMIDAD
# =========================================================

def limpiar_puntos_limpios_proximidad(documento: dict,) -> list[dict]:
    """
    Convierte cada punto limpio de proximidad
    en un documento independiente.
    """

    df = documento["content"]

    documentos = []

    for indice, fila in df.iterrows():

        direccion = _limpiar_texto(fila["DIRECCIÓN COMPLETA AMPLIADA"])

        if not direccion:
            direccion = _limpiar_texto(fila["DIRECCION_COMPLETA"])

        distrito = _limpiar_texto(fila["DISTRITO"])

        centro = _limpiar_texto(fila["CENTRO"])

        horario = _limpiar_texto(fila["HORARIO"])

        partes = [
            "Punto limpio de proximidad.",
            f"Dirección: {direccion}.",
            f"Distrito: {distrito}.",
        ]

        if centro:
            partes.append(f"Centro: {centro}.")

        if horario:
            partes.append(f"Horario: {horario}")

        documentos.append(
            _crear_documento(
                "\n".join(partes),
                documento,
                "punto_limpio_proximidad",
                district=distrito,
                latitude=_a_float(fila["LATITUD"]),
                longitude=_a_float(fila["LONGITUD"]),
                source_row=int(indice) + 2,
            )
        )

    return documentos


# =========================================================
# ASIGNACIÓN DE LIMPIADOR SEGÚN CSV
# =========================================================

LIMPIADORES_CSV = {
    "tipos_residuos.csv": limpiar_tipos_residuos,
    "contenedores_aceitevegetal_usado.csv": limpiar_contenedores_aceite,
    # "contenedores_papel_carton_todos.csv": limpiar_contenedores_ordinarios,
    "contenedores_pilas_marquesinas.csv": limpiar_contenedores_pilas,
    "contenedores_ropa.csv": limpiar_contenedores_ropa,
    "puntos_limpios_fijos.csv": limpiar_puntos_limpios_fijos,
    "puntos_limpios_moviles.csv": limpiar_puntos_limpios_moviles,
    "puntos_limpios_moviles_24h.csv": limpiar_puntos_limpios_moviles_24h,
    "puntos_limpios_proximidad.csv": limpiar_puntos_limpios_proximidad,
}


def limpiar_csv(documento: dict,) -> list[dict]:
    """
    Selecciona el limpiador correspondiente según
    el nombre del CSV.
    """

    source = documento["source"]

    limpiador = LIMPIADORES_CSV.get(source)

    if limpiador is None:
        raise ValueError(f"No existe un limpiador definido para '{source}'.")

    return limpiador(documento)


# =========================================================
# LIMPIEZA COMPLETA DEL CORPUS
# =========================================================

def limpiar_corpus(corpus: list[dict],) -> list[dict]:
    """
    Normaliza todos los documentos cargados por load.py.

    Entrada:
        CSV  -> DataFrame
        HTML -> HTML original como str

    Salida:
        lista de documentos con estructura común:

        {
            "text": "...",
            "metadata": {...}
        }
    """

    documentos_limpios = []

    for documento in corpus:

        formato = documento["format"]

        if formato == "csv":
            documentos_limpios.extend(limpiar_csv(documento))
        elif formato == "html":
            documentos_limpios.extend(limpiar_html(documento))
        else:
            raise ValueError(f"Formato no soportado: '{formato}'.")

    return documentos_limpios