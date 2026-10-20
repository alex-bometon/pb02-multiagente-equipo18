"""Contrato, abstenciones y validación de JSON de Gemini.

Las fuentes citadas solo pueden proceder del mapa de referencias enviado.
"""

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


class SalidaGeneracionInvalida(RuntimeError):
    """Gemini respondió, pero su salida no supera la validación."""


MENSAJE_ABSTENCION = (
    "No puedo responder con la información disponible en los documentos."
)

MOTIVOS_ABSTENCION = {
    "sin_contexto",
    "evidencia_insuficiente",
    "fuera_de_dominio",
    "ambiguedad",
}


def _abstenerse(motivo: str, respuesta: str = "") -> dict:
    """Devuelve una abstención sin atribuir fuentes no utilizadas."""
    if motivo == "ambiguedad" and respuesta.strip():
        texto = respuesta.strip()
    else:
        texto = MENSAJE_ABSTENCION

    return {
        "respuesta": texto,
        "abstencion": True,
        "motivo": motivo,
        "fuentes": [],
    }


def _descartar_salida(causa: str) -> None:
    """Señala un fallo estructural sin divulgar el texto de Gemini."""
    logger.warning("Salida de Gemini rechazada: %s", causa)
    raise SalidaGeneracionInvalida(
        f"La salida de Gemini no cumple el contrato: {causa}."
    )


def _normalizar_citas_respuesta(respuesta: str) -> tuple[str, set[str]]:
    """Acepta citas separadas o agrupadas y comprueba su sintaxis.

    Convierte '[C1, C2]' en '[C1] [C2]' sin alterar el contenido
    argumentativo. No interpreta como citas las referencias sin
    corchetes ni los grupos con identificadores malformados.
    """
    def expandir(coincidencia: re.Match) -> str:
        contenido = coincidencia.group(1).strip()
        if not re.fullmatch(r"C\d+(?:\s*,\s*C\d+)*", contenido):
            return coincidencia.group(0)

        identificadores = re.findall(r"C\d+", contenido)
        return " ".join(f"[{identificador}]" for identificador in
                        identificadores)

    normalizada = re.sub(r"\[([^\[\]]*)\]", expandir, respuesta)
    menciones = set(re.findall(r"\[C\d+\]", normalizada))
    sin_citas = re.sub(r"\[C\d+\]", "", normalizada)

    if re.search(r"\bC\d+\b", sin_citas):
        raise ValueError("Identificadores de cita con formato inválido")

    return normalizada, menciones


def _interpretar_respuesta(resultado: Any, referencias: dict) -> dict:
    """Valida la salida del modelo y obtiene fuentes del contexto."""
    try:
        datos = json.loads(resultado.text)
    except (TypeError, ValueError, AttributeError):
        _descartar_salida("JSON ilegible")

    if not isinstance(datos, dict):
        _descartar_salida("estructura JSON incorrecta")

    respuesta = datos.get("respuesta")
    abstencion = datos.get("abstencion")
    motivo = datos.get("motivo")
    citas = datos.get("citas")

    if (
        not isinstance(respuesta, str)
        or not respuesta.strip()
        or type(abstencion) is not bool
        or not isinstance(motivo, str)
        or not isinstance(citas, list)
    ):
        _descartar_salida("tipos o campos de respuesta incorrectos")

    if abstencion:
        if motivo not in MOTIVOS_ABSTENCION or citas:
            _descartar_salida("abstención con motivo o citas inválidos")
        logger.info("Gemini solicita abstención: %s", motivo)
        return _abstenerse(motivo, respuesta)

    if motivo != "ninguno" or not citas:
        _descartar_salida("respuesta afirmativa sin motivo o citas válidas")

    if any(
        not isinstance(cita, str) or cita not in referencias
        for cita in citas
    ):
        _descartar_salida("citas ajenas al contexto recuperado")

    try:
        respuesta, menciones = _normalizar_citas_respuesta(respuesta)
    except ValueError:
        _descartar_salida("formato de marcadores de cita inválido")

    citadas = {f"[{cita}]" for cita in citas}
    if not menciones or menciones != citadas:
        _descartar_salida("citas del texto distintas del campo citas")

    fuentes = []
    visitadas = set()
    for cita in citas:
        if cita not in visitadas:
            fuentes.append(referencias[cita].copy())
            visitadas.add(cita)

    return {
        "respuesta": respuesta.strip(),
        "abstencion": False,
        "motivo": None,
        "fuentes": fuentes,
    }


