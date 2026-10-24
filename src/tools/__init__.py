
"""Allowlist de herramientas de ReciclaTIA."""

from typing import Any, Literal, TypedDict

from config import AGENT_TOOLS_BY_ROLE

from .api_puntos_limpios import buscar_puntos_limpios_cercanos
from .describir_imagen import describir_imagen as _describir_imagen
from .hora_actual import hora_actual as _hora_actual
from .rag_tool import consultar_rag


# Herramientas autorizadas para el agente.
TOOLS = [
    consultar_rag,
    buscar_puntos_limpios_cercanos,
    _hora_actual,
    _describir_imagen,
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}


def ejecutar_tool(nombre: str, args: dict | None = None) -> str:
    """Ejecuta únicamente herramientas de la allowlist."""

    if not isinstance(nombre, str) or nombre not in TOOL_BY_NAME:
        return "Error: herramienta no permitida."

    if args is None:
        args = {}

    if not isinstance(args, dict):
        return "Error: argumentos inválidos."

    herramienta = TOOL_BY_NAME[nombre]

    if (
        any(not isinstance(clave, str) for clave in args)
        or set(args) - set(herramienta.args)
    ):
        return "Error: argumentos no permitidos."

    try:
        return str(herramienta.invoke(args))
    except Exception:
        return "Error: no se pudo ejecutar la herramienta."
