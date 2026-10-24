
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


# =========================================================
# CONTRATO DE EJECUCIÓN
# =========================================================

RolEspecialista = Literal["residuos", "puntos"]


class ResultadoTool(TypedDict):
    """Resultado de una herramienta sin perder su estructura."""

    ok: bool
    nombre: str
    datos: Any
    error: str | None


# =========================================================
# RESULTADOS DE ERROR
# =========================================================

def _fallo(nombre: str, mensaje: str) -> ResultadoTool:
    """Construye un fallo sin exponer excepciones ni secretos."""

    return {
        "ok": False,
        "nombre": nombre,
        "datos": None,
        "error": mensaje,
    }


# =========================================================
# EJECUCIÓN CENTRALIZADA DE HERRAMIENTAS
# =========================================================

def _ejecutar(
    nombre: str,
    args: dict | None,
    *,
    rol: RolEspecialista | None = None,
) -> ResultadoTool:
    """Valida permisos y argumentos; ejecuta una sola vez."""

    # 1. Comprobar que la herramienta está registrada.
    if not isinstance(nombre, str) or nombre not in TOOL_BY_NAME:
        return _fallo(
            nombre if isinstance(nombre, str) else "",
            "Error: herramienta no permitida.",
        )

    # 2. Aplicar permisos adicionales cuando hay especialista.
    if rol is not None and nombre not in AGENT_TOOLS_BY_ROLE.get(
        rol, ()
    ):
        return _fallo(
            nombre,
            "Error: herramienta no permitida para el rol.",
        )

    # 3. Normalizar y validar los argumentos.
    if args is None:
        args = {}

    if not isinstance(args, dict):
        return _fallo(
            nombre,
            "Error: argumentos inválidos.",
        )

    herramienta = TOOL_BY_NAME[nombre]

    if (
        any(not isinstance(clave, str) for clave in args)
        or set(args) - set(herramienta.args)
    ):
        return _fallo(
            nombre,
            "Error: argumentos no permitidos.",
        )

    # 4. Ejecutar la herramienta autorizada.
    try:
        datos = herramienta.invoke(args)

    except Exception:
        return _fallo(
            nombre,
            "Error: no se pudo ejecutar la herramienta.",
        )

    # 5. Conservar el resultado original.
    return {
        "ok": True,
        "nombre": nombre,
        "datos": datos,
        "error": None,
    }


# =========================================================
# INTERFAZ ORIGINAL — COMPATIBILIDAD
# =========================================================

def ejecutar_tool(
    nombre: str,
    args: dict | None = None,
) -> str:
    """Mantiene la interfaz textual existente."""

    resultado = _ejecutar(nombre, args)

    if not resultado["ok"]:
        return (
            resultado["error"]
            or "Error: no se pudo ejecutar la herramienta."
        )

    return str(resultado["datos"])


# =========================================================
# INTERFAZ MULTIAGENTE — DATOS ESTRUCTURADOS
# =========================================================

def ejecutar_tool_estructurado(
    nombre: str,
    args: dict | None = None,
    *,
    rol: RolEspecialista,
) -> ResultadoTool:
    """Ejecuta una tool respetando los permisos del especialista."""

    # Un rol desconocido no puede ejecutar ninguna herramienta.
    if rol not in AGENT_TOOLS_BY_ROLE:
        return _fallo(
            nombre if isinstance(nombre, str) else "",
            "Error: rol no autorizado.",
        )

    return _ejecutar(
        nombre,
        args,
        rol=rol,
    )