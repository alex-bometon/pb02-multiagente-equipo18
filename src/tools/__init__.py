"""Allowlist única y dispatcher seguro para el futuro grafo multiagente."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.tools import tool

from .api_puntos_limpios import API_consultar_puntos_limpios_fijos
from .describir_imagen import describir_imagen
from .hora_actual import hora_actual
from .rag_ask import rag_ask

# El decorador usa las firmas y docstrings de las funciones existentes;
# no se crean clientes ni se realiza trabajo externo al importar este paquete.
TOOLS = [
    tool(rag_ask),
    tool(API_consultar_puntos_limpios_fijos),
    tool(hora_actual),
    tool(describir_imagen),
]
TOOL_BY_NAME = {item.name: item for item in TOOLS}
_FUNCIONES = {
    fun.__name__: fun for fun in (
        rag_ask, API_consultar_puntos_limpios_fijos, hora_actual, describir_imagen
    )
}


@dataclass
class PresupuestoTools:
    """Se comparte entre TODAS las tool calls de un ciclo (nunca global)."""
    max_operaciones: int
    consumidas: int = 0
    traza: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.max_operaciones) is not int or self.max_operaciones < 1:
            raise ValueError("max_operaciones debe ser entero positivo")


def _validar_argumentos(nombre: str, args: dict[str, Any]) -> None:
    firma = inspect.signature(_FUNCIONES[nombre])
    if not isinstance(args, dict) or any(not isinstance(k, str) for k in args):
        raise ValueError("Los argumentos deben ser un objeto JSON")
    if set(args) - set(firma.parameters):
        raise ValueError("Argumentos no reconocidos")
    for parametro in firma.parameters.values():
        if parametro.name not in args:
            if parametro.default is inspect.Parameter.empty:
                raise ValueError("Falta un argumento obligatorio")
            continue
        valor = args[parametro.name]
        if parametro.annotation in (str, "str") and not isinstance(valor, str):
            raise ValueError("Argumento de texto incorrecto")
        if parametro.annotation in (int, "int") and type(valor) is not int:
            raise ValueError("Argumento entero incorrecto")
    # La semántica exacta de cada argumento se valida dentro de su tool.


def ejecutar_tool(
    nombre: str,
    args: dict[str, Any] | None,
    presupuesto: PresupuestoTools,
) -> dict[str, Any]:
    """Restringe nombres/argumentos/operaciones antes de ejecutar la tool."""
    inicio = perf_counter()
    argumentos = args if isinstance(args, dict) else {}
    traza: dict[str, Any] = {
        "nombre": nombre if isinstance(nombre, str) else "(no válido)",
        "estado": "rechazado",
        "duracion_ms": 0.0,
        "argumentos_seguros": sorted(str(k) for k in argumentos)[:12],
        "origen": None,
        "error": None,
    }
    salida = {"ok": False, "resultado": None, "error": None}
    try:
        if not isinstance(presupuesto, PresupuestoTools):
            raise ValueError("Presupuesto inválido")
        if nombre not in TOOL_BY_NAME:
            raise ValueError("Herramienta no autorizada")
        if args is None:
            argumentos = {}
        elif not isinstance(args, dict):
            raise ValueError("Argumentos no válidos")
        _validar_argumentos(nombre, argumentos)
        if presupuesto.consumidas >= presupuesto.max_operaciones:
            raise ValueError("Límite de operaciones agotado")
        presupuesto.consumidas += 1  # Reserva ANTES de ejecutar.
        traza["estado"] = "error"
        # Invocación Python directa: firmas ya validadas, sin coerción de Pydantic.
        resultado = _FUNCIONES[nombre](**argumentos)
        traza["estado"] = "ok"
        traza["origen"] = resultado.get("origen") if isinstance(resultado, dict) else None
        salida.update(ok=True, resultado=resultado)
    except Exception as error:  # frontera de seguridad del dispatcher
        # Jamás adjuntar el texto de excepción: puede contener credenciales.
        codigo = "rechazado" if traza["estado"] == "rechazado" else "error"
        traza["estado"] = codigo
        traza["error"] = type(error).__name__
        salida["error"] = (
            "La herramienta no pudo ejecutarse." if codigo == "error" else
            "La operación fue rechazada por la validación o el presupuesto."
        )
    finally:
        traza["duracion_ms"] = round((perf_counter() - inicio) * 1000, 2)
        if isinstance(presupuesto, PresupuestoTools):
            presupuesto.traza.append(traza)
    salida["traza"] = traza
    return salida
