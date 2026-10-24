"""Enrutamiento y fallback determinista del supervisor de ReciclaTIA.

No utiliza LLM ni herramientas. La propuesta del modelo nunca decide
la aprobación humana, el estado `done` ni la ejecución de un nodo arbitrario.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, TypedDict

from config import MAX_ITER
from src.text_utils import normalizar_texto


# =========================================================
# CONTRATO DE ENRUTAMIENTO
# =========================================================

Ruta = Literal["residuos", "puntos", "sintetizar", "FINISH"]


class DecisionRuta(TypedDict):
    """Decisión verificada que consumirá el nodo supervisor."""

    necesita_residuos: bool
    necesita_puntos: bool
    siguiente: Ruta
    ruta_solicitada: str | None
    corregida: bool
    motivo_correccion: str | None


# =========================================================
# IDENTIFICACIÓN DE NECESIDADES
# =========================================================

# Patrones de intención, no parámetros operativos.
_PATRON_RESIDUOS = re.compile(
    r"\b(?:contenedores?|fraccion(?:es)?|clasific\w*|"
    r"recicl(?:ar|o|a|an|amos)|tir(?:ar|o|a|an)|"
    r"deposit(?:ar|o|a|an)|desech\w*|separar)\b"
    r"|\b(?:tipos?|categorias?|destinos?)\s+(?:de\s+)?"
    r"(?:los\s+)?residuos?\b"
)

_PATRON_PUNTOS = re.compile(
    r"\b(?:puntos?\s+limpios?|centros?\s+de\s+recogida|"
    r"instalaciones?\s+de\s+recogida|"
    r"paradas?\s+(?:de\s+)?recogida)\b"
)

_RUTAS = frozenset({
    "residuos",
    "puntos",
    "sintetizar",
    "FINISH",
})


def _necesidades_texto(texto: str) -> tuple[bool, bool]:
    """Detecta indicios explícitos sin inferir ubicaciones ni resultados."""

    normalizado = normalizar_texto(texto)

    return (
        bool(_PATRON_RESIDUOS.search(normalizado)),
        bool(_PATRON_PUNTOS.search(normalizado)),
    )


def _necesidades_turno(
    estado: Mapping[str, Any],
    propuesta: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    """Fija las necesidades al inicio y las conserva durante el turno."""

    # Después de la primera decisión no volvemos a clasificar
    # el pedido: los especialistas necesarios ya están definidos.
    if estado.get("siguiente"):
        return (
            estado.get("necesita_residuos") is True,
            estado.get("necesita_puntos") is True,
        )

    pedido = estado.get("pedido", "")

    residuos, puntos = _necesidades_texto(
        pedido if isinstance(pedido, str) else ""
    )

    # El contexto de revisión sirve para desambiguar un nuevo
    # mensaje que no expresa por sí mismo una intención clara.
    if not (residuos or puntos):
        contexto = estado.get("contexto_revision", "")

        if isinstance(contexto, str):
            residuos, puntos = _necesidades_texto(contexto)

    # Una imagen aportada requiere el especialista de residuos.
    if estado.get("ruta_imagen"):
        residuos = True

    # Cuando hay indicios explícitos, no añadimos especialistas
    # por una propuesta del modelo sin respaldo en el pedido.
    if residuos or puntos:
        return residuos, puntos

    # Para expresiones no reconocidas por las reglas anteriores,
    # aceptamos la clasificación estructurada del LLM.
    if isinstance(propuesta, Mapping):
        r = propuesta.get("necesita_residuos")
        p = propuesta.get("necesita_puntos")

        if type(r) is bool and type(p) is bool:
            return r, p

    # Si no puede identificarse ninguna necesidad, la síntesis
    # deberá solicitar aclaración sin ejecutar tools arbitrarias.
    return False, False


# =========================================================
# VALIDACIÓN DE LA SALIDA
# =========================================================

def _hay_salida_valida(estado: Mapping[str, Any]) -> bool:
    """Comprueba que ya existe respuesta y un plan sintetizado."""

    respuesta = estado.get("respuesta")
    plan = estado.get("plan")

    return (
        estado.get("siguiente") == "sintetizar"
        and isinstance(respuesta, str)
        and bool(respuesta.strip())
        and isinstance(plan, list)
        and len(plan) >= 2
        and all(
            isinstance(p, str) and p.strip()
            for p in plan
        )
    )


# =========================================================
# FALLBACK Y CONTROL DE PROGRESO
# =========================================================

def _ruta_obligatoria(
    estado: Mapping[str, Any],
    residuos: bool,
    puntos: bool,
) -> Ruta:
    """Impone orden, progreso, síntesis y límites desde Python."""

    iteracion = estado.get("iteracion", 0)

    if type(iteracion) is not int or iteracion < 0:
        raise ValueError("Contador de supervisor inválido.")

    salida = _hay_salida_valida(estado)

    pendiente_residuos = (
        residuos and not estado.get("residuos_completado")
    )

    pendiente_puntos = (
        puntos and not estado.get("puntos_completado")
    )

    # iteracion cuenta las decisiones anteriores del supervisor.
    if iteracion >= MAX_ITER:
        raise RuntimeError(
            "Se ha superado MAX_ITER en el supervisor."
        )

    # La última decisión permitida debe finalizar un plan válido.
    if iteracion == MAX_ITER - 1:
        if salida and not (
            pendiente_residuos or pendiente_puntos
        ):
            return "FINISH"

        raise RuntimeError(
            "Última decisión del supervisor sin plan completo."
        )

    # Solo podemos terminar tras sintetizar y atender
    # todos los especialistas requeridos.
    if salida and not (
        pendiente_residuos or pendiente_puntos
    ):
        return "FINISH"

    # Reservamos una decisión para sintetizar y otra para FINISH.
    # Si todavía falta un especialista, no lo omitimos.
    if iteracion == MAX_ITER - 2:
        if pendiente_residuos or pendiente_puntos:
            raise RuntimeError(
                "Límite del supervisor: "
                "quedan especialistas pendientes."
            )

        if estado.get("siguiente") == "sintetizar":
            raise RuntimeError(
                "No se pudo completar la síntesis."
            )

        return "sintetizar"

    # Orden de trabajo para las consultas combinadas.
    if pendiente_residuos:
        return "residuos"

    if pendiente_puntos:
        return "puntos"

    # Completados los especialistas, pasamos a síntesis.
    if not salida:
        if estado.get("siguiente") == "sintetizar":
            raise RuntimeError(
                "La síntesis no produjo un plan válido."
            )

        return "sintetizar"

    return "FINISH"


# =========================================================
# VALIDACIÓN DE LA PROPUESTA DEL SUPERVISOR
# =========================================================

def resolver_enrutamiento(
    estado: Mapping[str, Any],
    propuesta: Mapping[str, Any] | None = None,
) -> DecisionRuta:
    """Valida la sugerencia del supervisor y obtiene la ruta ejecutable.

    No modifica el estado.

    src.nodes.supervisor registrará la decisión, incrementará
    iteracion y actualizará los indicadores del estado.
    """

    residuos, puntos = _necesidades_turno(
        estado,
        propuesta,
    )

    ruta = _ruta_obligatoria(
        estado,
        residuos,
        puntos,
    )

    solicitada = None

    if isinstance(propuesta, Mapping):
        valor = propuesta.get("siguiente")

        if isinstance(valor, str) and valor in _RUTAS:
            solicitada = valor

    corregida = solicitada != ruta

    if propuesta is None:
        motivo = (
            "Fallback: el supervisor no ha devuelto "
            "una propuesta."
        )
    elif solicitada is None:
        motivo = (
            "Fallback: ruta ausente o no permitida."
        )
    elif corregida:
        motivo = (
            "Ruta corregida por trabajo pendiente, "
            "orden o límites."
        )
    else:
        motivo = None

    return {
        "necesita_residuos": residuos,
        "necesita_puntos": puntos,
        "siguiente": ruta,
        "ruta_solicitada": solicitada,
        "corregida": corregida,
        "motivo_correccion": motivo,
    }


def fallback_siguiente(
    estado: Mapping[str, Any],
) -> Ruta:
    """Calcula la ruta si el LLM falla o no está disponible."""

    return resolver_enrutamiento(estado)["siguiente"]


# =========================================================
# RUTA CONDICIONAL DEL GRAFO
# =========================================================

def destino_condicional(
    estado: Mapping[str, Any],
) -> str:
    """Traduce FINISH al nodo aplicar de LangGraph."""

    ruta = estado.get("siguiente")

    if not isinstance(ruta, str) or ruta not in _RUTAS:
        raise ValueError(
            "Ruta del supervisor no permitida."
        )

    return (
        "aplicar"
        if ruta == "FINISH"
        else ruta
    )
