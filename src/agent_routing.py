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

# Detectores de intención explícita.
# Son reglas de dominio, no parámetros operativos.

_PATRON_RESIDUOS = re.compile(
    r"\b(?:contenedores?|fraccion(?:es)?|clasific\w*|"
    r"recicl\w*|tir\w*|deposit\w*|desech\w*|separ\w*)\b"
    r"|\b(?:tipos?|categorias?|destinos?)\s+(?:de\s+)?"
    r"(?:los\s+)?residuos?\b"
)

# Preguntas como «¿Dónde van las pilas?» son solicitudes de destino.
# Se evita confundirlas con preguntas sobre las propias instalaciones.
_PATRON_DESTINO_IMPLICITO = re.compile(
    r"\b(?:donde|adonde)\s+(?:va|van)\s+"
    r"(?:el|la|los|las|un|una|mi|mis|tu|tus|"
    r"este|esta|estos|estas|ese|esa|esos|esas)\s+"
    r"(?!(?:puntos?\s+limpios?|centros?\s+de\s+recogida|"
    r"paradas?\s+de\s+recogida|"
    r"instalaciones?\s+de\s+recogida)\b)"
    r"\w+"
)

_PATRON_PUNTOS = re.compile(
    r"\b(?:puntos?\s+limpios?|centros?\s+de\s+recogida|"
    r"instalaciones?\s+de\s+recogida|"
    r"paradas?\s+(?:de\s+)?recogida)\b"
)

# Algunas expresiones solicitan ubicaciones sin identificar
# claramente el tipo de instalación.
# En esos casos puede ser necesaria una interpretación LLM.
_PATRON_UBICACION_INDEFINIDA = re.compile(
    r"\b(?:cerca(?:\s+de)?|cercan\w*|proxim\w*|"
    r"por\s+mi\s+zona|en\s+mi\s+barrio|"
    r"ubicacion|localiz\w*)\b"
    r"|\b(?:donde\s+(?:puedo|podria|podemos)\s+"
    r"(?:llevar|entregar|dejar))\b"
)

_RUTAS = frozenset({
    "residuos",
    "puntos",
    "sintetizar",
    "FINISH",
})


def _necesidades_texto(
    texto: str,
) -> tuple[bool, bool]:
    """Detecta necesidades explícitas en texto normalizado."""
    normalizado = normalizar_texto(texto)

    residuos = bool(
        _PATRON_RESIDUOS.search(normalizado)
        or _PATRON_DESTINO_IMPLICITO.search(normalizado)
    )
    puntos = bool(_PATRON_PUNTOS.search(normalizado))

    return residuos, puntos


def _senales_iniciales(
    estado: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Devuelve necesidades reconocidas y posible ambigüedad.

    No llama al LLM ni realiza cambios en el estado.
    """
    pedido = estado.get("pedido", "")

    texto = normalizar_texto(
        pedido if isinstance(pedido, str) else ""
    )

    residuos, puntos = _necesidades_texto(texto)

    # Una solicitud de proximidad sin instalación explícita
    # puede necesitar interpretación adicional.
    ubicacion_indefinida = bool(
        _PATRON_UBICACION_INDEFINIDA.search(texto)
    ) and not puntos

    # El contexto previo solo interviene si la consulta
    # actual no contiene ninguna intención reconocible.
    if not (residuos or puntos):
        contexto = estado.get("contexto_revision", "")

        if isinstance(contexto, str):
            residuos, puntos = _necesidades_texto(contexto)

    # Una imagen proporcionada activa residuos.
    if estado.get("ruta_imagen"):
        residuos = True

    return residuos, puntos, ubicacion_indefinida


# =========================================================
# DECISIÓN DE CONSULTAR AL LLM
# =========================================================


def requiere_llm_supervisor(
    estado: Mapping[str, Any],
) -> bool:
    """Indica si Gemini aporta valor para decidir la ruta.

    Solo se plantea una llamada en la primera visita
    del supervisor durante el turno.

    En visitas posteriores, los especialistas pendientes
    y la finalización se determinan mediante Python.
    """
    if estado.get("siguiente"):
        return False

    residuos, puntos, ambigua = _senales_iniciales(estado)

    if not (residuos or puntos):
        return True

    if ambigua:
        return True

    return False


# =========================================================
# FIJACIÓN DE ESPECIALISTAS POR TURNO
# =========================================================


def _necesidades_turno(
    estado: Mapping[str, Any],
    propuesta: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    """Fija las especialidades necesarias para este turno.

    Las necesidades explícitas tienen prioridad sobre
    cualquier propuesta del modelo.
    """
    if estado.get("siguiente"):
        return (
            estado.get("necesita_residuos") is True,
            estado.get("necesita_puntos") is True,
        )

    residuos, puntos, ambigua = _senales_iniciales(estado)

    if (
        ambigua or not (residuos or puntos)
    ) and isinstance(propuesta, Mapping):
        sugerencia_residuos = propuesta.get("necesita_residuos")
        sugerencia_puntos = propuesta.get("necesita_puntos")

        if (
            type(sugerencia_residuos) is bool
            and type(sugerencia_puntos) is bool
        ):
            return (
                residuos or sugerencia_residuos,
                puntos or sugerencia_puntos,
            )

    return residuos, puntos


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

    if iteracion >= MAX_ITER:
        raise RuntimeError(
            "Se ha superado MAX_ITER en el supervisor."
        )

    if iteracion == MAX_ITER - 1:
        if salida and not (
            pendiente_residuos or pendiente_puntos
        ):
            return "FINISH"

        raise RuntimeError(
            "Última decisión del supervisor sin plan completo."
        )

    if salida and not (
        pendiente_residuos or pendiente_puntos
    ):
        return "FINISH"

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

    if pendiente_residuos:
        return "residuos"

    if pendiente_puntos:
        return "puntos"

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
    """Valida la sugerencia y obtiene la ruta ejecutable.

    No modifica el estado. src.nodes.supervisor registrará
    la decisión e incrementará el contador de iteraciones.
    """
    residuos, puntos = _necesidades_turno(estado, propuesta)
    ruta = _ruta_obligatoria(estado, residuos, puntos)

    solicitada = None
    if isinstance(propuesta, Mapping):
        valor = propuesta.get("siguiente")
        if isinstance(valor, str) and valor in _RUTAS:
            solicitada = valor

    corregida = propuesta is not None and solicitada != ruta

    if propuesta is None:
        motivo = "Ruta decidida por Python sin propuesta del modelo."
    elif solicitada is None:
        motivo = "Fallback: ruta ausente o no permitida."
    elif corregida:
        motivo = "Ruta corregida por trabajo pendiente, orden o límites."
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


def destino_condicional(
    estado: Mapping[str, Any],
) -> str:
    """Traduce FINISH al nodo aplicar de LangGraph."""
    ruta = estado.get("siguiente")

    if not isinstance(ruta, str) or ruta not in _RUTAS:
        raise ValueError(
            "Ruta del supervisor no permitida."
        )

    return "aplicar" if ruta == "FINISH" else ruta
