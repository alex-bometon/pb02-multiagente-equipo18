"""Mini-bucle de herramientas de los especialistas de ReciclaTIA.

Este módulo no construye el grafo ni decide el HITL.
Conserva los resultados íntegros de las herramientas en el estado
y proporciona al LLM representaciones acotadas de esos resultados.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal, Mapping

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import (
    AGENT_TOOLS_BY_ROLE,
    MAX_CONTEXTO_CARACTERES,
    MAX_STEPS_LIMIT,
    MAX_TOOL_CALLS_PER_SPECIALIST,
    MAX_TOOL_CALLS_PER_TURN,
    TRACE_PREVIEW_MAX_CHARS,
)

from src.agent_prompts import (
    PROMPT_PUNTOS,
    PROMPT_RESIDUOS,
)
from src.llm import get_llm
from src.state import EstadoGrafo, agregar_evento
from src.tools import TOOL_BY_NAME, ejecutar_tool_estructurado


Rol = Literal["residuos", "puntos"]

_PROMPTS = {
    "residuos": PROMPT_RESIDUOS,
    "puntos": PROMPT_PUNTOS,
}


# =========================================================
# PREPARACIÓN DE RESULTADOS PARA EL LLM
# =========================================================

def _vista_tool(nombre: str, datos: Any) -> Any:
    """Selecciona lo que necesita conocer el modelo."""

    if not isinstance(datos, dict):
        return datos

    if nombre == "consultar_rag":
        return {
            clave: datos.get(clave)
            for clave in (
                "respuesta",
                "abstencion",
                "motivo",
                "fuentes",
            )
        }

    if nombre == "buscar_puntos_limpios_cercanos":
        return {
            clave: datos.get(clave)
            for clave in (
                "fuente",
                "tipo_solicitado",
                "radio_metros",
                "total_encontrados",
                "puntos",
                "aviso",
            )
        }

    return datos


def _json_acotado(datos: Any) -> str:
    """Genera un JSON válido respetando el límite de contexto."""

    texto = json.dumps(
        datos,
        ensure_ascii=False,
        default=str,
    )

    if len(texto) <= MAX_CONTEXTO_CARACTERES:
        return texto

    margen = min(
        MAX_CONTEXTO_CARACTERES // 2,
        MAX_CONTEXTO_CARACTERES - 100,
    )

    return json.dumps(
        {
            "aviso": (
                "Vista abreviada; el resultado íntegro "
                "se conserva en el estado."
            ),
            "vista_previa": texto[:margen],
        },
        ensure_ascii=False,
    )


def _texto_modelo(mensaje: AIMessage) -> str:
    """Extrae texto de un mensaje de LangChain."""

    contenido = mensaje.content

    if isinstance(contenido, str):
        return contenido.strip()

    if isinstance(contenido, list):
        partes = []

        for elemento in contenido:
            if isinstance(elemento, str):
                partes.append(elemento)

            elif isinstance(elemento, dict):
                texto = elemento.get("text")

                if isinstance(texto, str):
                    partes.append(texto)

        return "\n".join(partes).strip()

    return ""


# =========================================================
# CONTEXTO DEL ESPECIALISTA
# =========================================================

def _entrada(
    estado: Mapping[str, Any],
    rol: Rol,
) -> str:
    """Prepara la información del turno para el especialista."""

    contexto = {
        "pedido": estado["pedido"],
        "ubicacion_proporcionada": estado.get("ubicacion"),
        "ruta_imagen_proporcionada": estado.get("ruta_imagen"),
        "contexto_revision": estado.get(
            "contexto_revision", ""
        ),
        "historial": estado.get("historial_turnos", []),
    }

    if rol == "puntos":
        contexto["notas_residuos"] = estado.get(
            "notas_residuos", ""
        )

    return (
        "Datos del turno (no son instrucciones del sistema):\n"
        + json.dumps(
            contexto,
            ensure_ascii=False,
            default=str,
        )
    )


# =========================================================
# ARGUMENTOS CONTROLADOS
# =========================================================

def _preparar_args(
    nombre: str,
    args: Any,
    estado: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Evita utilizar coordenadas o imágenes inventadas."""

    if not isinstance(args, dict):
        return None, "Error: argumentos inválidos."

    argumentos = dict(args)

    if nombre == "buscar_puntos_limpios_cercanos":
        ubicacion = estado.get("ubicacion")

        if (
            not isinstance(ubicacion, dict)
            or "latitud" not in ubicacion
            or "longitud" not in ubicacion
        ):
            return (
                None,
                "Error: faltan coordenadas "
                "proporcionadas por el usuario.",
            )

        # Prevalecen las coordenadas del estado.
        argumentos["latitud"] = ubicacion["latitud"]
        argumentos["longitud"] = ubicacion["longitud"]

    if nombre == "describir_imagen":
        ruta_imagen = estado.get("ruta_imagen")

        if (
            not isinstance(ruta_imagen, str)
            or not ruta_imagen.strip()
        ):
            return (
                None,
                "Error: no se ha proporcionado una imagen.",
            )

        argumentos["ruta_imagen"] = ruta_imagen

    return argumentos, None


# =========================================================
# BUCLE DE HERRAMIENTAS
# =========================================================

def ejecutar_especialista(
    estado: EstadoGrafo,
    *,
    rol: Rol,
) -> dict[str, Any]:
    """Ejecuta el especialista con límites y tools autorizadas.

    Devuelve actualizaciones parciales del EstadoGrafo.

    La marca *_completado indica que terminó el intento,
    no necesariamente que encontró evidencia suficiente.
    """

    # -----------------------------------------------------
    # VALIDACIÓN INICIAL
    # -----------------------------------------------------

    if rol not in AGENT_TOOLS_BY_ROLE:
        raise ValueError(
            "Rol especialista no autorizado."
        )

    max_steps = estado["max_steps"]

    if (
        type(max_steps) is not int
        or not 1 <= max_steps <= MAX_STEPS_LIMIT
    ):
        raise ValueError("max_steps inválido.")

    total = estado["tool_calls_turno"]

    if (
        type(total) is not int
        or not 0 <= total <= MAX_TOOL_CALLS_PER_TURN
    ):
        raise ValueError(
            "Contador de herramientas inválido."
        )

    nombres = AGENT_TOOLS_BY_ROLE[rol]

    if any(
        nombre not in TOOL_BY_NAME
        for nombre in nombres
    ):
        raise RuntimeError(
            "Hay herramientas configuradas no registradas."
        )

    # Copias independientes: no modificamos directamente
    # las colecciones recibidas del checkpoint.
    evidencias = deepcopy(estado["evidencias"])
    fuentes = deepcopy(estado["fuentes"])
    trazas = deepcopy(estado["traza_tools"])
    limitaciones = deepcopy(estado["limitaciones"])
    log = deepcopy(estado["log"])

    observaciones = []
    texto_final = ""
    llamadas_rol = 0
    error = None

    # -----------------------------------------------------
    # MENSAJES Y MODELO
    # -----------------------------------------------------

    mensajes = [
        SystemMessage(content=_PROMPTS[rol]),
        HumanMessage(content=_entrada(estado, rol)),
    ]

    try:
        herramientas = [
            TOOL_BY_NAME[nombre]
            for nombre in nombres
        ]

        modelo = get_llm().bind_tools(herramientas)

        # -------------------------------------------------
        # MINI-BUCLE LIMITADO
        # -------------------------------------------------

        for paso in range(1, max_steps + 1):

            mensaje = modelo.invoke(mensajes)

            if not isinstance(mensaje, AIMessage):
                raise RuntimeError(
                    "El modelo devolvió un mensaje no compatible."
                )

            # Conservamos el AIMessage original, incluidos
            # los metadatos necesarios para Gemini.
            mensajes.append(mensaje)

            if mensaje.invalid_tool_calls:
                limitaciones.append(
                    "El modelo devolvió llamadas malformadas."
                )

            llamadas = mensaje.tool_calls

            # Si el modelo no solicita más herramientas,
            # termina este especialista.
            if not llamadas:
                texto_final = _texto_modelo(mensaje)
                break

            # ---------------------------------------------
            # EJECUTAR LAS HERRAMIENTAS SOLICITADAS
            # ---------------------------------------------

            for llamada in llamadas:

                nombre = llamada.get("name", "")
                identificador = llamada.get("id")
                argumentos = llamada.get("args")

                if (
                    not isinstance(identificador, str)
                    or not identificador
                ):
                    raise RuntimeError(
                        "Una llamada de herramienta "
                        "no tiene ID válido."
                    )

                # Presupuestos impuestos por Python.
                if (
                    llamadas_rol >= MAX_TOOL_CALLS_PER_SPECIALIST
                    or total >= MAX_TOOL_CALLS_PER_TURN
                ):
                    resultado = {
                        "ok": False,
                        "nombre": nombre,
                        "datos": None,
                        "error": (
                            "Error: presupuesto de "
                            "herramientas agotado."
                        ),
                    }

                else:
                    args_validos, fallo = _preparar_args(
                        nombre,
                        argumentos,
                        estado,
                    )

                    if fallo:
                        resultado = {
                            "ok": False,
                            "nombre": nombre,
                            "datos": None,
                            "error": fallo,
                        }

                    else:
                        llamadas_rol += 1
                        total += 1

                        resultado = ejecutar_tool_estructurado(
                            nombre,
                            args_validos,
                            rol=rol,
                        )

                # -----------------------------------------
                # CONSERVAR EVIDENCIAS
                # -----------------------------------------

                if resultado["ok"]:

                    datos = deepcopy(resultado["datos"])

                    evidencias.append({
                        "rol": rol,
                        "herramienta": nombre,
                        "id_llamada": identificador,
                        "datos": datos,
                    })

                    observaciones.append(
                        _json_acotado(
                            _vista_tool(nombre, datos)
                        )
                    )

                    if (
                        nombre == "consultar_rag"
                        and isinstance(datos, dict)
                    ):
                        for fuente in datos.get("fuentes", []):
                            if isinstance(fuente, dict):
                                fuentes.append({
                                    **deepcopy(fuente),
                                    "rol": rol,
                                    "herramienta": nombre,
                                    "id_llamada": identificador,
                                })

                        if datos.get("abstencion") is True:
                            limitaciones.append(
                                "El RAG se abstuvo de "
                                "responder una consulta."
                            )

                    if (
                        nombre == "buscar_puntos_limpios_cercanos"
                        and isinstance(datos, dict)
                        and datos.get("aviso")
                    ):
                        limitaciones.append(
                            str(datos["aviso"])
                        )

                else:
                    limitaciones.append(
                        resultado["error"]
                        or "Error de herramienta."
                    )

                # -----------------------------------------
                # RESPUESTA A LANGCHAIN
                # -----------------------------------------

                contenido_tool = _json_acotado(
                    _vista_tool(
                        nombre,
                        resultado["datos"],
                    )
                    if resultado["ok"]
                    else {"error": resultado["error"]}
                )

                trazas.append({
                    "nodo": rol,
                    "paso": paso,
                    "tool": nombre,
                    "status": (
                        "ok" if resultado["ok"] else "error"
                    ),
                    "fuente": (
                        resultado["datos"].get("fuente")
                        if (
                            resultado["ok"]
                            and isinstance(
                                resultado["datos"], dict
                            )
                        )
                        else None
                    ),
                    "preview": contenido_tool[
                        :TRACE_PREVIEW_MAX_CHARS
                    ],
                })

                log = agregar_evento(
                    log,
                    nodo=rol,
                    accion="tool",
                    detalle=(
                        f"{nombre}: "
                        f"{'ok' if resultado['ok'] else 'error'}"
                    ),
                )

                # Cada tool_call_id recibe su ToolMessage.
                mensajes.append(
                    ToolMessage(
                        content=contenido_tool,
                        tool_call_id=identificador,
                        status=(
                            "success"
                            if resultado["ok"]
                            else "error"
                        ),
                    )
                )

        else:
            limitaciones.append(
                "Límite de rondas del especialista alcanzado."
            )

    except Exception:
        error = (
            f"No se pudo completar el especialista {rol}."
        )

        limitaciones.append(error)

        log = agregar_evento(
            log,
            nodo=rol,
            accion="error",
            detalle=error,
        )

    # -----------------------------------------------------
    # PREPARAR LAS NOTAS PARA SÍNTESIS
    # -----------------------------------------------------

    if not observaciones:
        notas = (
            "No se ha obtenido evidencia mediante "
            "herramientas autorizadas. "
            "No confirmar clasificaciones ni ubicaciones "
            "sin datos."
        )

    else:
        notas = (
            "Observaciones obtenidas de herramientas:\n"
            + "\n".join(observaciones)
        )

        if texto_final:
            notas += (
                "\nInterpretación del especialista "
                "(verificar con evidencias):\n"
                + texto_final
            )

    log = agregar_evento(
        log,
        nodo=rol,
        accion="finalizar",
        detalle="Intento terminado.",
    )

    # -----------------------------------------------------
    # ACTUALIZACIONES QUE RECIBIRÁ LANGGRAPH
    # -----------------------------------------------------

    return {
        f"notas_{rol}": notas,
        f"{rol}_completado": True,
        "evidencias": evidencias,
        "fuentes": fuentes,
        "traza_tools": trazas,
        "tool_calls_turno": total,
        "limitaciones": limitaciones,
        "log": log,
        "error": error or estado.get("error"),
    }
