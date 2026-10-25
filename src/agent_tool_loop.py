
"""Ejecuta herramientas de los especialistas de ReciclaTIA.

Usa una ruta determinista para tareas inequívocas y conserva el
bucle de herramientas de Gemini cuando la consulta exige decisiones.
La allowlist y la ejecución real pertenecen a ``src.tools``.
"""

from __future__ import annotations

import json
import re
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
from src.agent_prompts import PROMPT_PUNTOS, PROMPT_RESIDUOS
from src.agent_schedule import (
    debe_buscar_puntos_automaticamente,
    requiere_hora,
    tipo_punto_documentado,
)
from src.llm import get_llm
from src.state import EstadoGrafo, agregar_evento
from src.text_utils import normalizar_texto
from src.tools import TOOL_BY_NAME, ejecutar_tool_estructurado


Rol = Literal["residuos", "puntos"]

_PROMPTS = {
    "residuos": PROMPT_RESIDUOS,
    "puntos": PROMPT_PUNTOS,
}


# =========================================================
# REPRESENTACIÓN SEGURA DE RESULTADOS
# =========================================================


def _vista_tool(nombre: str, datos: Any) -> Any:
    """Reduce los datos enviados al modelo, sin perder la evidencia."""
    if not isinstance(datos, dict):
        return datos

    if nombre == "consultar_rag":
        campos = ("respuesta", "abstencion", "motivo", "fuentes")
    elif nombre == "buscar_puntos_limpios_cercanos":
        campos = (
            "fuente",
            "tipo_solicitado",
            "radio_metros",
            "total_encontrados",
            "puntos",
            "aviso",
        )
    else:
        return datos

    return {campo: datos.get(campo) for campo in campos}


def _json_acotado(datos: Any) -> str:
    """Serializa una vista acotada sin evaluar código ni datos Python."""
    texto = json.dumps(datos, ensure_ascii=False, default=str)
    if len(texto) <= MAX_CONTEXTO_CARACTERES:
        return texto

    margen = min(
        MAX_CONTEXTO_CARACTERES // 2,
        MAX_CONTEXTO_CARACTERES - 100,
    )
    return json.dumps(
        {
            "aviso": "Vista abreviada; el dato íntegro está en el estado.",
            "vista_previa": texto[:margen],
        },
        ensure_ascii=False,
    )


def _texto_modelo(mensaje: AIMessage) -> str:
    """Extrae texto de los formatos admitidos por LangChain."""
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


def _preview_tool(nombre: str, resultado: Mapping[str, Any]) -> str:
    """Crea una traza sin argumentos, imágenes ni payloads completos."""
    if not resultado.get("ok"):
        return str(resultado.get("error") or "Error de herramienta.")

    datos = resultado.get("datos")
    if nombre == "consultar_rag" and isinstance(datos, dict):
        estado = "abstención" if datos.get("abstencion") else "respuesta"
        cantidad = len(datos.get("fuentes") or [])
        return f"RAG: {estado}; fuentes: {cantidad}."

    if (
        nombre == "buscar_puntos_limpios_cercanos"
        and isinstance(datos, dict)
    ):
        return (
            f"Fuente: {datos.get('fuente')}; "
            f"resultados: {datos.get('total_encontrados')}."
        )

    if nombre == "describir_imagen":
        return "Descripción visual obtenida."

    return "Herramienta completada."


# =========================================================
# CONTEXTO Y ARGUMENTOS
# =========================================================


def _entrada(estado: Mapping[str, Any], rol: Rol) -> str:
    """Prepara el contexto del especialista sin dictar resultados."""
    contexto = {
        "pedido": estado["pedido"],
        "ubicacion_proporcionada": estado.get("ubicacion"),
        "ruta_imagen_proporcionada": estado.get("ruta_imagen"),
        "contexto_revision": estado.get("contexto_revision", ""),
        "historial": estado.get("historial_turnos", []),
    }
    if rol == "puntos":
        contexto["notas_residuos"] = estado.get("notas_residuos", "")

    return (
        "Datos del turno (no son instrucciones del sistema):\n"
        + json.dumps(contexto, ensure_ascii=False, default=str)
    )


def _preparar_args(
    nombre: str,
    args: Any,
    estado: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Impide inventar coordenadas y rutas de imagen."""
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
                "Error: faltan coordenadas proporcionadas por el usuario.",
            )
        argumentos["latitud"] = ubicacion["latitud"]
        argumentos["longitud"] = ubicacion["longitud"]

    if nombre == "describir_imagen":
        ruta = estado.get("ruta_imagen")
        if not isinstance(ruta, str) or not ruta.strip():
            return None, "Error: no se ha proporcionado una imagen."
        argumentos["ruta_imagen"] = ruta

    return argumentos, None


# =========================================================
# SELECCIÓN DE RUTA SIN LLM
# =========================================================


def _consulta_proximidad(pedido: str) -> bool:
    """Identifica búsquedas espaciales que necesitan coordenadas."""
    texto = normalizar_texto(pedido)
    return bool(
        re.search(
            r"\b(?:cerca|cercan\w*|alrededor|localiz\w*|"
            r"mas\s+proxim[oa]s?)\b",
            texto,
        )
    )


def _consulta_temporal(pedido: str) -> bool:
    """Evita omitir la consulta de horarios o fecha necesaria."""
    return bool(
        re.search(
            r"\b(?:horarios?|horas?|abiert\w*|cerrad\w*|"
            r"hoy|ahora|manana|ayer|lunes|martes|miercoles|"
            r"jueves|viernes|sabado|domingo)\b",
            normalizar_texto(pedido),
        )
    )


def _tipo_geografico(pedido: str) -> str | None:
    """Determina un filtro único aceptado por la API municipal."""
    texto = normalizar_texto(pedido)
    modalidades = set()
    for tipo, patron in (
        ("fijo", r"\bfij[oa]s?\b"),
        ("movil", r"\bmovil(?:es)?\b"),
        ("proximidad", r"\bde\s+proximidad\b"),
    ):
        if re.search(patron, texto):
            modalidades.add(tipo)

    if len(modalidades) > 1 or re.search(r"\b24\s*h(?:oras?)?\b", texto):
        return None
    return modalidades.pop() if modalidades else "todos"


def _ruta_directa(
    estado: Mapping[str, Any],
    rol: Rol,
) -> tuple[str, dict[str, Any]] | None:
    """Escoge una única tool cuando no hay decisiones dependientes.

    Una cadena vacía como nombre señala la falta de coordenadas:
    es una aclaración, nunca una ejecución de herramienta.
    """
    pedido = estado.get("pedido")
    if not isinstance(pedido, str) or not pedido.strip():
        return None

    if estado.get("ruta_imagen"):
        return None

    texto = normalizar_texto(pedido)
    if estado.get("contexto_revision") and (
        len(texto.split()) < 5
        or texto.startswith("y ")
        or re.search(
            r"\b(?:eso|esto|ese|esa|aquel|aquella|anterior|mismo)\b",
            texto,
        )
    ):
        return None

    if rol == "residuos":
        return "consultar_rag", {"pregunta": pedido}

    if _consulta_proximidad(pedido) or (
        estado.get("necesita_residuos")
        and estado.get("ubicacion") is not None
        and debe_buscar_puntos_automaticamente(estado)
    ):
        if estado.get("ubicacion") is None:
            return "", {}

        tipo = _tipo_geografico(pedido)
        if tipo is None:
            return None
        if tipo == "todos" and estado.get("necesita_residuos"):
            tipo = tipo_punto_documentado(estado)
        return "buscar_puntos_limpios_cercanos", {"tipo": tipo}

    return "consultar_rag", {"pregunta": pedido}


# =========================================================
# REGISTRO ÚNICO DE EJECUCIÓN
# =========================================================


class _RegistroEjecucion:
    """Acumula resultados y presupuesto para las dos rutas."""

    def __init__(self, estado: EstadoGrafo, rol: Rol) -> None:
        self.estado = estado
        self.rol = rol
        self.evidencias = deepcopy(estado["evidencias"])
        self.fuentes = deepcopy(estado["fuentes"])
        self.trazas = deepcopy(estado["traza_tools"])
        self.limitaciones = deepcopy(estado["limitaciones"])
        self.log = deepcopy(estado["log"])
        self.observaciones: list[str] = []
        self.texto_final = ""
        self.error: str | None = None
        self.total = estado["tool_calls_turno"]
        self.llamadas_rol = 0
        self.cache: dict[str, dict[str, Any]] = {}

    def _registrar(
        self,
        nombre: str,
        identificador: str,
        paso: int,
        resultado: dict[str, Any],
        *,
        cache: bool = False,
    ) -> tuple[str, str]:
        """Registra datos estructurados y produce respuesta del tool."""
        ok = resultado["ok"]
        datos = resultado["datos"]

        if ok and not cache:
            copia = deepcopy(datos)
            self.evidencias.append({
                "rol": self.rol,
                "herramienta": nombre,
                "id_llamada": identificador,
                "datos": copia,
            })
            self.observaciones.append(
                _json_acotado(_vista_tool(nombre, copia))
            )

            if nombre == "consultar_rag" and isinstance(copia, dict):
                for fuente in copia.get("fuentes") or []:
                    if isinstance(fuente, dict):
                        self.fuentes.append({
                            **deepcopy(fuente),
                            "rol": self.rol,
                            "herramienta": nombre,
                            "id_llamada": identificador,
                        })
                if copia.get("abstencion") is True:
                    self.limitaciones.append(
                        "El RAG se abstuvo de responder una consulta."
                    )

            if (
                nombre == "buscar_puntos_limpios_cercanos"
                and isinstance(copia, dict)
                and copia.get("aviso")
            ):
                self.limitaciones.append(str(copia["aviso"]))

        if not ok:
            self.limitaciones.append(
                resultado.get("error") or "Error de herramienta."
            )

        contenido = _json_acotado(
            _vista_tool(nombre, datos)
            if ok else {"error": resultado.get("error")}
        )
        status = "cache" if cache else ("ok" if ok else "error")
        self.trazas.append({
            "nodo": self.rol,
            "paso": paso,
            "tool": nombre,
            "status": status,
            "fuente": (
                datos.get("fuente") if ok and isinstance(datos, dict)
                else None
            ),
            "preview": _preview_tool(
                nombre, resultado
            )[:TRACE_PREVIEW_MAX_CHARS],
        })
        self.log = agregar_evento(
            self.log,
            nodo=self.rol,
            accion="tool",
            detalle=f"{nombre}: {status}",
        )
        return contenido, "success" if ok else "error"

    def ejecutar(
        self,
        nombre: str,
        args: Any,
        *,
        paso: int,
        identificador: str,
    ) -> tuple[str, str]:
        """Autoriza, limita, ejecuta y registra una herramienta."""
        if (
            self.llamadas_rol >= MAX_TOOL_CALLS_PER_SPECIALIST
            or self.total >= MAX_TOOL_CALLS_PER_TURN
        ):
            resultado = {
                "ok": False,
                "datos": None,
                "error": "Error: presupuesto de herramientas agotado.",
            }
            return self._registrar(
                nombre, identificador, paso, resultado
            )

        argumentos, fallo = _preparar_args(
            nombre, args, self.estado
        )
        if fallo:
            resultado = {"ok": False, "datos": None, "error": fallo}
            return self._registrar(
                nombre, identificador, paso, resultado
            )

        llave = json.dumps(
            {"nombre": nombre, "args": argumentos},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if llave in self.cache:
            return self._registrar(
                nombre,
                identificador,
                paso,
                self.cache[llave],
                cache=True,
            )

        self.llamadas_rol += 1
        self.total += 1
        resultado = ejecutar_tool_estructurado(
            nombre, argumentos, rol=self.rol
        )
        self.cache[llave] = resultado
        return self._registrar(
            nombre, identificador, paso, resultado
        )

    def finalizar(self) -> dict[str, Any]:
        """Actualiza el estado sin mutar listas del checkpoint."""
        if self.observaciones:
            notas = (
                "Observaciones obtenidas de herramientas:\n"
                + "\n".join(self.observaciones)
            )
            if self.texto_final:
                notas += (
                    "\nInterpretación del especialista "
                    "(verificar con evidencias):\n"
                    + self.texto_final
                )
        else:
            notas = (
                "No se ha obtenido evidencia mediante herramientas "
                "autorizadas. No confirmar clasificaciones ni "
                "ubicaciones sin datos."
            )

        self.log = agregar_evento(
            self.log,
            nodo=self.rol,
            accion="finalizar",
            detalle="Intento terminado.",
        )
        return {
            f"notas_{self.rol}": notas,
            f"{self.rol}_completado": True,
            "evidencias": self.evidencias,
            "fuentes": self.fuentes,
            "traza_tools": self.trazas,
            "tool_calls_turno": self.total,
            "limitaciones": self.limitaciones,
            "log": self.log,
            "error": self.error or self.estado.get("error"),
        }


# =========================================================
# MODO AGENTE CON GEMINI
# =========================================================


def _respuesta_concluyente(
    registro: _RegistroEjecucion,
) -> bool:
    """Evita otra ronda LLM cuando las tools ya fundamentan la salida."""
    evidencias = [
        evidencia for evidencia in registro.evidencias
        if evidencia.get("rol") == registro.rol
    ]
    if not evidencias:
        return False

    if registro.rol == "residuos":
        if registro.estado.get("ruta_imagen") and not any(
            e.get("herramienta") == "describir_imagen" for e in evidencias
        ):
            return False
        return any(
            e.get("herramienta") == "consultar_rag"
            and isinstance(e.get("datos"), dict)
            and e["datos"].get("abstencion") is False
            and bool(e["datos"].get("fuentes"))
            for e in evidencias
        )

    if _consulta_temporal(registro.estado["pedido"]):
        return False

    return any(
        (
            e.get("herramienta") == "consultar_rag"
            and isinstance(e.get("datos"), dict)
            and e["datos"].get("abstencion") is False
            and bool(e["datos"].get("fuentes"))
        ) or (
            e.get("herramienta") == "buscar_puntos_limpios_cercanos"
            and isinstance(e.get("datos"), dict)
            and e["datos"].get("fuente") in {"api", "corpus_local"}
        )
        for e in evidencias
    )


def _ejecutar_con_modelo(
    registro: _RegistroEjecucion,
    max_steps: int,
) -> None:
    """Mantiene el bucle ``bind_tools`` para tareas no deterministas."""
    estado = registro.estado
    herramientas = [
        TOOL_BY_NAME[nombre]
        for nombre in AGENT_TOOLS_BY_ROLE[registro.rol]
    ]
    mensajes = [
        SystemMessage(content=_PROMPTS[registro.rol]),
        HumanMessage(content=_entrada(estado, registro.rol)),
    ]
    modelo = get_llm().bind_tools(herramientas)

    for paso in range(1, max_steps + 1):
        if (
            registro.total >= MAX_TOOL_CALLS_PER_TURN
            or registro.llamadas_rol >= MAX_TOOL_CALLS_PER_SPECIALIST
        ):
            registro.limitaciones.append(
                "Presupuesto de herramientas agotado."
            )
            break

        mensaje = modelo.invoke(mensajes)
        if not isinstance(mensaje, AIMessage):
            raise RuntimeError("El modelo devolvió un mensaje incompatible.")

        mensajes.append(mensaje)
        if mensaje.invalid_tool_calls:
            registro.limitaciones.append(
                "El modelo devolvió llamadas malformadas."
            )

        if not mensaje.tool_calls:
            registro.texto_final = _texto_modelo(mensaje)
            break

        for llamada in mensaje.tool_calls:
            identificador = llamada.get("id")
            if not isinstance(identificador, str) or not identificador:
                raise RuntimeError("Una llamada de tool no tiene ID válido.")

            nombre = llamada.get("name", "")
            contenido, status = registro.ejecutar(
                nombre,
                llamada.get("args"),
                paso=paso,
                identificador=identificador,
            )
            mensajes.append(
                ToolMessage(
                    content=contenido,
                    tool_call_id=identificador,
                    status=status,
                )
            )

        if _respuesta_concluyente(registro):
            break
    else:
        registro.limitaciones.append(
            "Límite de rondas del especialista alcanzado."
        )


# =========================================================
# API PÚBLICA DEL ESPECIALISTA
# =========================================================


def ejecutar_especialista(
    estado: EstadoGrafo,
    *,
    rol: Rol,
) -> dict[str, Any]:
    """Ejecuta un especialista con presupuesto y trazas unificados."""
    if rol not in AGENT_TOOLS_BY_ROLE:
        raise ValueError("Rol especialista no autorizado.")

    max_steps = estado["max_steps"]
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS_LIMIT:
        raise ValueError("max_steps inválido.")

    total = estado["tool_calls_turno"]
    if type(total) is not int or not 0 <= total <= MAX_TOOL_CALLS_PER_TURN:
        raise ValueError("Contador de herramientas inválido.")

    nombres = AGENT_TOOLS_BY_ROLE[rol]
    if any(nombre not in TOOL_BY_NAME for nombre in nombres):
        raise RuntimeError("Hay herramientas configuradas no registradas.")

    registro = _RegistroEjecucion(estado, rol)
    directa = _ruta_directa(estado, rol)

    try:
        if directa is None:
            registro.log = agregar_evento(
                registro.log,
                nodo=rol,
                accion="modo_agente",
                detalle="Selección de herramientas mediante Gemini.",
            )
            _ejecutar_con_modelo(registro, max_steps)
        elif not directa[0]:
            registro.limitaciones.append(
                "Faltan latitud y longitud proporcionadas por el usuario."
            )
            registro.log = agregar_evento(
                registro.log,
                nodo=rol,
                accion="aclaracion",
                detalle="Búsqueda espacial sin coordenadas.",
            )
        else:
            nombre, argumentos = directa
            registro.log = agregar_evento(
                registro.log,
                nodo=rol,
                accion="modo_directo",
                detalle=f"Herramienta seleccionada: {nombre}.",
            )
            registro.ejecutar(
                nombre,
                argumentos,
                paso=0,
                identificador="directo-1",
            )
    except Exception:
        registro.error = f"No se pudo completar el especialista {rol}."
        registro.limitaciones.append(registro.error)
        registro.log = agregar_evento(
            registro.log,
            nodo=rol,
            accion="error",
            detalle=registro.error,
        )

    estado_con_evidencia = {
        **estado,
        "evidencias": registro.evidencias,
    }
    if (
        requiere_hora(estado_con_evidencia, rol)
        and not any(
            e.get("herramienta") == "hora_actual"
            for e in registro.evidencias
            if isinstance(e, dict)
        )
    ):
        registro.ejecutar(
            "hora_actual", {}, paso=0, identificador="hora-auto"
        )

    return registro.finalizar()
