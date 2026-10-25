"""Cliente Streamlit de ReciclaTIA PB02.

La interfaz no ejecuta nodos ni herramientas directamente. Toda consulta
se procesa mediante la API pública de src.agent y su checkpoint LangGraph.
"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import (
    ASSETS_DIR,
    IMAGE_ALLOWED_FORMATS,
    IMAGE_MAX_BYTES,
    IMAGE_MAX_PIXELS,
    MAX_STEPS_DEFAULT,
    MAX_STEPS_LIMIT,
    MAX_TURNS_DEFAULT,
    MAX_TURNS_LIMIT,
)
from src.agent import aprobar_plan, nuevo_thread_id, procesar_turno
from src.state import EstadoProducto, crear_estado_producto


# =========================================================
# PRESENTACIÓN Y ESTADO DE SESIÓN
# =========================================================

st.set_page_config(
    page_title="ReciclaTIA | Asistente de reciclaje",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ESTILOS = """
<style>
.block-container {max-width: 1220px; padding-top: 2rem;}
.rt-hero {
    padding: 1.35rem 1.65rem;
    border: 1px solid rgba(66, 145, 105, .24);
    border-radius: 18px;
    background: linear-gradient(120deg,
        rgba(35, 115, 79, .11), rgba(62, 138, 116, .025));
    margin-bottom: 1.1rem;
}
.rt-eyebrow {
    font-size: .75rem; letter-spacing: .12em;
    font-weight: 700; color: #27875e;
}
.rt-hero h1 {font-size: 2.2rem; margin: .2rem 0 .4rem 0;}
.rt-hero p {margin: 0; opacity: .8;}
.rt-note {font-size: .86rem; opacity: .8;}
div[data-testid="stChatMessage"] {border-radius: 14px;}
</style>
"""

EJEMPLOS = (
    "¿Dónde tiro una botella de plástico?",
    "¿Qué es un punto limpio móvil?",
    "Busca puntos limpios de proximidad cerca de mí",
)


def _iniciar_sesion() -> None:
    """Inicializa una conversación por sesión de navegador."""
    if "rt_max_turns" not in st.session_state:
        st.session_state.rt_max_turns = MAX_TURNS_DEFAULT
    if "rt_max_steps" not in st.session_state:
        st.session_state.rt_max_steps = MAX_STEPS_DEFAULT
    if "rt_usar_ubicacion" not in st.session_state:
        st.session_state.rt_usar_ubicacion = False
    if "rt_latitud" not in st.session_state:
        st.session_state.rt_latitud = 40.4168
    if "rt_longitud" not in st.session_state:
        st.session_state.rt_longitud = -3.7038
    if "rt_mostrar_mapa" not in st.session_state:
        st.session_state.rt_mostrar_mapa = True
    if "rt_mostrar_debug" not in st.session_state:
        st.session_state.rt_mostrar_debug = False
    if "rt_version_imagen" not in st.session_state:
        st.session_state.rt_version_imagen = 0
    if "rt_aviso" not in st.session_state:
        st.session_state.rt_aviso = None
    if "rt_thread_id" not in st.session_state:
        _nueva_conversacion()


def _nueva_conversacion() -> None:
    """Crea un hilo aislado y conserva solo ajustes no sensibles."""
    hilo = nuevo_thread_id()
    st.session_state.rt_thread_id = hilo
    st.session_state.rt_estado = crear_estado_producto(
        thread_id=hilo,
        max_turns=st.session_state.rt_max_turns,
    )
    st.session_state.rt_mensajes = []
    st.session_state.rt_usar_ubicacion = False
    st.session_state.rt_version_imagen += 1
    st.session_state.rt_aviso = (
        "info",
        "Nueva conversación iniciada. La ubicación se ha desactivado.",
    )


def _mostrar_aviso() -> None:
    """Muestra una sola vez el aviso de la acción anterior."""
    aviso = st.session_state.rt_aviso
    if aviso is None:
        return
    clase, mensaje = aviso
    st.session_state.rt_aviso = None
    if clase == "error":
        st.error(mensaje)
    elif clase == "success":
        st.success(mensaje)
    else:
        st.info(mensaje)


def _estado() -> EstadoProducto:
    """Obtiene el último estado devuelto por el orquestador."""
    return st.session_state.rt_estado


# =========================================================
# ENTRADAS E IMÁGENES
# =========================================================


def _ubicacion() -> dict[str, float] | None:
    """Usa coordenadas solo si el usuario activó expresamente la opción."""
    if not st.session_state.rt_usar_ubicacion:
        return None

    lat = float(st.session_state.rt_latitud)
    lon = float(st.session_state.rt_longitud)
    if not isfinite(lat) or not isfinite(lon):
        raise ValueError("Las coordenadas deben ser números finitos.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Las coordenadas están fuera del rango permitido.")
    return {"latitud": lat, "longitud": lon}


def _guardar_imagen(archivo: Any) -> Path | None:
    """Valida el archivo y lo coloca temporalmente dentro de assets/."""
    if archivo is None:
        return None

    if archivo.size > IMAGE_MAX_BYTES:
        raise ValueError("La imagen supera el límite de 5 MiB.")

    contenido = archivo.getvalue()
    if not contenido or len(contenido) > IMAGE_MAX_BYTES:
        raise ValueError("La imagen está vacía o supera el tamaño permitido.")

    try:
        with Image.open(BytesIO(contenido)) as imagen:
            formato = imagen.format
            if formato not in IMAGE_ALLOWED_FORMATS:
                raise ValueError("Solo se admiten imágenes JPEG o PNG.")
            if imagen.width * imagen.height > IMAGE_MAX_PIXELS:
                raise ValueError("La imagen supera los 16 megapíxeles.")
            imagen.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("El archivo no es una imagen válida.") from exc

    extension = ".png" if formato == "PNG" else ".jpg"
    directorio = ASSETS_DIR / ".streamlit_temporal"
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{uuid4().hex}{extension}"
    ruta.write_bytes(contenido)
    return ruta


def _limpiar_imagen(ruta: Path | None) -> None:
    """Elimina la copia temporal sin afectar los ejemplos del repositorio."""
    if ruta is not None:
        try:
            ruta.unlink(missing_ok=True)
        except OSError:
            pass


def _datos_geograficos(estado: EstadoProducto) -> list[dict[str, Any]]:
    """Conserva resultados reales de la herramienta geográfica."""
    resultados = []
    for evidencia in estado.get("evidencias", []):
        if evidencia.get("herramienta") != "buscar_puntos_limpios_cercanos":
            continue
        datos = evidencia.get("datos")
        if isinstance(datos, dict):
            resultados.append(deepcopy(datos))
    return resultados


def _registrar_respuesta(estado: EstadoProducto) -> None:
    """Guarda una instantánea presentable del turno."""
    texto = estado.get("respuesta") or ""
    if not texto and estado.get("error"):
        texto = "No se ha podido completar esta consulta."
    st.session_state.rt_mensajes.append({
        "rol": "assistant",
        "texto": texto,
        "plan": deepcopy(estado.get("plan", [])),
        "fuentes": deepcopy(estado.get("fuentes", [])),
        "geografia": _datos_geograficos(estado),
        "limitaciones": deepcopy(estado.get("limitaciones", [])),
        "evidencia": estado.get("estado_evidencia"),
        "abstencion": estado.get("abstencion", False),
        "motivo": estado.get("motivo"),
        "error": estado.get("error"),
        "decision": None,
    })


def _consultar(mensaje: str, archivo: Any) -> None:
    """Envía exactamente un turno al agente y limpia el archivo temporal."""
    estado = _estado()
    if estado["pendiente_hitl"] or estado["done"]:
        st.session_state.rt_aviso = (
            "error", "Resuelve el HITL o inicia una nueva conversación."
        )
        return
    if estado["turnos_consumidos"] >= estado["max_turns"]:
        st.session_state.rt_aviso = (
            "error", "Se ha alcanzado el máximo de turnos del hilo."
        )
        return

    ruta = None
    try:
        coordenadas = _ubicacion()
        ruta = _guardar_imagen(archivo)
        with st.spinner("Analizando la consulta con ReciclaTIA..."):
            resultado = procesar_turno(
                estado,
                mensaje,
                st.session_state.rt_thread_id,
                max_steps=st.session_state.rt_max_steps,
                ubicacion=coordenadas,
                ruta_imagen=str(ruta) if ruta else None,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        st.session_state.rt_aviso = ("error", str(exc))
        return
    except Exception:
        st.session_state.rt_aviso = (
            "error", "Fallo inesperado. Consulta la consola del servidor."
        )
        return
    finally:
        _limpiar_imagen(ruta)

    st.session_state.rt_estado = resultado
    st.session_state.rt_mensajes.append({
        "rol": "user",
        "texto": mensaje,
        "imagen": archivo.name if archivo is not None else None,
    })
    _registrar_respuesta(resultado)
    if archivo is not None:
        st.session_state.rt_version_imagen += 1

    if resultado.get("error"):
        st.session_state.rt_aviso = ("error", resultado["error"])


def _resolver_hitl(aprobado: bool) -> None:
    """Reanuda el checkpoint real y actualiza la decisión visible."""
    estado = _estado()
    if estado.get("error") or not estado.get("pendiente_hitl"):
        st.session_state.rt_aviso = (
            "error", "No existe un plan válido pendiente de decisión."
        )
        return
    try:
        with st.spinner("Registrando la decisión en LangGraph..."):
            resultado = aprobar_plan(
                st.session_state.rt_thread_id,
                aprobado,
                estado,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        st.session_state.rt_aviso = ("error", str(exc))
        return
    except Exception:
        st.session_state.rt_aviso = (
            "error", "No se pudo confirmar la decisión en el servidor."
        )
        return

    st.session_state.rt_estado = resultado
    if resultado.get("error"):
        st.session_state.rt_aviso = ("error", resultado["error"])
        return

    for registro in reversed(st.session_state.rt_mensajes):
        if registro["rol"] == "assistant" and registro["decision"] is None:
            registro["decision"] = (
                "aprobado" if aprobado else "rechazado"
            )
            break

    if aprobado:
        st.session_state.rt_aviso = (
            "success",
            "Plan aprobado y conversación finalizada. La aprobación "
            "no ejecuta comprobaciones adicionales ni acciones físicas.",
        )
    else:
        st.session_state.rt_aviso = (
            "info", "Plan rechazado. Puedes reformular en el mismo hilo."
        )


# =========================================================
# VISUALIZACIÓN DE RESULTADOS
# =========================================================


def _mostrar_fuentes(fuentes: list[dict[str, Any]]) -> None:
    """Presenta referencias documentales verificables del RAG."""
    with st.expander(f"Fuentes documentales ({len(fuentes)})"):
        if not fuentes:
            st.caption("Esta respuesta no contiene citas documentales RAG.")
            return
        for indice, fuente in enumerate(fuentes, start=1):
            identificador = fuente.get("chunk_id") or str(indice)
            nombre = fuente.get("source") or "Fuente sin nombre"
            st.markdown(f"**[{identificador}] {nombre}**")
            if fuente.get("path"):
                st.caption(f"Ruta: {fuente['path']}")


def _mostrar_geografia(resultados: list[dict[str, Any]]) -> None:
    """Presenta tabla, horarios documentados y mapa de la tool."""
    for indice, resultado in enumerate(resultados, start=1):
        puntos = resultado.get("puntos") or []
        procedencia = resultado.get("fuente", "desconocida")
        cantidad = resultado.get("total_encontrados", len(puntos))
        with st.expander(
            f"Puntos limpios · búsqueda {indice} "
            f"({procedencia}; {cantidad} coincidencias)",
            expanded=True,
        ):
            if procedencia == "corpus_local":
                st.warning(
                    "Resultados del corpus local, potencialmente "
                    "desactualizados. No son una consulta en tiempo real."
                )
            else:
                st.caption(
                    "Procedencia geográfica declarada por la herramienta: "
                    f"{procedencia}. Distancias en línea recta."
                )
            if resultado.get("aviso"):
                st.info(str(resultado["aviso"]))
            if not puntos:
                st.write("No se encontraron registros en el radio consultado.")
                continue

            filas = []
            mapa = []
            for punto in puntos:
                if not isinstance(punto, dict):
                    continue
                filas.append({
                    "Tipo": punto.get("tipo") or "—",
                    "Dirección": punto.get("direccion") or "—",
                    "Distancia (m)": punto.get("distancia_metros"),
                    "Día": punto.get("dia_semana") or "—",
                    "Apertura": punto.get("horario_apertura") or "—",
                    "Cierre": punto.get("horario_cierre") or "—",
                    "Turno": punto.get("turno") or "—",
                })
                lat = punto.get("latitud")
                lon = punto.get("longitud")
                if (
                    isinstance(lat, (int, float))
                    and isinstance(lon, (int, float))
                    and isfinite(lat)
                    and isfinite(lon)
                ):
                    mapa.append({"lat": lat, "lon": lon})

            if filas:
                st.dataframe(
                    pd.DataFrame(filas),
                    hide_index=True,
                    use_container_width=True,
                )
            st.caption(
                "Horarios según el registro disponible; no equivalen "
                "a una confirmación de apertura actual."
            )
            if mapa and st.session_state.rt_mostrar_mapa:
                st.map(pd.DataFrame(mapa))


def _mostrar_historial() -> None:
    """Dibuja las consultas y sus instantáneas, sin reejecutar tools."""
    for registro in st.session_state.rt_mensajes:
        rol = registro["rol"]
        with st.chat_message(
            rol,
            avatar="♻️" if rol == "assistant" else "👤",
        ):
            st.markdown(registro["texto"])
            if rol == "user":
                if registro.get("imagen"):
                    st.caption(f"Imagen adjunta: {registro['imagen']}")
                continue

            if registro.get("error"):
                st.error(registro["error"])
            if registro.get("evidencia"):
                abstencion = "sí" if registro.get("abstencion") else "no"
                st.caption(
                    f"Evidencia: {registro['evidencia']} · "
                    f"Abstención: {abstencion}"
                )
            if registro.get("motivo"):
                st.caption(f"Motivo: {registro['motivo']}")
            if registro.get("limitaciones"):
                with st.expander("Limitaciones de la respuesta"):
                    for limitacion in registro["limitaciones"]:
                        st.write(f"• {limitacion}")
            if registro.get("plan"):
                st.markdown("**Plan sujeto a revisión humana**")
                for numero, paso in enumerate(registro["plan"], 1):
                    st.write(f"{numero}. {paso}")
            decision = registro.get("decision")
            if decision == "aprobado":
                st.success("Plan aprobado; hilo finalizado.")
            elif decision == "rechazado":
                st.warning("Plan rechazado; el hilo admite otra consulta.")

            _mostrar_fuentes(registro.get("fuentes") or [])
            if registro.get("geografia"):
                _mostrar_geografia(registro["geografia"])


def _mostrar_diagnostico(estado: EstadoProducto) -> None:
    """Visualiza el estado y la traza ya producidos por el backend."""
    if not st.session_state.rt_mostrar_debug:
        return

    with st.expander("Diagnóstico técnico · LangGraph", expanded=False):
        st.markdown("**Estado del turno**")
        st.json({
            "thread_id": estado["thread_id"],
            "turnos_consumidos": estado["turnos_consumidos"],
            "max_turns": estado["max_turns"],
            "max_steps": estado["max_steps"],
            "tool_calls_turno": estado["tool_calls_turno"],
            "necesita_residuos": estado["necesita_residuos"],
            "necesita_puntos": estado["necesita_puntos"],
            "residuos_completado": estado["residuos_completado"],
            "puntos_completado": estado["puntos_completado"],
            "estado_evidencia": estado["estado_evidencia"],
            "abstencion": estado["abstencion"],
            "pendiente_hitl": estado["pendiente_hitl"],
            "plan_aprobado": estado["plan_aprobado"],
            "done": estado["done"],
            "error": estado["error"],
        })
        st.markdown("**Log del grafo**")
        st.json(estado.get("log") or [])
        st.markdown("**Traza de herramientas**")
        st.json(estado.get("traza_tools") or [])
        st.caption(
            "La traza muestra qué se ejecutó. No equivale "
            "a una auditoría de todas las llamadas internas al SDK."
        )


# =========================================================
# LAYOUT Y ACCIONES
# =========================================================


def _dibujar_sidebar() -> Any:
    """Devuelve el adjunto vigente; los parámetros no alteran config.py."""
    estado = _estado()
    with st.sidebar:
        if estado["pendiente_hitl"] and not estado["error"]:
            st.markdown("### Decisión humana · HITL")
            st.caption("Revisa el plan antes de aprobar o rechazar.")
            aceptar, rechazar = st.columns(2)
            if aceptar.button(
                "Aprobar", key="rt_sidebar_aprobar",
                type="primary", use_container_width=True,
            ):
                _resolver_hitl(True)
                st.rerun()
            if rechazar.button(
                "Rechazar", key="rt_sidebar_rechazar",
                use_container_width=True,
            ):
                _resolver_hitl(False)
                st.rerun()
            st.divider()

        st.markdown("## Sesión")
        st.caption(f"Hilo: `{st.session_state.rt_thread_id[:12]}…`")
        if st.button(
            "Nueva conversación",
            use_container_width=True,
            type="primary",
        ):
            _nueva_conversacion()
            st.rerun()

        st.markdown("### Límites de conversación")
        st.slider(
            "Máximo de turnos",
            min_value=1,
            max_value=MAX_TURNS_LIMIT,
            key="rt_max_turns",
            disabled=estado["turnos_consumidos"] > 0,
            help="Solo se puede cambiar antes del primer mensaje "
            "o al crear una nueva conversación.",
        )
        st.slider(
            "Rondas LLM por especialista",
            min_value=1,
            max_value=MAX_STEPS_LIMIT,
            key="rt_max_steps",
            disabled=estado["pendiente_hitl"] or estado["done"],
            help="Es el límite por especialista para el próximo turno. "
            "Un valor menor reduce exploración y consumo potencial.",
        )
        if estado["turnos_consumidos"] == 0:
            estado["max_turns"] = st.session_state.rt_max_turns

        st.divider()
        st.markdown("### Ubicación voluntaria")
        st.checkbox(
            "Aportar coordenadas",
            key="rt_usar_ubicacion",
            help="No se lee GPS ni se deduce tu ubicación. "
            "Debes habilitar expresamente esta opción.",
        )
        if st.session_state.rt_usar_ubicacion:
            st.number_input(
                "Latitud", min_value=-90.0, max_value=90.0,
                format="%.6f", key="rt_latitud",
            )
            st.number_input(
                "Longitud", min_value=-180.0, max_value=180.0,
                format="%.6f", key="rt_longitud",
            )
            st.caption(
                "Los valores iniciales son solo un ejemplo: centro "
                "de Madrid. Introduce las coordenadas que quieras usar."
            )

        st.divider()
        st.markdown("### Imagen para la siguiente consulta")
        archivo = st.file_uploader(
            "Adjuntar imagen de un residuo",
            type=["jpg", "jpeg", "png"],
            key=f"rt_imagen_{st.session_state.rt_version_imagen}",
            help="JPEG/PNG de hasta 5 MiB y 16 megapíxeles.",
        )
        if archivo is not None:
            st.image(archivo, caption=archivo.name)
            st.caption(
                "El archivo se utiliza en la siguiente consulta; "
                "su copia temporal se elimina al terminar."
            )

        st.divider()
        st.markdown("### Visualización")
        st.checkbox("Mostrar mapas", key="rt_mostrar_mapa")
        st.checkbox("Mostrar depuración", key="rt_mostrar_debug")
        st.caption(
            "Los límites ajustables se envían al orquestador. "
            "No se cambian credenciales, modelos ni reglas de seguridad."
        )
        st.caption(
            "Memoria de conversaciones solo durante la ejecución "
            "del servidor (MemorySaver)."
        )
    return archivo


def main() -> None:
    """Muestra el chat y procesa solamente interacciones explícitas."""
    _iniciar_sesion()
    archivo = _dibujar_sidebar()
    estado = _estado()

    st.markdown(ESTILOS, unsafe_allow_html=True)
    st.markdown(
        '<div class="rt-hero">'
        '<div class="rt-eyebrow">MADRID · RESIDUOS Y PUNTOS LIMPIOS</div>'
        '<h1>♻️ ReciclaTIA</h1>'
        '<p>Asistente multiagente con búsqueda documental, '
        'localización de puntos limpios y revisión humana.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    izquierda, medio, derecha = st.columns(3)
    izquierda.metric(
        "Turnos utilizados",
        f"{estado['turnos_consumidos']}/{estado['max_turns']}",
    )
    medio.metric("Herramientas (turno)", estado["tool_calls_turno"])
    if estado.get("error"):
        situacion = "Error"
    elif estado["pendiente_hitl"]:
        situacion = "Revisión HITL"
    elif estado["done"]:
        situacion = "Finalizado"
    else:
        situacion = "Disponible"
    derecha.metric("Estado", situacion)

    _mostrar_aviso()
    if not st.session_state.rt_mensajes:
        st.markdown("#### ¿Qué necesitas reciclar?")
        st.write(
            "Describe un objeto, pregunta por un punto limpio o "
            "adjunta una fotografía desde la barra lateral."
        )
        columnas = st.columns(3)
        for indice, ejemplo in enumerate(EJEMPLOS):
            if columnas[indice].button(
                ejemplo, key=f"rt_ejemplo_{indice}",
                use_container_width=True,
            ):
                _consultar(ejemplo, archivo)
                st.rerun()

    _mostrar_historial()

    if estado["pendiente_hitl"] and not estado["error"]:
        st.divider()
        st.markdown("### Revisión humana del plan")
        st.info(
            "Aprobar registra la aceptación y finaliza este hilo. "
            "No ejecuta las comprobaciones propuestas ni produce "
            "cambios fuera del sistema. Rechazar permite reformular "
            "tu consulta en el mismo hilo."
        )
        aceptar, rechazar = st.columns(2)
        if aceptar.button(
            "Aprobar plan", type="primary",
            use_container_width=True, key="rt_aprobar",
        ):
            _resolver_hitl(True)
            st.rerun()
        if rechazar.button(
            "Rechazar plan", use_container_width=True,
            key="rt_rechazar",
        ):
            _resolver_hitl(False)
            st.rerun()

    if estado["done"]:
        st.success(
            "Conversación finalizada. Usa «Nueva conversación» "
            "para realizar otra consulta."
        )
    elif estado.get("error") and not estado["pendiente_hitl"]:
        st.warning(
            "Este turno sufrió un fallo técnico. Si el checkpoint "
            "quedó incompleto, inicia una nueva conversación."
        )
    elif estado["turnos_consumidos"] >= estado["max_turns"]:
        st.warning("Límite de turnos alcanzado. Inicia otro hilo.")

    _mostrar_diagnostico(estado)

    bloqueado = (
        estado["pendiente_hitl"]
        or estado["done"]
        or estado["turnos_consumidos"] >= estado["max_turns"]
    )
    entrada = st.chat_input(
        "Escribe tu consulta sobre reciclaje en Madrid...",
        disabled=bloqueado,
        max_chars=500,
    )
    if entrada and entrada.strip():
        _consultar(entrada.strip(), archivo)
        st.rerun()

    st.caption(
        "Información limitada a Madrid ciudad y a las fuentes "
        "disponibles. La proximidad es en línea recta; los horarios "
        "publicados no garantizan apertura real."
    )


if __name__ == "__main__":
    main()
