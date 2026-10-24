"""Pruebas de regresión del sistema multiagente ReciclaTIA.

Se comprueban la orquestación, las herramientas, los límites y el HITL
sin utilizar Gemini, la API municipal ni otros servicios externos.
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from config import (
    GEO_DEFAULT_RADIUS_METERS,
    MAX_ITER,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TURNS_DEFAULT,
)
from src.agent import aprobar_plan, nuevo_thread_id, procesar_turno
from src.agent_routing import (
    requiere_llm_supervisor,
    resolver_enrutamiento,
)
from src.agent_tool_loop import ejecutar_especialista
from src.graph import build_graph
from src.nodes import supervisor
from src.state import crear_estado_grafo, crear_estado_producto
from src.tools import ejecutar_tool_estructurado


# =========================================================
# DOBLES DE PRUEBA
# =========================================================


class ModeloFalso:
    """Emula respuestas de LangChain sin inicializar Gemini."""

    def __init__(self, respuestas: list[AIMessage]) -> None:
        self._respuestas = iter(respuestas)
        self.herramientas: list[str] = []

    def bind_tools(self, herramientas):
        """Registra las herramientas disponibles para el especialista."""
        self.herramientas = [tool.name for tool in herramientas]
        return self

    def invoke(self, mensajes):
        """Devuelve la siguiente respuesta preparada."""
        assert mensajes
        return next(self._respuestas)


def _llamada(
    nombre: str,
    argumentos: dict,
    identificador: str,
) -> AIMessage:
    """Construye una solicitud de herramienta de LangChain."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": nombre,
            "args": argumentos,
            "id": identificador,
        }],
    )


def _resultado_rag(*, abstencion: bool = False) -> dict:
    """Simula la respuesta estructurada del RAG PB01."""
    if abstencion:
        return {
            "respuesta": "No hay evidencia documental suficiente.",
            "abstencion": True,
            "motivo": "evidencia_insuficiente",
            "fuentes": [],
        }

    return {
        "respuesta": "Los envases van al contenedor amarillo. [C1]",
        "abstencion": False,
        "motivo": None,
        "fuentes": [{
            "chunk_id": "C1",
            "source": "residuos.csv",
            "path": "data/raw/csv/residuos.csv",
        }],
    }


def _resultado_geografico() -> dict:
    """Simula la respuesta estructurada de la API geográfica."""
    return {
        "fuente": "api",
        "tipo_solicitado": "proximidad",
        "radio_metros": GEO_DEFAULT_RADIUS_METERS,
        "total_encontrados": 1,
        "puntos": [{
            "tipo": "Proximidad",
            "direccion": "Calle de Prueba, 1",
            "distancia_metros": 350.0,
        }],
        "aviso": None,
    }


def _crear_conversacion(
    *,
    max_turns: int = MAX_TURNS_DEFAULT,
):
    """Crea una conversación con identificador independiente."""
    thread_id = nuevo_thread_id()
    estado = crear_estado_producto(
        thread_id,
        max_turns=max_turns,
    )
    return estado, thread_id


def _ejecutor_simulado(nombre, args, *, rol):
    """Simula resultados RAG, geográficos y visuales."""
    assert rol in {"residuos", "puntos"}

    if nombre == "consultar_rag":
        assert isinstance(args["pregunta"], str)
        datos = _resultado_rag()

        if rol == "puntos":
            datos["respuesta"] = (
                "Los puntos limpios de proximidad son "
                "instalaciones municipales. [C1]"
            )
            datos["fuentes"][0]["source"] = "puntos.csv"

    elif nombre == "buscar_puntos_limpios_cercanos":
        assert isinstance(args["latitud"], float)
        assert isinstance(args["longitud"], float)
        datos = _resultado_geografico()

    elif nombre == "describir_imagen":
        assert args["ruta_imagen"] == "assets/imagen_demo.jpg"
        datos = "Una botella de plástico transparente."

    else:
        raise AssertionError(f"Herramienta no prevista: {nombre}")

    return {
        "ok": True,
        "nombre": nombre,
        "datos": deepcopy(datos),
        "error": None,
    }


# =========================================================
# AISLAMIENTO DE DEPENDENCIAS
# =========================================================


@pytest.fixture
def aislado(monkeypatch):
    """Aísla MemorySaver y bloquea llamadas externas no simuladas."""
    app = build_graph()

    monkeypatch.setattr("src.agent.get_app", lambda: app)
    monkeypatch.setattr("src.graph.get_app", lambda: app)

    def prohibido(*args, **kwargs):
        raise AssertionError(
            "No se permiten llamadas externas en estos tests."
        )

    monkeypatch.setattr("src.nodes.get_llm", prohibido)
    monkeypatch.setattr("src.agent_tool_loop.get_llm", prohibido)
    monkeypatch.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        prohibido,
    )

    return monkeypatch


# =========================================================
# CLASIFICACIÓN Y HITL
# =========================================================


def test_clasificacion_directa_y_aprobacion(aislado):
    """El RAG directo alcanza HITL sin LLM adicional."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una botella de plástico?",
        hilo,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["done"] is False
    assert estado["respuesta"] == _resultado_rag()["respuesta"]
    assert estado["estado_evidencia"] == "suficiente"

    decisiones_directas = [
        evento
        for evento in estado["log"]
        if evento["accion"] == "ruta_directa"
    ]

    assert len(decisiones_directas) == 3
    herramienta.assert_called_once()
    assert herramienta.call_args.args[0] == "consultar_rag"
    assert len(estado["fuentes"]) == 1

    estado = aprobar_plan(hilo, True, estado)

    assert estado["done"] is True
    assert estado["pendiente_hitl"] is False
    assert estado["historial_turnos"][-1]["aprobado"] is True


def test_rechazo_y_reformulacion(aislado):
    """El rechazo permite una nueva consulta en el mismo hilo."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una botella?",
        hilo,
    )

    estado = aprobar_plan(hilo, False, estado)

    assert estado["done"] is False
    assert estado["pendiente_hitl"] is False
    assert estado["plan"] == []
    assert estado["historial_turnos"][-1]["aprobado"] is False

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una lata de refresco?",
        hilo,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["turnos_consumidos"] == 2
    assert herramienta.call_count == 2


def test_no_se_admiten_turnos_mientras_hitl(aislado):
    """Impide realizar otra consulta antes de decidir sobre el plan."""
    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        _ejecutor_simulado,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una botella?",
        hilo,
    )

    with pytest.raises(ValueError, match="pendiente"):
        procesar_turno(
            estado,
            "¿Dónde tiro una lata?",
            hilo,
        )


def test_limite_turnos_tras_rechazo(aislado):
    """El rechazo no permite superar max_turns."""
    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        _ejecutor_simulado,
    )

    estado, hilo = _crear_conversacion(max_turns=1)

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una botella?",
        hilo,
    )

    estado = aprobar_plan(hilo, False, estado)

    with pytest.raises(ValueError, match="límite de turnos"):
        procesar_turno(
            estado,
            "¿Dónde tiro una lata?",
            hilo,
        )


# =========================================================
# CONSULTAS DE PUNTOS LIMPIOS
# =========================================================


def test_punto_de_proximidad_documental(aislado):
    """La modalidad 'proximidad' no exige geolocalización."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Qué es un punto limpio de proximidad?",
        hilo,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["estado_evidencia"] == "suficiente"
    assert "puntos limpios" in estado["respuesta"].lower()
    assert "latitud" not in estado["respuesta"].lower()

    herramienta.assert_called_once()
    assert herramienta.call_args.args[0] == "consultar_rag"
    assert herramienta.call_args.kwargs["rol"] == "puntos"


def test_punto_cercano_sin_coordenadas(aislado):
    """Solicita coordenadas sin llamar a ninguna herramienta."""
    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "Busca un punto limpio de proximidad cerca de mí",
        hilo,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["estado_evidencia"] == "insuficiente"
    assert "latitud" in estado["respuesta"].lower()
    assert "longitud" in estado["respuesta"].lower()
    assert estado["tool_calls_turno"] == 0
    assert estado["evidencias"] == []


def test_punto_cercano_con_coordenadas(aislado):
    """Ejecuta una consulta geográfica con coordenadas aportadas."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    estado, hilo = _crear_conversacion()

    ubicacion = {
        "latitud": 40.4168,
        "longitud": -3.7038,
    }

    estado = procesar_turno(
        estado,
        "Busca un punto limpio de proximidad cerca de mí",
        hilo,
        ubicacion=ubicacion,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["estado_evidencia"] == "suficiente"
    assert "Calle de Prueba, 1" in estado["respuesta"]

    herramienta.assert_called_once()

    assert herramienta.call_args.args[0] == (
        "buscar_puntos_limpios_cercanos"
    )

    argumentos = herramienta.call_args.args[1]

    assert argumentos["latitud"] == 40.4168
    assert argumentos["longitud"] == -3.7038
    assert argumentos["tipo"] == "proximidad"


# =========================================================
# ABSTENCIÓN Y ERRORES
# =========================================================


def test_abstencion_rag_sin_generacion_adicional(aislado):
    """Una abstención del RAG no se convierte en una afirmación."""
    def rag_sin_evidencia(nombre, args, *, rol):
        assert nombre == "consultar_rag"

        return {
            "ok": True,
            "nombre": nombre,
            "datos": _resultado_rag(abstencion=True),
            "error": None,
        }

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        rag_sin_evidencia,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro esto?",
        hilo,
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["abstencion"] is True
    assert estado["estado_evidencia"] == "insuficiente"
    assert estado["fuentes"] == []
    assert "No hay evidencia" in estado["respuesta"]


def test_fallo_tecnico_no_inventa_resultado(aislado):
    """Un fallo externo no produce una clasificación inventada."""
    def herramienta_fallida(nombre, args, *, rol):
        return {
            "ok": False,
            "nombre": nombre,
            "datos": None,
            "error": "Error: servicio no disponible.",
        }

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta_fallida,
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una botella?",
        hilo,
    )

    assert estado["error"] is not None
    assert estado["pendiente_hitl"] is False
    assert estado["plan"] == []
    assert estado["done"] is False


# =========================================================
# CONSULTA COMPUESTA
# =========================================================


def test_consulta_compuesta_con_dos_especialistas(aislado):
    """Ejecuta ambos especialistas en secuencia y conserva pruebas."""
    herramientas = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramientas,
    )

    modelos = [
        ModeloFalso([
            _llamada(
                "consultar_rag",
                {"pregunta": "¿Dónde tiro una batería?"},
                "residuo-1",
            ),
        ]),
        ModeloFalso([
            _llamada(
                "buscar_puntos_limpios_cercanos",
                {"tipo": "todos"},
                "puntos-1",
            ),
        ]),
    ]

    creador = Mock(side_effect=modelos)

    aislado.setattr(
        "src.agent_tool_loop.get_llm",
        creador,
    )

    sintesis = Mock(return_value={
        "respuesta": (
            "Clasificación documentada [C1] y punto encontrado."
        ),
        "plan": [
            "Consultar las fuentes.",
            "Comprobar los datos del punto limpio.",
        ],
    })

    aislado.setattr("src.nodes._generar_json", sintesis)

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿Dónde tiro una batería y qué punto limpio tengo cerca?",
        hilo,
        ubicacion={
            "latitud": 40.4168,
            "longitud": -3.7038,
        },
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True
    assert estado["necesita_residuos"] is True
    assert estado["necesita_puntos"] is True
    assert estado["residuos_completado"] is True
    assert estado["puntos_completado"] is True

    nombres_ejecutados = [
        llamada.args[0]
        for llamada in herramientas.call_args_list
    ]

    assert nombres_ejecutados == [
        "consultar_rag",
        "buscar_puntos_limpios_cercanos",
    ]

    assert creador.call_count == 2
    sintesis.assert_called_once()
    assert len(estado["evidencias"]) == 2
    assert estado["tool_calls_turno"] == 2
    assert estado["fuentes"]


# =========================================================
# MULTIMODALIDAD
# =========================================================


def test_imagen_y_rag_en_el_mismo_especialista(aislado):
    """Describe una imagen antes de solicitar su clasificación RAG."""
    herramientas = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramientas,
    )

    modelo = ModeloFalso([
        _llamada(
            "describir_imagen",
            {},
            "imagen-1",
        ),
        _llamada(
            "consultar_rag",
            {"pregunta": "Botella de plástico transparente"},
            "rag-1",
        ),
    ])

    aislado.setattr(
        "src.agent_tool_loop.get_llm",
        Mock(return_value=modelo),
    )

    aislado.setattr(
        "src.nodes._generar_json",
        Mock(return_value={
            "respuesta": "Según el RAG, envase amarillo. [C1]",
            "plan": [
                "Comprobar el material.",
                "Seguir el destino documentado.",
            ],
        }),
    )

    estado, hilo = _crear_conversacion()

    estado = procesar_turno(
        estado,
        "¿En qué contenedor va lo que aparece en esta foto?",
        hilo,
        ruta_imagen="assets/imagen_demo.jpg",
    )

    assert estado["error"] is None
    assert estado["pendiente_hitl"] is True

    herramientas_ejecutadas = [
        evidencia["herramienta"]
        for evidencia in estado["evidencias"]
    ]

    assert herramientas_ejecutadas == [
        "describir_imagen",
        "consultar_rag",
    ]

    assert len(estado["fuentes"]) == 1
    assert estado["tool_calls_turno"] == 2

    assert modelo.herramientas == [
        "consultar_rag",
        "describir_imagen",
    ]


# =========================================================
# PRESUPUESTOS Y DEDUPLICACIÓN
# =========================================================


def test_presupuesto_de_tools_aplicado_en_modo_directo(aislado):
    """Un presupuesto agotado bloquea rutas deterministas."""
    estado = crear_estado_grafo(
        "¿Dónde tiro una botella?"
    )

    estado["tool_calls_turno"] = MAX_TOOL_CALLS_PER_TURN

    resultado = ejecutar_especialista(
        estado,
        rol="residuos",
    )

    assert (
        resultado["tool_calls_turno"]
        == MAX_TOOL_CALLS_PER_TURN
    )
    assert resultado["evidencias"] == []
    assert resultado["traza_tools"][-1]["status"] == "error"
    assert "presupuesto" in resultado["limitaciones"][-1]


def test_deduplicacion_de_herramientas(aislado):
    """Dos llamadas idénticas ejecutan realmente la tool una vez."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    mensaje = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "consultar_rag",
                "args": {"pregunta": "botella de plástico"},
                "id": "primera",
            },
            {
                "name": "consultar_rag",
                "args": {"pregunta": "botella de plástico"},
                "id": "segunda",
            },
        ],
    )

    modelo = ModeloFalso([mensaje])

    aislado.setattr(
        "src.agent_tool_loop.get_llm",
        Mock(return_value=modelo),
    )

    estado = crear_estado_grafo(
        "¿Dónde tiro una botella?"
    )

    estado["contexto_revision"] = "Plan rechazado previamente."

    resultado = ejecutar_especialista(
        estado,
        rol="residuos",
    )

    herramienta.assert_called_once()
    assert resultado["tool_calls_turno"] == 1
    assert len(resultado["evidencias"]) == 1

    estados_traza = [
        traza["status"]
        for traza in resultado["traza_tools"]
    ]

    assert estados_traza == ["ok", "cache"]


def test_limite_de_rondas_del_especialista(aislado):
    """El bucle agente no rebasa max_steps."""
    herramienta = Mock(side_effect=_ejecutor_simulado)

    aislado.setattr(
        "src.agent_tool_loop.ejecutar_tool_estructurado",
        herramienta,
    )

    modelo = ModeloFalso([
        _llamada(
            "describir_imagen",
            {},
            "imagen-unica",
        ),
    ])

    aislado.setattr(
        "src.agent_tool_loop.get_llm",
        Mock(return_value=modelo),
    )

    estado = crear_estado_grafo(
        "¿En qué contenedor va lo que aparece en esta foto?",
        max_steps=1,
        ruta_imagen="assets/imagen_demo.jpg",
    )

    resultado = ejecutar_especialista(
        estado,
        rol="residuos",
    )

    assert resultado["tool_calls_turno"] == 1
    assert resultado["evidencias"][0]["herramienta"] == (
        "describir_imagen"
    )
    assert "Límite de rondas" in resultado["limitaciones"][-1]

    herramienta.assert_called_once()


def test_allowlist_rol_residuos():
    """El especialista de residuos no puede utilizar la API de puntos."""
    resultado = ejecutar_tool_estructurado(
        "buscar_puntos_limpios_cercanos",
        {
            "latitud": 40.4168,
            "longitud": -3.7038,
        },
        rol="residuos",
    )

    assert resultado["ok"] is False
    assert "no permitida" in resultado["error"]


# =========================================================
# SUPERVISOR Y LÍMITES DEL GRAFO
# =========================================================


def test_supervisor_ambiguo_consulta_modelo_una_vez(aislado):
    """Una intención desconocida admite una propuesta del modelo."""
    estado = crear_estado_grafo(
        "¿Dónde llevo mis pilas?"
    )

    assert requiere_llm_supervisor(estado) is True

    propuesta = Mock(return_value={
        "necesita_residuos": True,
        "necesita_puntos": False,
        "siguiente": "residuos",
    })

    aislado.setattr(
        "src.nodes._generar_json",
        propuesta,
    )

    resultado = supervisor(estado)

    assert resultado["siguiente"] == "residuos"
    assert resultado["necesita_residuos"] is True

    propuesta.assert_called_once()


def test_max_iter_bloquea_nuevas_decisiones():
    """El supervisor respeta MAX_ITER configurado en Python."""
    estado = crear_estado_grafo(
        "¿Dónde tiro una botella?"
    )

    estado["iteracion"] = MAX_ITER

    with pytest.raises(RuntimeError, match="MAX_ITER"):
        resolver_enrutamiento(estado)
