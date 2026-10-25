"""Pruebas de integración del agente con herramientas simuladas (sin red)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.agent import aprobar_plan, nuevo_thread_id, procesar_turno
from src.graph import build_graph
from src.state import crear_estado_producto


@pytest.fixture
def entorno(monkeypatch):
    """Crea checkpoints independientes y prohíbe Gemini y redes reales."""
    grafo = build_graph()
    monkeypatch.setattr("src.agent.get_app", lambda: grafo)
    monkeypatch.setattr("src.graph.get_app", lambda: grafo)

    def gemini_bloqueado(*args, **kwargs):
        raise AssertionError("No debe llamarse Gemini para elegir herramientas")

    monkeypatch.setattr("src.agent_tool_loop.get_llm", gemini_bloqueado)
    monkeypatch.setattr("src.nodes.get_llm", gemini_bloqueado)
    return monkeypatch


def _rag(texto):
    return {
        "abstencion": False,
        "respuesta": texto,
        "motivo": None,
        "fuentes": [{"chunk_id": "C1", "source": "catalogo.csv"}],
    }


def _ejecutar_hilo(mensaje, *, ubicacion=None):
    hilo = nuevo_thread_id()
    estado = crear_estado_producto(hilo)
    return procesar_turno(estado, mensaje, hilo, ubicacion=ubicacion), hilo


def test_vidrio_consulta_rag_y_hora_y_pausa_hitl(entorno):
    llamadas = []

    def herramienta(nombre, args, *, rol):
        llamadas.append((nombre, rol))
        if nombre == "consultar_rag":
            dato = _rag("Botella de vidrio: contenedor verde. [C1]")
        elif nombre == "hora_actual":
            dato = "2026-10-25T23:10:00+01:00"
        else:
            raise AssertionError(f"Herramienta inesperada: {nombre}")
        return {"ok": True, "nombre": nombre, "datos": dato, "error": None}

    entorno.setattr("src.agent_tool_loop.ejecutar_tool_estructurado", herramienta)
    estado, hilo = _ejecutar_hilo("¿Dónde tiro una botella de vidrio?")

    assert estado["error"] is None
    assert estado["pendiente_hitl"] and not estado["done"]
    assert llamadas == [("consultar_rag", "residuos"), ("hora_actual", "residuos")]
    assert "NO está permitido" in estado["respuesta"]
    assert estado["tool_calls_turno"] == 2
    final = aprobar_plan(hilo, True, estado)
    assert final["done"] and not final["pendiente_hitl"]


def test_plastico_no_consume_hora_ni_geo(entorno):
    llamadas = []

    def herramienta(nombre, args, *, rol):
        llamadas.append(nombre)
        assert nombre == "consultar_rag"
        return {
            "ok": True, "nombre": nombre,
            "datos": _rag("La botella de plástico va al contenedor amarillo. [C1]"),
            "error": None,
        }

    entorno.setattr("src.agent_tool_loop.ejecutar_tool_estructurado", herramienta)
    estado, _ = _ejecutar_hilo("¿Dónde tiro una botella de plástico?")
    assert estado["error"] is None
    assert estado["pendiente_hitl"]
    assert llamadas == ["consultar_rag"]


def test_pilas_con_ubicacion_activan_puntos_y_hora(entorno):
    llamadas = []

    def herramienta(nombre, args, *, rol):
        llamadas.append((nombre, rol))
        if nombre == "consultar_rag":
            dato = _rag("Las pilas domésticas deben ir al punto limpio. [C1]")
        elif nombre == "buscar_puntos_limpios_cercanos":
            dato = {
                "fuente": "api", "tipo_solicitado": "todos",
                "radio_metros": 2000, "total_encontrados": 1,
                "puntos": [
                    {"tipo": "Fijo", "direccion": "Calle Ejemplo, 1",
                     "distancia_metros": 450.0,
                     "dia_semana": "lunes a viernes",
                     "horario_apertura": "09:00",
                     "horario_cierre": "18:00"}
                ],
                "aviso": None,
            }
        elif nombre == "hora_actual":
            dato = "2026-10-26T10:30:00+01:00"
        else:
            raise AssertionError(f"Herramienta inesperada: {nombre}")
        return {"ok": True, "nombre": nombre, "datos": dato, "error": None}

    entorno.setattr("src.agent_tool_loop.ejecutar_tool_estructurado", herramienta)
    entorno.setattr(
        "src.nodes._generar_json",
        Mock(return_value={
            "respuesta": "Pilas domésticas: punto limpio. Calle Ejemplo, 1.",
            "plan": ["Comprobar el tipo de pila", "Acudir en horario vigente"],
        }),
    )
    ubicacion = {"latitud": 40.4168, "longitud": -3.7038}
    estado, _ = _ejecutar_hilo("¿Dónde tiro las pilas usadas?", ubicacion=ubicacion)

    assert estado["error"] is None
    assert estado["necesita_residuos"] and estado["necesita_puntos"]
    assert estado["pendiente_hitl"]
    assert [nombre for nombre, _ in llamadas] == [
        "consultar_rag", "buscar_puntos_limpios_cercanos", "hora_actual"
    ]
    assert "Dentro del horario publicado" in estado["respuesta"]
    assert "450 m" in estado["respuesta"]


def test_pilas_sin_ubicacion_no_busca_api(entorno):
    llamadas = []

    def herramienta(nombre, args, *, rol):
        llamadas.append(nombre)
        assert nombre == "consultar_rag"
        return {"ok": True, "nombre": nombre,
                "datos": _rag("Las pilas domésticas van al punto limpio. [C1]"),
                "error": None}

    entorno.setattr("src.agent_tool_loop.ejecutar_tool_estructurado", herramienta)
    estado, _ = _ejecutar_hilo("¿Dónde tiro las pilas usadas?")
    assert estado["error"] is None
    assert estado["pendiente_hitl"]
    assert llamadas == ["consultar_rag"]
