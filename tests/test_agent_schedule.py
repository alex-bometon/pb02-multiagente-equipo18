"""Pruebas de las decisiones temporales y geográficas, sin red ni Gemini."""

from src.agent_schedule import (
    _dias_permitidos,
    _situacion_punto,
    anexos_temporales,
    debe_buscar_puntos_automaticamente,
    es_vidrio_envase_confirmado,
    requiere_hora,
    tipo_punto_documentado,
)
from datetime import datetime


def _base(pedido="¿Dónde tiro una botella de vidrio?", ubicacion=None):
    return {"pedido": pedido, "ubicacion": ubicacion, "evidencias": []}


def _rag(estado, respuesta, *, abstencion=False):
    estado["evidencias"].append({
        "rol": "residuos",
        "herramienta": "consultar_rag",
        "datos": {
            "abstencion": abstencion,
            "respuesta": respuesta,
            "fuentes": [{"chunk_id": "C1", "source": "datos.csv"}],
        },
    })
    return estado


def _hora(estado, valor):
    estado["evidencias"].append({
        "rol": "residuos", "herramienta": "hora_actual", "datos": valor,
    })
    return estado


def test_vidrio_hora_permitida_y_fuente_oficial():
    estado = _hora(_rag(_base(), "Envases de vidrio al contenedor verde [C1]."), "2026-10-25T17:01:00+01:00")
    assert es_vidrio_envase_confirmado(estado)
    texto, fuentes = anexos_temporales(estado)
    assert "dentro del horario permitido" in texto
    assert "NORMA-VIDRIO" in [f["chunk_id"] for f in fuentes]


def test_vidrio_noche_fuera_de_horario():
    estado = _hora(_rag(_base(), "Envases de vidrio al contenedor verde [C1]."), "2026-10-25T23:01:00+01:00")
    texto, _ = anexos_temporales(estado)
    assert "NO está permitido" in texto
    assert "08:00–22:30" in texto


def test_vidrio_sin_hora_no_finge_disponibilidad():
    estado = _rag(_base(), "Envases de vidrio al contenedor verde [C1].")
    texto, _ = anexos_temporales(estado)
    assert "No se pudo verificar la hora actual" in texto
    assert requiere_hora(estado, "residuos")


def test_plastico_no_necesita_hora():
    estado = _rag(_base("¿Dónde tiro una botella de plástico?"), "Envases de plástico al contenedor amarillo [C1].")
    assert not requiere_hora(estado, "residuos")
    assert anexos_temporales(estado) == ("", [])


def test_vaso_de_cristal_no_es_contenedor_verde():
    estado = _rag(_base("¿Dónde tiro un vaso de cristal?"), "Los vasos de cristal deben ir al punto limpio [C1].")
    assert not es_vidrio_envase_confirmado(estado)


def test_geo_auto_requiere_residuo_documentado_y_coordenadas():
    ubicacion = {"latitud": 40.4168, "longitud": -3.7038}
    estado = _rag(_base("¿Dónde tiro pilas usadas?", ubicacion), "Las pilas domésticas se depositan en un punto limpio [C1].")
    assert debe_buscar_puntos_automaticamente(estado)
    estado["ubicacion"] = None
    assert not debe_buscar_puntos_automaticamente(estado)


def test_no_busca_geo_para_contenedor_comun():
    ubicacion = {"latitud": 40.4168, "longitud": -3.7038}
    estado = _rag(_base("¿Dónde tiro el papel?", ubicacion), "El papel se deposita en el contenedor azul [C1].")
    assert not debe_buscar_puntos_automaticamente(estado)


def test_no_activa_geo_por_abstencion():
    ubicacion = {"latitud": 40.4168, "longitud": -3.7038}
    estado = _rag(_base("¿Dónde tiro esto?", ubicacion), "No sé si requiere punto limpio.", abstencion=True)
    assert not debe_buscar_puntos_automaticamente(estado)


def test_geo_con_horario_comprobable():
    estado = _base("Busca puntos limpios cerca de mí", {"latitud": 40.4, "longitud": -3.7})
    estado["evidencias"] = [
        {
            "rol": "puntos", "herramienta": "buscar_puntos_limpios_cercanos",
            "datos": {
                "fuente": "api", "puntos": [{
                    "direccion": "Calle Ejemplo, 1", "distancia_metros": 333,
                    "horario_apertura": "09:00", "horario_cierre": "18:00",
                    "dia_semana": "lunes a viernes",
                }],
            },
        },
        {"rol": "puntos", "herramienta": "hora_actual", "datos": "2026-10-26T10:00:00+01:00"},
    ]
    assert requiere_hora(estado, "puntos")
    texto, _ = anexos_temporales(estado)
    assert "Dentro del horario publicado" in texto
    assert "333 m" in texto


def test_geo_dia_cerrado_y_horario_desconocido():
    lunes = datetime.fromisoformat("2026-10-26T10:00:00+01:00")
    assert "Fuera del horario" in _situacion_punto({
        "horario_apertura": "09:00", "horario_cierre": "18:00",
        "dia_semana": "sábado",
    }, lunes)
    assert "No se puede determinar" in _situacion_punto({
        "horario_apertura": "9 a 18", "horario_cierre": None,
        "dia_semana": "lunes",
    }, lunes)
    assert _dias_permitidos("horarios varios") is None


def test_rag_horario_documental_movil_con_citas():
    estado = _base("¿Qué horario tiene el punto limpio móvil de Calle Ejemplo?")
    estado["evidencias"] = [{
        "rol": "puntos", "herramienta": "consultar_rag",
        "datos": {
            "abstencion": False,
            "fuentes": [{"chunk_id": "C1", "source": "paradas.csv"}],
            "chunks": [{
                "text": (
                    "Dirección: Calle Ejemplo, 1\n"
                    "Día: lunes\n"
                    "Horario: 09:00 a 14:00\n"
                ),
                "metadata": {"source": "paradas.csv"},
            }],
        },
    }, {
        "rol": "puntos", "herramienta": "hora_actual",
        "datos": "2026-10-26T10:00:00+01:00",
    }]
    texto, _ = anexos_temporales(estado)
    assert "Dentro del horario publicado" in texto
    assert "Calle Ejemplo, 1" in texto


def test_rag_horario_no_evaluable_sin_hora_completa():
    estado = _base("¿Qué horario tiene el punto limpio móvil de Calle Ejemplo?")
    estado["evidencias"] = [{
        "rol": "puntos", "herramienta": "consultar_rag",
        "datos": {
            "abstencion": False,
            "fuentes": [{"chunk_id": "C1", "source": "paradas.csv"}],
            "chunks": [{
                "text": "Dirección: Calle Ejemplo, 1\nDía: lunes\nHorario: mañanas",
            }],
        },
    }, {
        "rol": "puntos", "herramienta": "hora_actual",
        "datos": "2026-10-26T10:00:00+01:00",
    }]
    texto, _ = anexos_temporales(estado)
    assert "no interpretable" in texto
    assert "Dentro del horario publicado" not in texto


def test_intervalo_de_dias_y_plurales():
    assert _dias_permitidos("De lunes a sábado") == set(range(6))
    assert _dias_permitidos("sábados y domingos") == {5, 6}
    assert _dias_permitidos("lunes a viernes") == set(range(5))


def test_ventana_vidrio_limites_exactos():
    estado = _rag(_base(), "Botella de vidrio al contenedor verde [C1].")
    _hora(estado, "2026-10-25T22:29:00+01:00")
    texto, _ = anexos_temporales(estado)
    assert "dentro del horario permitido" in texto
    estado["evidencias"][-1]["datos"] = "2026-10-25T22:30:00+01:00"
    texto, _ = anexos_temporales(estado)
    assert "NO está permitido" in texto


def test_modalidad_compatible_documentada():
    estado = _rag(_base("¿Dónde tiro mis pilas?"),
                  "Las pilas se depositan en el punto limpio móvil [C1].")
    assert tipo_punto_documentado(estado) == "movil"
    estado["evidencias"][0]["datos"]["respuesta"] = (
        "Punto limpio fijo o punto limpio móvil [C1]."
    )
    assert tipo_punto_documentado(estado) == "todos"
