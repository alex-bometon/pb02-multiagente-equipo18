"""Regresiones de síntesis unificada, horarios y prioridad geográfica."""

from datetime import datetime

from src.agent_resultado import respuesta_geografica_integrada
from src.agent_schedule import estado_horario


DOMINGO = datetime.fromisoformat("2026-10-25T17:45:00+01:00")


def _punto(tipo, direccion, metros, horario, dia=None):
    return {
        "tipo": tipo,
        "direccion": direccion,
        "distancia_metros": metros,
        "horario_apertura": horario,
        "dia_semana": dia,
        "horario_cierre": None,
    }


def _estado(puntos, *, pide="¿Dónde tiro unas pilas alcalinas cerca de mí?", hora=True):
    evidencias = [{
        "rol": "residuos", "herramienta": "consultar_rag", "datos": {
            "respuesta": (
                "Las pilas alcalinas se depositan en el punto limpio de "
                "proximidad [C1], el punto limpio fijo [C2] y el "
                "punto limpio móvil [C3]."
            ),
            "abstencion": False,
            "fuentes": [{"chunk_id": "C1", "source": "residuos.csv"}],
        },
    }, {
        "rol": "puntos", "herramienta": "buscar_puntos_limpios_cercanos",
        "datos": {
            "fuente": "api", "radio_metros": 2000,
            "puntos": puntos[:10], "puntos_candidatos": puntos,
            "total_encontrados": len(puntos),
        },
    }]
    if hora:
        evidencias.append({
            "rol": "puntos", "herramienta": "hora_actual",
            "datos": "2026-10-25T17:45:00+01:00",
        })
    return {
        "pedido": pide, "necesita_residuos": True,
        "necesita_puntos": True, "evidencias": evidencias,
    }


def test_24_horas_documentadas_no_quedan_desconocidas():
    punto = _punto("Proximidad", "Calle Alameda", 1000, "Todos los días 24 horas.")
    assert estado_horario(punto, DOMINGO) == "dentro"


def test_domingo_cerrado_para_sabado():
    punto = _punto("Móvil", "Plaza Mayor", 287, "Sábado de 10:30 a 12:00 h.", "SABADO")
    assert estado_horario(punto, DOMINGO) == "fuera"


def test_domingo_cerrado_horario_compuesto_de_lunes_a_sabado():
    horario = (
        "Lunes a viernes de 8:30 a 21:30 h. "
        "Sábados 10:00 a 15:00 h. y de 16:00 a 21:30 h."
    )
    punto = _punto("Proximidad", "Calle Casino", 1200, horario)
    assert estado_horario(punto, DOMINGO) == "fuera"


def test_sabado_dentro_segunda_franja_horario_compuesto():
    horario = (
        "Lunes a viernes de 8:30 a 21:30 h. "
        "Sábados 10:00 a 15:00 h. y de 16:00 a 21:30 h."
    )
    instante = datetime.fromisoformat("2026-10-24T17:45:00+02:00")
    punto = _punto("Proximidad", "Calle Casino", 1200, horario)
    assert estado_horario(punto, instante) == "dentro"


def test_fuera_de_horario_se_prioriza_abierto_mas_lejano():
    puntos = [
        _punto("Móvil", "Plaza Mayor", 287, "Sábado de 10:30 a 12:00 h.", "SABADO"),
        _punto("Proximidad", "Calle Alameda", 1011, "Todos los días 24 horas."),
        _punto("Proximidad", "Plaza Nueva", 1400, "Todos los días 24 horas."),
    ]
    respuesta, plan = respuesta_geografica_integrada(_estado(puntos))
    assert "Calle Alameda" in respuesta
    assert "Plaza Mayor" in respuesta
    assert respuesta.index("Calle Alameda") < respuesta.index("Plaza Mayor")
    assert "Dentro del horario publicado" in respuesta
    assert "Fuera del horario publicado" in respuesta
    assert len(plan) >= 2


def test_orden_explicito_por_distancia_invierte_prioridad():
    puntos = [
        _punto("Móvil", "Plaza Mayor", 287, "Sábado de 10:30 a 12:00 h.", "SABADO"),
        _punto("Proximidad", "Calle Alameda", 1011, "Todos los días 24 horas."),
    ]
    estado = _estado(puntos, pide="¿Cuáles son los más cercanos para tirar pilas?")
    respuesta, _ = respuesta_geografica_integrada(estado)
    assert respuesta.index("Plaza Mayor") < respuesta.index("Calle Alameda")


def test_sin_hora_no_se_declara_abierto():
    punto = _punto("Proximidad", "Calle Alameda", 1000, "Todos los días 24 horas.")
    respuesta, _ = respuesta_geografica_integrada(_estado([punto], hora=False))
    assert "No se pudo obtener la hora actual" in respuesta
    assert "Horario no comprobable" in respuesta


def test_sin_clasificacion_no_se_certifica_aceptacion():
    punto = _punto("Proximidad", "Calle Alameda", 1000, "Todos los días 24 horas.")
    estado = _estado([punto])
    estado["evidencias"][0]["datos"]["abstencion"] = True
    assert respuesta_geografica_integrada(estado) is None


def test_no_se_repite_misma_instalacion_por_varios_dias():
    puntos = [
        _punto("Móvil", "Calle Norte", 300, "Sábado de 10:30 a 12:00 h.", "SABADO"),
        _punto("Móvil", "Calle Norte", 300, "Domingo de 17:00 a 19:00 h.", "DOMINGO"),
    ]
    respuesta, _ = respuesta_geografica_integrada(_estado(puntos))
    assert respuesta.count("Móvil — Calle Norte") == 1
    assert "Dentro del horario publicado" in respuesta


def test_no_se_incluyen_modalidades_incompatibles():
    puntos = [
        _punto("Reutilización", "Otra zona", 100, "Todos los días 24 horas."),
        _punto("Fijo", "Centro válido", 500, "Todos los días 24 horas."),
    ]
    respuesta, _ = respuesta_geografica_integrada(_estado(puntos))
    assert "Otra zona" not in respuesta
    assert "Centro válido" in respuesta


def test_proximidad_24h_no_se_confunde_con_movil():
    punto = _punto("Proximidad 24h", "Calle Alameda", 1000, "Todos los días 24 horas.")
    respuesta, _ = respuesta_geografica_integrada(_estado([punto]))
    assert "Calle Alameda" in respuesta
    assert "Dentro del horario publicado" in respuesta
