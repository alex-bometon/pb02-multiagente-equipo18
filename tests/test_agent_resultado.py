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


def test_respuesta_util_elige_una_abierta_y_pocos_cercanos():
    """La mejor abierta no eclipsa los móviles mucho más cercanos."""
    puntos = [
        _punto("Móvil", "Plaza Mayor", 288,
               "Sábado de 10:30 a 12:00 h.", "SABADO"),
        _punto("Móvil", "Calle Príncipe", 336,
               "Martes de 15:45 a 17:15 h.", "MARTES"),
        _punto("Móvil", "Tirso de Molina", 473,
               "Sábado de 12:15 a 13:45 h.", "SABADO"),
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
        _punto("Proximidad", "Gran Vía", 1120,
               "Todos los días 24 horas."),
        _punto("Proximidad", "Plasencia", 1341,
               "Todos los días 24 horas."),
        _punto("Proximidad", "Santa Bárbara", 1364,
               "Todos los días 24 horas."),
    ]
    respuesta, plan = respuesta_geografica_integrada(_estado(puntos))
    assert "**Mejor opción para acudir ahora" in respuesta
    assert respuesta.index("Alameda") < respuesta.index("Plaza Mayor")
    assert "Calle Príncipe" in respuesta
    assert "Tirso de Molina" in respuesta
    assert "Plasencia" not in respuesta
    assert "Santa Bárbara" not in respuesta
    assert respuesta.count("Todos los días 24 horas") == 2
    assert "Fuera del horario publicado" in respuesta
    assert "Alameda" in plan[0]


def test_con_muchas_opciones_usa_candidatos_extendidos():
    """No deduce que todos estén cerrados de solo los diez primeros."""
    cerrados = [
        _punto("Móvil", f"Móvil {i}", i * 60,
               "Sábado de 10:30 a 12:00 h.", "SABADO")
        for i in range(1, 13)
    ]
    abierto = _punto("Proximidad", "Proximidad abierta", 1100,
                     "Todos los días 24 horas.")
    respuesta, _ = respuesta_geografica_integrada(
        _estado([*cerrados, abierto])
    )
    assert "**Mejor opción para acudir ahora" in respuesta
    assert "Proximidad abierta" in respuesta
    assert "Móvil 1" in respuesta
    assert "Móvil 2" in respuesta


def test_marquesinas_documentadas_sin_fingir_geolocalizacion():
    """Menciona esa modalidad solo cuando aparece en el RAG citado."""
    datos = _estado([
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
    ])
    rag = datos["evidencias"][0]["datos"]
    rag["respuesta"] += " También en marquesinas de autobús [C4]."
    respuesta, _ = respuesta_geografica_integrada(datos)
    assert "Marquesinas de autobús" in respuesta
    assert "marquesinas" in respuesta
    assert "no devuelve sus ubicaciones" in respuesta


def test_sin_marquesinas_en_rag_no_se_introducen_como_aceptadas():
    datos = _estado([
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
    ])
    respuesta, _ = respuesta_geografica_integrada(datos)
    assert "**Marquesinas de autobús:**" not in respuesta


def test_solo_horarios_desconocidos_no_inventa_disponibilidad():
    datos = _estado([
        _punto("Móvil", "Calle A", 100, "No consta horario"),
        _punto("Móvil", "Calle B", 200, "Horario pendiente"),
    ])
    respuesta, plan = respuesta_geografica_integrada(datos)
    assert "No se encontró una instalación" in respuesta
    assert "Mejor opción para acudir ahora" not in respuesta
    assert "Horario no comprobable" in respuesta
    assert len(plan) == 2


def test_mas_cercanos_explicitamente_respeta_distancia_y_explica_estado():
    datos = _estado([
        _punto("Móvil", "Plaza Mayor", 288,
               "Sábado de 10:30 a 12:00 h.", "SABADO"),
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
    ], pide="Dame los más cercanos, aunque estén cerrados")
    respuesta, _ = respuesta_geografica_integrada(datos)
    assert respuesta.index("Plaza Mayor") < respuesta.index("Alameda")
    assert "Fuera del horario publicado" in respuesta
    assert "Dentro del horario publicado" in respuesta


def test_sin_hora_ninguna_opcion_se_declara_disponible():
    datos = _estado([
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
    ], hora=False)
    respuesta, _ = respuesta_geografica_integrada(datos)
    assert "No se pudo obtener la hora actual" in respuesta
    assert "No se encontró una instalación" in respuesta
    assert "Horario no comprobable" in respuesta


def test_pilas_sin_marquesinas_en_rag_declara_limite_sin_inventar():
    datos = _estado([
        _punto("Proximidad", "Alameda", 1011,
               "Todos los días 24 horas."),
    ])
    respuesta, _ = respuesta_geografica_integrada(datos)
    assert "Límite geográfico" in respuesta
    assert "no confirma esa modalidad" in respuesta
