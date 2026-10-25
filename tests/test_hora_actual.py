"""Prueba unitaria de la herramienta de fecha y hora."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.tools.hora_actual import hora_actual
from config import TIMEZONE


def test_hora_actual():
    """Comprueba el registro, formato y zona horaria de la tool."""

    zona_madrid = ZoneInfo(TIMEZONE)

    # Comprobar el registro en LangChain.
    assert hora_actual.name == "hora_actual"
    assert hora_actual.args == {}

    # Obtener la hora antes y después de invocar la tool.
    inicio = datetime.now(zona_madrid)
    resultado = hora_actual.invoke({})
    fin = datetime.now(zona_madrid)

    # Comprobar que devuelve una cadena ISO 8601 válida.
    assert isinstance(resultado, str)

    fecha = datetime.fromisoformat(resultado)

    # Comprobar que incluye información de zona horaria.
    assert fecha.tzinfo is not None

    # Comprobar que corresponde al horario de Madrid.
    fecha_madrid = fecha.astimezone(zona_madrid)

    assert fecha.utcoffset() == fecha_madrid.utcoffset()

    # Comprobar que la hora devuelta es la actual.
    assert inicio.replace(microsecond=0) <= fecha_madrid <= fin
