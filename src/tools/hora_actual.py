"""Hora civil española, sin red, LLM ni dependencia del reloj local del SO."""
from datetime import datetime
from zoneinfo import ZoneInfo

ZONA = "Europe/Madrid"


def _ahora() -> datetime:
    return datetime.now(ZoneInfo(ZONA))


def hora_actual() -> dict[str, str]:
    """Devuelve fecha/hora actual en horario oficial peninsular español."""
    return {"fecha_hora": _ahora().isoformat(timespec="seconds"), "zona_horaria": ZONA}
