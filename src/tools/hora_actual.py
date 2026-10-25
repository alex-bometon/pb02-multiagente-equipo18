"""Herramienta para consultar la fecha y hora actuales."""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from config import TIMEZONE


@tool
def hora_actual() -> str:
    """Devuelve la fecha y hora actuales de Madrid en formato ISO 8601."""
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
