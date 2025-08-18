"""
Carga y preparación de datos para LSTM.
Reemplazá los TODO por tu lógica real (SQL Server, Influx, merges).
"""

from datetime import datetime
from app.utils.timeutils import as_local

def fetch_signal_series(equip_grp: str, equipment: str, signal_id: str,
                        t_start: datetime, t_end: datetime):
    """
    Devuelve una lista de (ts, value, old, bad) entre t_start y t_end en TZ -03.
    TODO: Implementar tu consulta real (SQL/Influx).
    """
    # Placeholder seguro: estructura vacía
    return []

def merge_with_weather(points, weather_rows=None):
    """
    Enlaza tu serie analógica con clima horario.
    TODO: Implementar merge real con tus columnas y llaves temporales.
    """
    return points  # de momento, passthrough

def clean_bad_old(points):
    """Descarta lecturas con OLD o BAD == 1."""
    return [p for p in points if not (int(p[2] or 0) == 1 or int(p[3] or 0) == 1)]

def to_frame(points):
    """Convierte lista [(ts, val, old, bad), ...] en estructura tabular simple."""
    # Evito pandas para no forzar dependencia; adaptá si querés usar DataFrame
    data = [{"ts": as_local(p[0]), "value": float(p[1])} for p in points]
    return data
