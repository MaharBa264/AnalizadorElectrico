from datetime import datetime, timedelta
from app.utils.timeutils import as_local, LOCAL_TZ
import os

def _sql_query_template():
    tpl = os.getenv("SQL_ANALOG_QUERY", "").strip()
    if not tpl:
        raise RuntimeError(
            "Falta SQL_ANALOG_QUERY en .env. Ejemplo: "
            "SELECT Time, VALUE, OLD, BAD FROM ANALOG_NUEVA "
            "WHERE EQUIP_GRP=? AND EQUIPMENT=? AND ID=? AND Time BETWEEN ? AND ? ORDER BY Time"
        )
    return tpl

def fetch_signal_series(equip_grp: str, equipment: str, signal_id: str,
                        t_start: datetime, t_end: datetime):
    """
    Devuelve lista de tuplas (ts, value, old, bad) entre t_start y t_end.
    t_start/t_end pueden venir naive o con tz; las convertimos a -03.
    """
    from app.services import sql
    q = _sql_query_template()

    # Normalizamos a -03 y mandamos naive (str) al SQL
    ts = as_local(t_start)
    te = as_local(t_end)
    params = [equip_grp, equipment, signal_id, ts.replace(tzinfo=None), te.replace(tzinfo=None)]

    rows = sql.query_list(q, params=params)
    out = []
    for r in rows:
        # Campos esperados: Time, VALUE, OLD, BAD (nombres no case sensitive en pyodbc)
        ts = r.get("Time") or r.get("time") or r.get("TIME")
        val = r.get("VALUE") or r.get("value") or r.get("Value")
        old = r.get("OLD") or r.get("old") or r.get("Old") or 0
        bad = r.get("BAD") or r.get("bad") or r.get("Bad") or 0
        if ts is None or val is None:
            continue
        # Consideramos ts como naive (SQL), y lo interpretamos en -03
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=LOCAL_TZ)
        else:
            ts = ts.astimezone(LOCAL_TZ)
        out.append((ts, float(val), int(old or 0), int(bad or 0)))
    return out

def clean_bad_old(points):
    """Descarta lecturas con OLD o BAD == 1."""
    return [p for p in points if not (int(p[2] or 0) == 1 or int(p[3] or 0) == 1)]

def resample_hourly(points, agg="mean"):
    """
    Re-muestrea a 1h para alinear con clima. Devuelve lista de (ts_hour, value).
    agg: 'mean'|'last'
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for ts, val, _, _ in points:
        # Piso a hora en -03
        th = ts.replace(minute=0, second=0, microsecond=0)
        buckets[th].append(float(val))
    out = []
    for th in sorted(buckets.keys()):
        vals = buckets[th]
        if not vals:
            continue
        out.append((th, (sum(vals)/len(vals) if agg=="mean" else vals[-1])))
    return out

def merge_with_weather(hourly_points, weather_rows=None):
    """
    Une por timestamp hora exacta (th). Si weather_rows es None o vacío, devuelve solo analog.
    hourly_points: [(th, value)]
    weather_rows:  [{'ts': datetime, 'temperature': float, 'windspeed': float, 'relative_humidity': float}, ...]
    Retorna lista de dicts con llaves: ts, value, temperature?, windspeed?, relative_humidity?
    """
    if not weather_rows:
        return [{"ts": th, "value": v} for th, v in hourly_points]

    wmap = { r["ts"].replace(minute=0, second=0, microsecond=0): r for r in weather_rows if r.get("ts") }
    out = []
    for th, v in hourly_points:
        row = {"ts": th, "value": v}
        w = wmap.get(th)
        if w:
            for k in ("temperature","windspeed","relative_humidity"):
                if k in w:
                    row[k] = w[k]
        out.append(row)
    return out

def to_frame(points):
    """Convierte lista [(ts, val, old, bad), ...] o [(ts, val)] en lista de dict para render."""
    # tolera ambas formas
    out = []
    for p in points:
        if len(p) >= 4:
            out.append({"ts": as_local(p[0]), "value": float(p[1])})
        else:
            out.append({"ts": as_local(p[0]), "value": float(p[1])})
    return out
