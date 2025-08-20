import os
import json as _json
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

def make_client():
    """Retorna InfluxDBClient v2 o levanta excepción amigable si no está instalada la lib."""
    try:
        from influxdb_client import InfluxDBClient
    except Exception as e:
        raise RuntimeError("Falta instalar influxdb-client para Python 3.6 (v2).") from e

    url = os.getenv("INFLUX_URL", "http://localhost:8086")
    token = os.getenv("INFLUX_TOKEN", "")
    org = os.getenv("INFLUX_ORG", "")
    return InfluxDBClient(url=url, token=token, org=org)

def test_ping():
    try:
        c = make_client()
        health = c.health()
        ok = getattr(health, "status", "fail") == "pass"
        return ok, f"Influx health: {getattr(health, 'status', '?')}"
    except Exception as e:
        return False, str(e)

def _iso(dt):
    # to ISO8601 con 'Z' (UTC). Si dt no tiene tz, asumimos UTC.
    ts = pd.to_datetime(dt)
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def query_weather_hourly(equip_grp, start_dt, end_dt):
    """
    Lee clima horario desde InfluxDB v2 (API HTTP) y devuelve lista de dicts:
    [{time, temperature, relative_humidity, windspeed, winddirection}, ...]
    Filtra por TAG del grupo (configurable).
    Requiere en .env:
      INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_WEATHER_BUCKET
    Opcionales:
      INFLUX_WEATHER_MEAS (default: 'weather')
      INFLUX_WEATHER_TAG  (default: 'equip_grp')
      INFLUX_WEATHER_TEMP_FIELD (default: 'temperature')
      INFLUX_WEATHER_HUM_FIELD  (default: 'relative_humidity')
      INFLUX_WEATHER_WSPD_FIELD (default: 'windspeed')
      INFLUX_WEATHER_WDIR_FIELD (default: 'winddirection')
    """
    url = os.getenv("INFLUX_URL", "http://localhost:8086").rstrip("/")
    token = os.getenv("INFLUX_TOKEN", "")
    org = os.getenv("INFLUX_ORG", "")
    bucket = os.getenv("INFLUX_WEATHER_BUCKET", "").strip()
    meas = os.getenv("INFLUX_WEATHER_MEAS", "weather")
    tag_key = os.getenv("INFLUX_WEATHER_TAG", "equip_grp")

    f_temp = os.getenv("INFLUX_WEATHER_TEMP_FIELD", "temperature")
    f_hum  = os.getenv("INFLUX_WEATHER_HUM_FIELD", "relative_humidity")
    f_wspd = os.getenv("INFLUX_WEATHER_WSPD_FIELD", "windspeed")
    f_wdir = os.getenv("INFLUX_WEATHER_WDIR_FIELD", "winddirection")

    if not (url and token and org and bucket):
        # Faltan credenciales/params
        return []

    # Escapar comillas por seguridad
    eg = str(equip_grp).replace('"', r'\"')
    start_iso = _iso(start_dt)
    stop_iso = _iso(end_dt)

    flux = f'''
from(bucket: "{bucket}")
  |> range(start: time(v: "{start_iso}"), stop: time(v: "{stop_iso}"))
  |> filter(fn: (r) => r._measurement == "{meas}")
  |> filter(fn: (r) => r["{tag_key}"] == "{eg}")
  |> filter(fn: (r) => r._field == "{f_temp}" or r._field == "{f_hum}" or r._field == "{f_wspd}" or r._field == "{f_wdir}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","{f_temp}","{f_hum}","{f_wspd}","{f_wdir}"])
  |> sort(columns: ["_time"])
'''

    endpoint = f"{url}/api/v2/query?org={org}"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/csv"
    }
    payload = {
        "query": flux,
        "type": "flux",
        "dialect": {
            "annotations": ["group", "datatype", "default"],
            "delimiter": ",",
            "header": True
        }
    }

    try:
        resp = requests.post(endpoint, headers=headers, data=_json.dumps(payload), timeout=(5, 30))
        resp.raise_for_status()
        # CSV con líneas de comentario '#'; pandas las ignora con comment='#'
        csv_txt = resp.text
        df = pd.read_csv(StringIO(csv_txt), comment="#")
        if df.empty:
            return []
        # columnas esperadas: _time, opcionalmente result/table; más nuestras fields
        cols = df.columns.tolist()
        if "_time" not in cols:
            return []

        # Renombrar a nuestro esquema
        out = pd.DataFrame({
            "time": pd.to_datetime(df["_time"]).dt.floor("H"),
            "temperature": df.get(f_temp),
            "relative_humidity": df.get(f_hum),
            "windspeed": df.get(f_wspd),
            "winddirection": df.get(f_wdir)
        })

        out = out.sort_values("time").dropna(subset=["time"])
        # dicts
        out_rows = []
        for _, r in out.iterrows():
            out_rows.append({
                "time": r["time"].to_pydatetime(),
                "temperature": None if pd.isna(r["temperature"]) else float(r["temperature"]),
                "relative_humidity": None if pd.isna(r["relative_humidity"]) else float(r["relative_humidity"]),
                "windspeed": None if pd.isna(r["windspeed"]) else float(r["windspeed"]),
                "winddirection": None if pd.isna(r["winddirection"]) else float(r["winddirection"]),
            })
        return out_rows
    except Exception:
        return []