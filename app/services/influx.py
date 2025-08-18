import os

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

def query_weather_hourly(tag_value: str, start, stop):
    """
    Devuelve lista de dicts: {'ts', 'temperature'?, 'windspeed'?, 'relative_humidity'?}
    Requiere influxdb-client v2 disponible.
    """
    meas = os.getenv("INFLUX_WEATHER_MEAS", "weather_hourly")
    tagk = os.getenv("INFLUX_WEATHER_TAG_KEY", "equip_grp")
    bucket = os.getenv("INFLUX_BUCKET", "")

    c = make_client()
    q = c.query_api()

    # Formateo tiempo ISO
    s = start.astimezone().isoformat()
    e = stop.astimezone().isoformat()

    flux = f'''
from(bucket: "{bucket}")
  |> range(start: time(v: "{s}"), stop: time(v: "{e}"))
  |> filter(fn: (r) => r._measurement == "{meas}")
  |> filter(fn: (r) => r["{tagk}"] == "{tag_value}")
  |> filter(fn: (r) => r._field == "temperature" or r._field == "windspeed" or r._field == "relative_humidity")
  |> aggregateWindow(every: 1h, fn: last, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","temperature","windspeed","relative_humidity"])
'''
    tables = q.query(flux)
    out = []
    for tbl in tables:
        for rec in tbl.records:
            row = {"ts": rec.get_time()}
            # los campos pueden o no estar en el registro según pivot
            for k in ("temperature","windspeed","relative_humidity"):
                if k in rec.values:
                    row[k] = rec.values[k]
            out.append(row)
    # Orden por ts
    out.sort(key=lambda r: r["ts"])
    return out
