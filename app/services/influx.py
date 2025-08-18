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
