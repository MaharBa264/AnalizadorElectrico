import os, csv
import pandas as pd
from app.services import sql
from app.services.files import EXPORTS
from datetime import datetime, timedelta
from app.services.weather import history_hourly, forecast_hourly


def sql_demo_rows():
    """
    Demo segura que no depende de tu schema:
    lista las primeras tablas de la DB (sys.tables).
    """
    try:
        rows = sql.query_list("SELECT TOP 20 name AS table_name FROM sys.tables ORDER BY name")
        return True, rows
    except Exception as e:
        return False, [{"error": str(e)}]



def list_tables():
    q = "SELECT name AS table_name FROM sys.tables ORDER BY name"
    return [r["table_name"] for r in sql.query_list(q)]

def table_preview(table: str, top: int = 50):
    # Whitelist: el nombre debe existir en sys.tables
    tables = set(list_tables())
    if table not in tables:
        raise ValueError("Tabla no permitida.")
    # Consulta segura (escapando con [])
    q = f"SELECT TOP {int(top)} * FROM [{table}]"
    return sql.query_list(q)

def export_preview_csv(table: str, rows: list):
    os.makedirs(EXPORTS, exist_ok=True)
    path = os.path.join(EXPORTS, f"preview_{table}.csv")
    if not rows:
        # crear CSV vacío con solo BOM para evitar problemas de Excel
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("") 
        return path
    hdr = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path




def get_signal_hierarchy():
    """
    Estructura: {EQUIP_GRP: {EQUIPMENT: [ID, ...], ...}, ...}
    Usa ANALOG_NUEVA_KEY; si no existe, intenta ANALOG_KEY.
    """
    from collections import defaultdict
    estructura = defaultdict(lambda: defaultdict(list))
    for table in ("ANALOG_NUEVA_KEY", "ANALOG_KEY"):
        try:
            rows = sql.query_list(f"SELECT EQUIP_GRP, EQUIPMENT, ID FROM {table} WHERE ID IS NOT NULL ORDER BY EQUIP_GRP, EQUIPMENT, ID")
            if rows:
                for r in rows:
                    estructura[r["EQUIP_GRP"]][r["EQUIPMENT"]].append(r["ID"])
                return estructura
        except Exception:
            continue
    return estructura  # vacío si falló

def get_coords_para_equipo(grupo, equipo):
    """
    Lee coords desde CSV (COORDS_CSV) con columnas: EQUIP_GRP, EQUIPMENT, LAT, LON
    """
    path = os.getenv("COORDS_CSV", "instance/equipos_coords.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, dtype=str).fillna("")
    m = df[(df["EQUIP_GRP"] == grupo) & ((df["EQUIPMENT"] == equipo) | (df["EQUIPMENT"] == ""))]
    if m.empty:
        return None, None
    try:
        return float(m.iloc[0]["LAT"]), float(m.iloc[0]["LON"])
    except Exception:
        return None, None

def fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt):
    """
    Usa la consulta de .env (SQL_ANALOG_QUERY). Retorna DataFrame con
    columnas: TIME_FULL (datetime), VALUE (float), OLD (int), BAD (int)
    """
    q = os.getenv("SQL_ANALOG_QUERY", "").strip()
    if not q:
        raise RuntimeError("Falta SQL_ANALOG_QUERY en .env")

    ts = start_dt
    te = end_dt
    params = [equip_grp, equipment, signal_id, ts, te]
    rows = sql.query_list(q, params=params)
    if not rows:
        return pd.DataFrame(columns=["TIME_FULL", "VALUE", "OLD", "BAD"])

    df = pd.DataFrame(rows)
    # Normalización de nombres
    for c in list(df.columns):
        if c.lower() == "time":
            df.rename(columns={c: "TIME_FULL"}, inplace=True)
    df["TIME_FULL"] = pd.to_datetime(df["TIME_FULL"], errors="coerce", infer_datetime_format=True)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    for c in ("OLD","BAD"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0
    return df.sort_values("TIME_FULL")

def fetch_weather_history(lat, lon, start_dt, end_dt):
    """
    Wrapper a WeatherAPI histórico (por día) -> DataFrame horario con columna 'temperature'.
    """
    # history_hourly pide lat/lon y fecha (YYYY-MM-DD) por día
    curr = start_dt.date()
    endd = end_dt.date()
    rows = []
    while curr <= endd:
        data = history_hourly(lat, lon, curr.isoformat())
        # WeatherAPI history.json entrega 'forecast'->'forecastday'[0]->'hour'
        try:
            hours = (data.get("forecast", {}).get("forecastday", [])[0]).get("hour", [])
        except Exception:
            hours = []
        for h in hours:
            # h['time'] es en local time de la ubicación; lo tomamos como naive y lo serializamos ISO
            t = pd.to_datetime(h.get("time"))
            rows.append({
                "time": t,
                "temperature": h.get("temp_c"),
                "relative_humidity": h.get("humidity"),
                "uv": h.get("uv"),
                "windspeed": h.get("wind_kph"),
                "winddirection": h.get("wind_degree"),
            })
        curr += timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["time","temperature","relative_humidity","uv","windspeed","winddirection"])
    dfw = pd.DataFrame(rows)
    # Redondeamos a hora exacta para merge
    dfw["time"] = pd.to_datetime(dfw["time"]).dt.floor("H")
    # Si hay duplicados por hora, nos quedamos con el último
    dfw = dfw.drop_duplicates(subset=["time"], keep="last")
    return dfw.sort_values("time")

def make_trend_payload(df_sql):
    # D3: [{"Time": ISO, "VALUE": v}, ...]
    tmp = df_sql.dropna(subset=["TIME_FULL","VALUE"])[["TIME_FULL","VALUE"]].copy()
    tmp["Time"] = pd.to_datetime(tmp["TIME_FULL"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return tmp[["Time","VALUE"]].to_dict(orient="records")


def fetch_weather_history_influx(equip_grp, start_dt, end_dt):
    """
    Lee clima horario desde Influx (bucket/measurement/tag configurados en .env vía services.influx).
    Retorna DataFrame con columnas: time, temperature, relative_humidity, windspeed, winddirection
    """
    try:
        from app.services.influx import query_weather_hourly
    except Exception as e:
        # Si la lib no está instalada o hay error, no rompemos
        return pd.DataFrame(columns=["time","temperature","relative_humidity","windspeed","winddirection"])

    try:
        rows = query_weather_hourly(equip_grp, start_dt, end_dt)  # lista de dicts
        if not rows:
            return pd.DataFrame(columns=["time","temperature","relative_humidity","windspeed","winddirection"])
        df = pd.DataFrame(rows)
        if "time" not in df.columns:
            # algunos drivers devuelven "_time"
            if "_time" in df.columns:
                df.rename(columns={"_time":"time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"]).dt.floor("H")
        # columnas ausentes → NaN
        for c in ("temperature","relative_humidity","windspeed","winddirection"):
            if c not in df.columns:
                df[c] = pd.NA
        # si viene en m/s o mph podrías normalizar acá (dejamos tal cual)
        return df.sort_values("time")
    except Exception:
        return pd.DataFrame(columns=["time","temperature","relative_humidity","windspeed","winddirection"])

def _load_sql_query():
    """
    Prioriza SQL_ANALOG_QUERY_FILE si existe; si no, usa SQL_ANALOG_QUERY.
    """
    path = os.getenv("SQL_ANALOG_QUERY_FILE", "").strip()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            q = f.read().strip()
            if q:
                return q
    q = os.getenv("SQL_ANALOG_QUERY", "").strip()
    if not q:
        raise RuntimeError("Falta SQL_ANALOG_QUERY o SQL_ANALOG_QUERY_FILE en .env")
    return q

def fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt):
    q = _load_sql_query()  # <-- usa archivo o env
    ts = start_dt.replace(minute=0, second=0, microsecond=0)
    te = end_dt.replace(minute=0, second=0, microsecond=0)
    params = [equip_grp, equipment, signal_id, ts, te]
    rows = sql.query_list(q, params=params)
    if not rows:
        return pd.DataFrame(columns=["TIME_FULL", "VALUE", "OLD", "BAD"])

    df = pd.DataFrame(rows)
    # Normalización robusta
    for c in list(df.columns):
        if c.lower() == "time":
            df.rename(columns={c: "TIME_FULL"}, inplace=True)
    df["TIME_FULL"] = pd.to_datetime(df["TIME_FULL"], errors="coerce", infer_datetime_format=True)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    for c in ("OLD","BAD"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0
    return df.sort_values("TIME_FULL")