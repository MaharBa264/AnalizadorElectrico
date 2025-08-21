# app/services/signal_adapter.py
import os
from datetime import timedelta
try:
    from app.blueprints.analizador.services import get_signal_hierarchy, fetch_sql_series
    _USE_ANALIZADOR = True
except Exception:
    _USE_ANALIZADOR = False

from app.services.sql import query_list as sql_query
import pandas as pd

def list_hierarchy():
    if _USE_ANALIZADOR:
        return get_signal_hierarchy()
    # Fallback simple (si fallara el import): leer de *_KEY
    for key_tbl in [os.getenv("ANALOG_KEY_TABLE","").strip() or None, "ANALOG_NUEVA_KEY", "ANALOG_KEY", "dbo.ANALOG_KEY"]:
        if not key_tbl: continue
        try:
            sql_query(f"SELECT TOP 1 1 FROM {key_tbl}", [])
            rows = sql_query(f"SELECT EQUIP_GRP, EQUIPMENT, ID FROM {key_tbl}", []) or []
            est = {}
            for r in rows:
                g = str(r.get("EQUIP_GRP","")).strip()
                e = str(r.get("EQUIPMENT","")).strip()
                i = r.get("ID")
                if not g or not e or i is None: continue
                est.setdefault(g, {}).setdefault(e, []).append(i)
            return est
        except Exception:
            continue
    return {}

def series(equip_grp, equipment, signal_id, start_dt, end_dt):
    if _USE_ANALIZADOR:
        return fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt)
    return pd.DataFrame(columns=["TIME_FULL","VALUE","OLD","BAD"])

def snapshot(equip_grp, equipment, signal_id, when_dt, tolerance_minutes=10):
    start = when_dt - timedelta(minutes=tolerance_minutes)
    end   = when_dt + timedelta(minutes=tolerance_minutes)
    df = series(equip_grp, equipment, signal_id, start, end)
    if df is None or df.empty: return None
    ok = pd.Series([True]*len(df))
    if "OLD" in df: ok &= (df["OLD"].fillna(0)==0)
    if "BAD" in df: ok &= (df["BAD"].fillna(0)==0)
    df = df.loc[ok]
    if df.empty: return None
    if "TIME_FULL" in df.columns:
        df["_d"] = (pd.to_datetime(df["TIME_FULL"]) - when_dt).abs()
        row = df.sort_values("_d").iloc[0]
    else:
        row = df.iloc[-1]
    try:
        return float(row["VALUE"])
    except Exception:
        return None
