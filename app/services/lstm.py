import os, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def ensure_dirs():
    base = Path(os.getenv("MODELS_DIR", "instance/models"))
    base.mkdir(parents=True, exist_ok=True)
    return str(base)

MODELS = ensure_dirs()

def _paths(equip_grp, equipment, signal_id):
    def safe(s): return "".join(c if str(c).isalnum() or c in "-_." else "_" for c in str(s))
    base = f"naive_{safe(equip_grp)}_{safe(equipment)}_{safe(signal_id)}"
    model = Path(MODELS) / f"{base}.joblib"
    meta  = Path(MODELS) / f"{base}.meta.json"
    return {"model": model, "meta": meta}

def _hourly_series(equip_grp, equipment, signal_id, start_dt, end_dt):
    from app.blueprints.analizador.services import fetch_sql_series
    df = fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt)
    if df is None or df.empty:
        return pd.DataFrame(columns=["TIME_FULL","VALUE"]).set_index("TIME_FULL")
    df = df[df["BAD"]==0].copy()
    df["TIME_FULL"] = pd.to_datetime(df["TIME_FULL"], errors="coerce")
    df = df.dropna(subset=["TIME_FULL"]).sort_values("TIME_FULL")
    df = df.set_index("TIME_FULL")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    y = pd.to_numeric(df["VALUE"], errors="coerce").resample("1H").mean().interpolate(limit=3, limit_direction="both")
    return y.to_frame(name="VALUE")

def _windowize(df_hourly, lookback=24):
    vals = df_hourly["VALUE"].values.astype(float)
    X, y = [], []
    for i in range(lookback, len(vals)):
        X.append(vals[i-lookback:i])
        y.append(vals[i])
    if not X:
        return np.empty((0,lookback)), np.array([])
    return np.array(X), np.array(y)

def train_or_update(equip_grp, equipment, signal_id, start_dt, end_dt, lookback=24, epochs=0):
    paths = _paths(equip_grp, equipment, signal_id)
    model_path, meta_path = paths["model"], paths["meta"]

    df_hourly = _hourly_series(equip_grp, equipment, signal_id, start_dt, end_dt)
    X, y = _windowize(df_hourly, lookback=lookback)
    if len(y) < 8:
        raise RuntimeError("No hay datos suficientes para entrenar (mín. 8 pts)." )

    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))])
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)

    meta = {
        "backend":"mlp",
        "lookback": int(lookback),
        "last_train_end": str(df_hourly.index.max() if len(df_hourly) else ""),
        "y_min": float(np.nanmin(df_hourly["VALUE"])) if len(df_hourly) else None,
        "y_max": float(np.nanmax(df_hourly["VALUE"])) if len(df_hourly) else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"trained": True, "samples": int(len(y)), "last_ts": meta["last_train_end"], "model_path": str(model_path), "backend":"mlp"}

def _load_model_meta(equip_grp, equipment, signal_id, lookback_default=24):
    paths = _paths(equip_grp, equipment, signal_id)
    model_path, meta_path = paths["model"], paths["meta"]
    if not model_path.exists() or not meta_path.exists():
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        train_or_update(equip_grp, equipment, signal_id, now - timedelta(days=14), now, lookback=lookback_default)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(model_path)
    return model, meta

def predict_last_hours(equip_grp, equipment, signal_id, hours=48):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=14)
    full = _hourly_series(equip_grp, equipment, signal_id, start, now)
    if len(full) < hours + 30:
        series_bt = full.iloc[-hours:].copy()
        preds = series_bt["VALUE"].rolling(12, min_periods=1).mean()
        return pd.DataFrame({"TIME_FULL": series_bt.index, "PRED_VALUE": preds.values}), series_bt["VALUE"]
    train_end = full.index.max() - timedelta(hours=hours)
    train_or_update(equip_grp, equipment, signal_id, full.index.min(), train_end, lookback=24)
    model, meta = _load_model_meta(equip_grp, equipment, signal_id)
    lookback = int(meta.get("lookback", 24))
    hist = full[full.index <= train_end].copy()
    test = full[full.index > train_end].iloc[:hours].copy()
    context = hist["VALUE"].values[-lookback:]
    if len(context) < lookback:
        pad = np.repeat(context[:1] if len(context) else np.array([0.0]), lookback - len(context))
        context = np.concatenate([pad, context])
    preds = []
    for ts, true_v in test["VALUE"].items():
        x = context.reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append((ts, yhat))
        context = np.roll(context, -1); context[-1] = true_v if not np.isnan(true_v) else yhat
    df_pred = pd.DataFrame(preds, columns=["TIME_FULL","PRED_VALUE"])
    return df_pred, test["VALUE"]

def forecast_next_hours(equip_grp, equipment, signal_id, steps=48):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    hist = _hourly_series(equip_grp, equipment, signal_id, now - timedelta(days=14), now)
    model, meta = _load_model_meta(equip_grp, equipment, signal_id)
    lookback = int(meta.get("lookback", 24))
    context = hist["VALUE"].values[-lookback:]
    if len(context) < lookback:
        pad = np.repeat(context[:1] if len(context) else np.array([0.0]), lookback - len(context))
        context = np.concatenate([pad, context])
    preds = []
    t = now + timedelta(hours=1)
    for i in range(steps):
        x = context.reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append((t, yhat))
        context = np.roll(context, -1); context[-1] = yhat
        t = t + timedelta(hours=1)
    return pd.DataFrame({"TIME_FULL": [ts for ts,_ in preds], "FORECAST_VALUE": [v for _,v in preds]})