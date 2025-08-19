
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Dependencias externas (con fallback limpio si no están)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    tf = None
    Sequential = load_model = LSTM = Dense = Dropout = EarlyStopping = None

# Fallback: scikit-learn MLP
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from .files import MODELS, ensure_dirs
from app.blueprints.analizador.services import (
    fetch_sql_series, get_coords_para_equipo, fetch_weather_history_influx
)

MODEL_DIR = Path(MODELS)  # storage/models/lstm

def _paths(equip_grp, equipment, signal_id):
    safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
    base = f"lstm_{safe(equip_grp)}_{safe(equipment)}_{safe(signal_id)}"
    suffix = ".h5" if tf is not None else ".joblib"
    return {
        "model": MODEL_DIR / f"{base}{suffix}",
        "meta":  MODEL_DIR / f"{base}.meta.json"
    }

def _load_meta(pmeta: Path):
    if pmeta.exists():
        try:
            return json.loads(pmeta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_meta(pmeta: Path, d: dict):
    pmeta.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _prepare_dataset(equip_grp, equipment, signal_id, start_dt, end_dt, lookback=24):
    """
    Arma un dataset horario con features de clima (si disponibilidad) y VALUE como target.
    Devuelve (X, y, feature_cols, last_ts, df_hourly)
    """
    # 1) SQL (descartar OLD/BAD)
    df_sql = fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt).copy()
    if df_sql.empty:
        return None, None, None, None, pd.DataFrame()
    df_sql = df_sql[(df_sql["OLD"] == 0) & (df_sql["BAD"] == 0)]
    if df_sql.empty:
        return None, None, None, None, pd.DataFrame()

    # Resample a 1H (si viene con mayor frecuencia)
    df = df_sql.set_index("TIME_FULL").sort_index()
    # Dato puntual: promedio por hora (o último)
    hourly = df["VALUE"].resample("1H").mean()

    # 2) Clima desde Influx si hay mapeo de coords
    lat, lon = get_coords_para_equipo(equip_grp, equipment)
    clima = None
    if lat is not None and lon is not None:
        try:
            clima = fetch_weather_history_influx(equip_grp, start_dt, end_dt)
        except Exception:
            clima = None

    if isinstance(clima, pd.DataFrame) and not clima.empty:
        clima = clima.set_index("time").sort_index()
        # Normalizamos nombres esperados
        for c in ("temperature","relative_humidity","windspeed","winddirection"):
            if c not in clima.columns:
                clima[c] = np.nan
        clima_hour = clima[["temperature","relative_humidity","windspeed","winddirection"]].resample("1H").mean()
        # Merge outer para no perder VALUE; usar solo VALUE si clima faltante
        df_hourly = pd.DataFrame({"VALUE": hourly}).join(clima_hour, how="left")
    else:
        df_hourly = pd.DataFrame({"VALUE": hourly})
        for c in ("temperature","relative_humidity","windspeed","winddirection"):
            df_hourly[c] = np.nan

    # Completar valores faltantes de clima con fflll / interpolación suave
    for c in ("temperature","relative_humidity","windspeed","winddirection"):
        if c in df_hourly.columns:
            df_hourly[c] = df_hourly[c].interpolate(limit_direction="both")
            df_hourly[c] = df_hourly[c].fillna(0.0)

    # No forzar dropna de VALUE: quitar solo ventanas incompletas
    df_hourly = df_hourly.dropna(subset=["VALUE"])

    # Construir ventanas lookback -> target siguiente hora
    feats = ["VALUE","temperature","relative_humidity","windspeed"]
    feats = [c for c in feats if c in df_hourly.columns]
    data = df_hourly[feats].values.astype("float32")

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # predecir VALUE de la hora siguiente
    if not X:
        return None, None, feats, df_hourly.index.max(), df_hourly
    X = np.array(X)  # (N, lookback, F)
    y = np.array(y).reshape(-1, 1)

    last_ts = df_hourly.index.max()
    return X, y, feats, last_ts, df_hourly

def _build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_or_update(equip_grp, equipment, signal_id, start_dt, end_dt, lookback=24, epochs=10):
    ensure_dirs()

    paths = _paths(equip_grp, equipment, signal_id)
    pmodel, pmeta = paths["model"], paths["meta"]
    meta = _load_meta(pmeta)

    # Elegir rango: si ya tenemos último entrenamiento, usar desde ahí
    if "last_train_end" in meta:
        try:
            last_trained = pd.to_datetime(meta["last_train_end"])
            if last_trained.tzinfo is not None:
                last_trained = last_trained.tz_convert(None).to_pydatetime()
            # entrenar desde la hora siguiente
            start_dt = max(start_dt, last_trained + timedelta(hours=1))
        except Exception:
            pass

    X, y, feats, last_ts, df_hourly = _prepare_dataset(
        equip_grp, equipment, signal_id, start_dt, end_dt, lookback=lookback
    )
    if X is None or len(X) == 0:
        return {"trained": False, "reason": "No hay datos suficientes para entrenar."}

    backend = "lstm" if tf is not None else "mlp"

    if backend == "lstm":
        if pmodel.exists():
            model = tf.keras.models.load_model(str(pmodel))
        else:
            model = _build_model((X.shape[1], X.shape[2]))
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
        model.save(str(pmodel))
    else:
        # Flatten windows to (N, lookback*F)
        N, L, F = X.shape
        X2 = X.reshape(N, L*F)
        # Pipeline: Standardize -> MLP
        if pmodel.exists():
            model = joblib.load(pmodel)
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
                                       solver="adam", max_iter=400, early_stopping=True,
                                       n_iter_no_change=10, random_state=42))
            ])
        model.fit(X2, y.ravel())
        joblib.dump(model, pmodel)

    meta.update({
        "feats": feats,
        "lookback": lookback,
        "last_train_end": str(last_ts),
        "backend": backend
    })
    _save_meta(pmeta, meta)

    return {"trained": True, "samples": int(len(X)), "last_ts": str(last_ts), "model_path": str(pmodel), "backend": backend}

def forecast_next_hours(equip_grp, equipment, signal_id, steps=24):
    """
    Proyección autoregresiva hora a hora (usa el último 'lookback' del df_hourly).
    """
    paths = _paths(equip_grp, equipment, signal_id)
    pmodel, pmeta = paths["model"], paths["meta"]
    if not pmodel.exists():
        raise RuntimeError("No hay modelo entrenado para esta señal. Entrena primero.")

    meta = _load_meta(pmeta)
    lookback = int(meta.get("lookback", 24))
    feats = meta.get("feats", ["VALUE"])
    backend = meta.get("backend", "lstm" if tf is not None else "mlp")

    # Reconstruir df_hourly hasta ahora para obtener la ventana reciente
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    Xh, yh, feats2, last_ts, df_hourly = _prepare_dataset(
        equip_grp, equipment, signal_id,
        start_dt=now - timedelta(days=30),
        end_dt=now,
        lookback=lookback
    )
    if df_hourly is None or df_hourly.empty:
        raise RuntimeError("No hay datos recientes para proyectar.")

    # Tomar ultima ventana
    last_window = df_hourly[feats].values.astype("float32")[-lookback:]  # (lookback, F)
    # Cargar modelo según backend
    if backend == "lstm":
        model = tf.keras.models.load_model(str(pmodel))
    else:
        model = joblib.load(pmodel)

    preds = []
    curr_ts = df_hourly.index.max()
    window = last_window.copy()
    for _ in range(steps):
        if backend == "lstm":
            x = window.reshape(1, lookback, len(feats))
            yhat = float(model.predict(x, verbose=0)[0,0])
        else:
            x = window.reshape(1, lookback * len(feats))
            yhat = float(model.predict(x)[0])
        preds.append((curr_ts + timedelta(hours=1), yhat))
        next_row = window[-1].copy()
        next_row[0] = yhat
        window = np.vstack([window[1:], next_row])

        curr_ts = curr_ts + timedelta(hours=1)

    out_df = pd.DataFrame(preds, columns=["TIME_FULL","FORECAST_VALUE"])
    return out_df
