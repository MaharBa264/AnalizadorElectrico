
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor

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
from app.services.weather import forecast_hourly

from .files import MODELS, ensure_dirs
from app.blueprints.analizador.services import (
    fetch_sql_series, get_coords_para_equipo, fetch_weather_history_influx
)

MODEL_DIR = Path(MODELS)  # storage/models/lstm

def _paths(equip_grp, equipment, signal_id, backend_hint=None):
    """
    Devuelve rutas de modelo y meta. Soporta backend_hint opcional.
    Si ya existe un modelo (.h5 o .joblib), usa el más reciente.
    Si no existe, decide por backend_hint o por disponibilidad de TF (LSTM) -> .h5, sino .joblib.
    """
    from pathlib import Path
    try:
        from .files import MODELS, ensure_dirs
    except Exception:
        # fallback si cambia la importación
        from app.services.files import MODELS, ensure_dirs

    ensure_dirs()

    def safe(s):
        return "".join(c if str(c).isalnum() or c in "-_." else "_" for c in str(s))

    base = f"lstm_{safe(equip_grp)}_{safe(equipment)}_{safe(signal_id)}"
    model_dir = Path(MODELS)
    meta_path = model_dir / f"{base}.meta.json"

    # Si ya hay modelos guardados, elegimos el más nuevo
    candidates = [model_dir / f"{base}.h5", model_dir / f"{base}.joblib"]
    existing = [p for p in candidates if p.exists()]
    if existing:
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        return {"model": latest, "meta": meta_path}

    # No existe ninguno: decidir por backend_hint o por TF disponible
    use_lstm = False
    try:
        # tf puede ser None si no está instalado
        use_lstm = (backend_hint == "lstm") or (backend_hint is None and (tf is not None))
    except NameError:
        use_lstm = (backend_hint == "lstm")

    suffix = ".h5" if use_lstm else ".joblib"
    return {"model": model_dir / f"{base}{suffix}", "meta": meta_path}


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
    Arma dataset horario con VALUE como target y clima como exógenos si hay.
    Devuelve: (X, y, feats, last_ts, df_hourly, (y_min, y_max))
    """
    # 1) Traer serie desde SQL
    df_sql = fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt).copy()
    if df_sql.empty:
        return None, None, None, None, pd.DataFrame(), (None, None)

    # Flags robustos
    if "OLD" not in df_sql.columns: df_sql["OLD"] = 0
    if "BAD" not in df_sql.columns: df_sql["BAD"] = 0
    for c in ("OLD","BAD"):
        df_sql[c] = pd.to_numeric(df_sql[c], errors="coerce").fillna(0).astype(int)

    # Tiempo
    if "TIME_FULL" not in df_sql.columns:
        return None, None, None, None, pd.DataFrame(), (None, None)
    df_sql["TIME_FULL"] = pd.to_datetime(df_sql["TIME_FULL"], errors="coerce")
    df_sql = df_sql.dropna(subset=["TIME_FULL"]).sort_values("TIME_FULL")

    # Filtro (nos quedamos con BAD==0; OLD algunas veces es muy agresivo)
    df_f = df_sql[(df_sql["BAD"] == 0)]
    if df_f.empty:
        return None, None, None, None, pd.DataFrame(), (None, None)

    # Remuestreo a 1H + gap fill suave en VALUE (hasta 3h)
    df = df_f.set_index("TIME_FULL").sort_index()
    hourly_val = df["VALUE"].resample("1H").mean()
    hourly_val = hourly_val.interpolate(limit=3, limit_direction="both")

    # 2) Clima histórico (si disponible)
    lat, lon = get_coords_para_equipo(equip_grp, equipment)
    clima = None
    if lat is not None and lon is not None:
        try:
            clima = fetch_weather_history_influx(equip_grp, start_dt, end_dt)
        except Exception:
            clima = None

    if isinstance(clima, pd.DataFrame) and not clima.empty and "time" in clima.columns:
        clima = clima.set_index("time").sort_index()
        for c in ("temperature","relative_humidity","windspeed","winddirection"):
            if c not in clima.columns:
                clima[c] = np.nan
        clima_h = clima[["temperature","relative_humidity","windspeed","winddirection"]].resample("1H").mean()
        df_hourly = pd.DataFrame({"VALUE": hourly_val}).join(clima_h, how="left")
    else:
        df_hourly = pd.DataFrame({"VALUE": hourly_val})
        for c in ("temperature","relative_humidity","windspeed","winddirection"):
            df_hourly[c] = np.nan

    # Interpolar clima y rellenar restos (features) a 0
    for c in ("temperature","relative_humidity","windspeed","winddirection"):
        df_hourly[c] = df_hourly[c].interpolate(limit_direction="both").fillna(0.0)

    # Quitar filas sin VALUE
    df_hourly = df_hourly.dropna(subset=["VALUE"])

    # Features de hora del día (como hacía la app vieja)
    hours = df_hourly.index.hour.astype(float)
    df_hourly["hora_sin"] = np.sin(2*np.pi*(hours/24.0))
    df_hourly["hora_cos"] = np.cos(2*np.pi*(hours/24.0))

    # Orden de features (VALUE primero)
    feats = ["VALUE","temperature","relative_humidity","windspeed","hora_sin","hora_cos"]
    feats = [c for c in feats if c in df_hourly.columns]

    # Ventanas
    values = df_hourly[feats].values.astype("float32")
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i, 0])  # siguiente VALUE
    if not X:
        return None, None, feats, df_hourly.index.max(), df_hourly, (None, None)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    y_min = float(np.nanmin(df_hourly["VALUE"])) if len(df_hourly) else None
    y_max = float(np.nanmax(df_hourly["VALUE"])) if len(df_hourly) else None
    return X, y, feats, df_hourly.index.max(), df_hourly, (y_min, y_max)

def _get_forecast_df(equip_grp, equipment, steps):
    """
    Devuelve un DataFrame con columnas:
      time, temperature, relative_humidity, windspeed, winddirection, hora_sin, hora_cos
    para las próximas 'steps' horas. Aglutina TODOS los 'forecastday',
    usa time_epoch si viene, redondea a la hora y se queda sólo con FUTURO.
    """
    import pandas as pd
    import numpy as np

    if forecast_hourly is None:
        return None

    lat, lon = get_coords_para_equipo(equip_grp, equipment)
    if lat is None or lon is None:
        return None

    try:
        data = forecast_hourly(lat, lon)
    except Exception:
        return None

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    rows = []
    try:
        days = data.get('forecast', {}).get('forecastday', [])
        for d in days:
            for h in d.get('hour', []):
                if 'time_epoch' in h:
                    ts = pd.to_datetime(int(h['time_epoch']), unit='s')
                else:
                    ts = pd.to_datetime(h.get('time') or h.get('date') or h.get('datetime'))
                rows.append({
                    'time': ts,
                    'temperature': to_float(h.get('temp_c', h.get('temperature'))),
                    'relative_humidity': to_float(h.get('humidity', h.get('relative_humidity'))),
                    'windspeed': to_float(h.get('wind_kph', h.get('windspeed'))),
                    'winddirection': to_float(h.get('wind_degree', h.get('winddirection'))),
                })
    except Exception:
        rows = []

    if not rows:
        return None

    df = pd.DataFrame(rows).dropna(subset=['time'])
    # Sólo futuro, redondeado a la hora y sin duplicados
    now = pd.Timestamp.now().floor('H')
    df = df[df['time'] > now].copy()
    if df.empty:
        return None

    df['time'] = pd.to_datetime(df['time']).dt.floor('H')
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')

    # hora del día para series cíclicas
    hrs = df['time'].dt.hour.astype(float)
    df['hora_sin'] = np.sin(2 * np.pi * (hrs / 24.0))
    df['hora_cos'] = np.cos(2 * np.pi * (hrs / 24.0))

    return df.head(steps)



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

def train_or_update(equip_grp, equipment, signal_id, start_dt, end_dt, lookback=24, epochs=8):
    """
    Entrena o reentrena. Si no hay TF, usa MLP + escalado en X e Y (estable).
    Guarda meta con feats, lookback, último ts y límites de y para clipping.
    """
    ensure_dirs()
    backend = "lstm" if tf is not None else "mlp"
    paths = _paths(equip_grp, equipment, signal_id, backend_hint=backend)
    pmodel, pmeta = paths["model"], paths["meta"]
    meta = _load_meta(pmeta)

    # Dataset con lookback adaptativo
    tried = []
    X = y = feats = last_ts = df_hourly = None
    y_bounds = (None, None)
    for lb in (lookback, 12, 6):
        X, y, feats, last_ts, df_hourly, y_bounds = _prepare_dataset(
            equip_grp, equipment, signal_id, start_dt, end_dt, lookback=lb
        )
        tried.append({"lookback": lb, "n_samples": 0 if X is None else len(X)})
        if X is not None and len(X) > 0:
            lookback = lb
            break

    if X is None or len(X) == 0:
        horas_validas = 0
        try:
            if df_hourly is not None and not df_hourly.empty:
                horas_validas = df_hourly.shape[0]
        except Exception:
            pass
        return {"trained": False, "reason": f"No hay ventanas suficientes. Horas útiles={horas_validas}. Intentos={tried}"}

    # Entrenar
    if backend == "lstm":
        model = _build_lstm((X.shape[1], X.shape[2]))
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
        model.save(str(pmodel))
    else:
        # Aplanar y escalar X e Y
        N, L, F = X.shape
        X2 = X.reshape(N, L*F)
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128,64),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=42
            ))
        ])
        # Escalar también Y y revertir automáticamente
        model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
        model.fit(X2, y.ravel())
        joblib.dump(model, pmodel)

    # Guardar meta (con margen 10% para clipping)
    y_min, y_max = y_bounds
    if y_min is not None and y_max is not None:
        span = max(1e-6, (y_max - y_min))
        y_min_clip = y_min - 0.1*span
        y_max_clip = y_max + 0.1*span
    else:
        y_min_clip = y_min
        y_max_clip = y_max

    meta.update({
        "feats": feats,
        "lookback": lookback,
        "last_train_end": str(last_ts),
        "backend": backend,
        "y_min": y_min_clip,
        "y_max": y_max_clip
    })
    _save_meta(pmeta, meta)

    return {"trained": True, "samples": int(len(X)), "last_ts": str(last_ts), "model_path": str(pmodel), "backend": backend}



def forecast_next_hours(equip_grp, equipment, signal_id, steps=24):
    """
    Proyección autoregresiva para 'steps' horas.
    Usa exógenos del pronóstico; si no hay, mantiene exógenos y actualiza hora_sin/cos.
    Alinea cada hora futura con la fila de pronóstico MÁS CERCANA (±59 min).
    """
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    paths = _paths(equip_grp, equipment, signal_id)
    pmodel, pmeta = paths["model"], paths["meta"]
    if not pmodel.exists():
        raise RuntimeError("No hay modelo entrenado para esta señal. Entrena primero.")

    meta = _load_meta(pmeta)
    lookback = int(meta.get("lookback", 24))
    feats = meta.get("feats", ["VALUE","temperature","relative_humidity","windspeed","hora_sin","hora_cos"])
    backend = meta.get("backend", "lstm" if tf is not None else "mlp")
    y_min = meta.get("y_min", None)
    y_max = meta.get("y_max", None)

    # Últimas ventanas para arrancar
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    Xh, yh, feats2, last_ts, df_hourly, _ = _prepare_dataset(
        equip_grp, equipment, signal_id,
        start_dt=now - timedelta(days=30),
        end_dt=now,
        lookback=lookback
    )
    if df_hourly is None or df_hourly.empty or len(df_hourly) < lookback:
        Xh, yh, feats2, last_ts, df_hourly, _ = _prepare_dataset(
            equip_grp, equipment, signal_id,
            start_dt=now - timedelta(days=120),
            end_dt=now,
            lookback=lookback
        )
        if df_hourly is None or df_hourly.empty or len(df_hourly) < lookback:
            raise RuntimeError("No hay datos recientes suficientes para proyectar.")

    # Alinear features
    for c in feats:
        if c not in df_hourly.columns:
            df_hourly[c] = 0.0
    last_window = df_hourly[feats].values.astype("float32")[-lookback:]

    # Cargar modelo
    if backend == "lstm":
        model = tf.keras.models.load_model(str(pmodel))
    else:
        model = joblib.load(pmodel)

    # Pronóstico exógeno (horario)
    forecast_df = _get_forecast_df(equip_grp, equipment, steps)
    use_forecast = isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty
    fdf = forecast_df.set_index('time').sort_index() if use_forecast else None

    preds = []
    curr_ts = df_hourly.index.max()
    window = last_window.copy()

    for i in range(steps):
        future_ts = curr_ts + timedelta(hours=1)

        # 1) Predecir
        if backend == "lstm":
            x_in = window.reshape(1, lookback, len(feats))
            yhat = float(model.predict(x_in, verbose=0)[0,0])
        else:
            x_in = window.reshape(1, lookback * len(feats))
            yhat = float(model.predict(x_in)[0])

        # 2) Clip al rango entrenado (evita explosiones)
        if (y_min is not None) and (y_max is not None):
            yhat = max(min(yhat, float(y_max)), float(y_min))

        preds.append((future_ts, yhat))

        # 3) Construir siguiente fila: yhat en VALUE
        next_row = window[-1].copy()
        next_row[0] = yhat

        # Actualizar hora_sin/cos
        h = float(future_ts.hour)
        hsin = np.sin(2 * np.pi * (h / 24.0))
        hcos = np.cos(2 * np.pi * (h / 24.0))
        if "hora_sin" in feats: next_row[feats.index("hora_sin")] = hsin
        if "hora_cos" in feats: next_row[feats.index("hora_cos")] = hcos

        # 4) Inyectar exógenos del pronóstico (fila más cercana)
        if fdf is not None:
            try:
                target = pd.to_datetime(future_ts).floor('H')
                if target in fdf.index:
                    row = fdf.loc[target]
                else:
                    idxpos = fdf.index.get_indexer([target], method='nearest')
                    row = fdf.iloc[int(idxpos[0])] if idxpos[0] != -1 else None
                if row is not None:
                    for name in ('temperature','relative_humidity','windspeed','winddirection','hora_sin','hora_cos'):
                        if name in feats and name in fdf.columns:
                            j = feats.index(name)
                            val = float(row[name]) if pd.notna(row[name]) else None
                            if val is not None and j < len(next_row):
                                next_row[j] = val
            except Exception:
                pass

        # 5) Desplazar ventana
        window = np.vstack([window[1:], next_row])
        curr_ts = future_ts

    out_df = pd.DataFrame(preds, columns=["TIME_FULL","FORECAST_VALUE"])
    return out_df


def predict_last_hours(equip_grp, equipment, signal_id, hours=48):
    """
    Predice AUTORREGRESIVO sobre las ÚLTIMAS 'hours' horas usando los
    exógenos OBSERVADOS (no pronóstico), para comparar Real vs Predicha
    en la misma ventana. Devuelve (df_pred, serie_real).
    """
    paths = _paths(equip_grp, equipment, signal_id)
    pmodel, pmeta = paths["model"], paths["meta"]
    if not pmodel.exists():
        raise RuntimeError("No hay modelo entrenado para esta señal.")

    meta = _load_meta(pmeta)
    lookback = int(meta.get("lookback", 24))
    feats = meta.get("feats", ["VALUE","temperature","relative_humidity","windspeed","hora_sin","hora_cos"])
    backend = meta.get("backend", "lstm" if tf is not None else "mlp")
    y_min = meta.get("y_min", None)
    y_max = meta.get("y_max", None)

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    # margen para armar la 1ra ventana
    start_dt = now - timedelta(hours=hours + max(lookback, 24))

    Xh, yh, feats2, last_ts, df_hourly, _ = _prepare_dataset(
        equip_grp, equipment, signal_id, start_dt, now, lookback=lookback
    )
    if df_hourly is None or df_hourly.empty:
        raise RuntimeError("No hay datos suficientes para backtest.")

    # asegurar features
    for c in feats:
        if c not in df_hourly.columns:
            df_hourly[c] = 0.0

    # índice de inicio para cubrir EXACTAMENTE las últimas 'hours' horas
    start_idx = max(lookback, len(df_hourly) - hours)
    if start_idx + hours > len(df_hourly):
        hours = len(df_hourly) - start_idx
    if hours <= 0:
        raise RuntimeError("Ventana demasiado corta para backtest.")

    window = df_hourly[feats].values.astype("float32")[start_idx - lookback : start_idx]

    # cargar modelo
    if backend == "lstm":
        model = tf.keras.models.load_model(str(pmodel))
    else:
        model = joblib.load(pmodel)

    preds = []
    times = []

    for i in range(hours):
        # predecir 1 paso
        if backend == "lstm":
            x_in = window.reshape(1, lookback, len(feats))
            yhat = float(model.predict(x_in, verbose=0)[0,0])
        else:
            x_in = window.reshape(1, lookback * len(feats))
            yhat = float(model.predict(x_in)[0])

        # clip a rango entrenado (evita explosiones)
        if (y_min is not None) and (y_max is not None):
            yhat = max(min(yhat, float(y_max)), float(y_min))

        t = df_hourly.index[start_idx + i]
        preds.append(yhat)
        times.append(t)

        # avanzar ventana: usar exógenos OBSERVADOS en t y el yhat como VALUE
        next_exog = df_hourly[feats].iloc[start_idx + i].values.astype("float32").copy()
        next_exog[0] = yhat
        window = np.vstack([window[1:], next_exog])

    df_pred = pd.DataFrame({"TIME_FULL": times, "PRED_VALUE": preds})
    serie_real = df_hourly.iloc[start_idx:start_idx+hours]["VALUE"].copy()
    return df_pred, serie_real

