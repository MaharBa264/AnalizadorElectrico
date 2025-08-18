"""
Entrenamiento incremental (stub). No usa sklearn; scaler propio.
"""

import os
from app.services.files import MODELS
from .model import build_lstm, MinMaxScalerLite, save_model, save_scaler
from .features import make_supervised

def train_incremental(values, model_name: str,
                      lookback=24, horizon=1, epochs=3, batch_size=32):
    """
    values: lista de floats (serie ya limpiada)
    Devuelve dict con info básica. Guarda modelo/scaler en storage/models/lstm/
    """
    os.makedirs(MODELS, exist_ok=True)
    model_path = os.path.join(MODELS, f"{model_name}.h5")
    scaler_path = os.path.join(MODELS, f"{model_name}_scaler.pkl")

    # Escalado simple
    scaler = MinMaxScalerLite()
    scaler.fit(values)
    scaled = scaler.transform(values)

    # Features
    X, y = make_supervised(scaled, lookback=lookback, horizon=horizon)
    if not X or not y:
        return {"trained": False, "message": "Serie insuficiente para entrenar."}

    # Dar forma (samples, timesteps, features)
    X3 = [[[v] for v in row] for row in X]

    # Modelo
    m = build_lstm(len(X[0]))

    # Entrenar
    try:
        m.fit(X3, y, epochs=epochs, batch_size=batch_size, verbose=0)
    except Exception as e:
        return {"trained": False, "message": f"Error entrenando: {e}"}

    # Guardar
    try:
        save_model(model_path, m)
        save_scaler(scaler_path, scaler)
    except Exception as e:
        return {"trained": True, "saved": False, "message": f"Modelo entrenado pero no guardado: {e}"}

    return {"trained": True, "saved": True, "model_path": model_path, "scaler_path": scaler_path}
