"""
Proyección (stub).
"""

import os
from app.services.files import MODELS
from .model import load_model, load_scaler

def one_step_forecast(last_values, model_name: str, lookback=24):
    """
    last_values: lista de floats en escala ORIGINAL (no escalada)
    """
    model_path = os.path.join(MODELS, f"{model_name}.h5")
    scaler_path = os.path.join(MODELS, f"{model_name}_scaler.pkl")

    m = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Escalar ventana
    window = last_values[-lookback:]
    sw = scaler.transform(window)
    X3 = [[[v] for v in sw]]  # shape (1, timesteps, features)

    yhat_scaled = m.predict(X3, verbose=0)[0][0]
    yhat = scaler.inverse_transform([yhat_scaled])[0]
    return yhat
