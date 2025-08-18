"""
Creación/carga de modelo. Intenta usar TensorFlow si está disponible.
Sin TensorFlow, levanta una excepción clara al entrenar.
"""

import os
import pickle

def build_lstm(input_dim: int):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, InputLayer
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras no está disponible en este entorno.") from e

    m = Sequential()
    m.add(InputLayer(input_shape=(input_dim, 1)))
    m.add(LSTM(32))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m

def save_model(path: str, model):
    try:
        model.save(path)
    except Exception as e:
        raise RuntimeError(f"No se pudo guardar el modelo en {path}: {e}")

def load_model(path: str):
    try:
        from tensorflow.keras.models import load_model as _load
        return _load(path)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo desde {path}: {e}")

class MinMaxScalerLite:
    """Scaler mínimo sin sklearn (compatible Py3.6)."""
    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, arr):
        self._min = min(arr) if arr else 0.0
        self._max = max(arr) if arr else 1.0

    def transform(self, arr):
        if self._max == self._min:
            return [0.0 for _ in arr]
        return [(v - self._min) / (self._max - self._min) for v in arr]

    def inverse_transform(self, arr):
        return [v * (self._max - self._min) + self._min for v in arr]

def save_scaler(path: str, scaler):
    with open(path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_scaler(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
