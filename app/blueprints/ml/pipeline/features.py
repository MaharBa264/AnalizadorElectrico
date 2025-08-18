"""
Construcción de features supervisadas (ventaneo).
No depende de pandas/sklearn.
"""

def make_supervised(values, lookback=24, horizon=1, step=1):
    """
    values: lista/iterable de floats
    lookback: cuántos pasos mirar hacia atrás
    horizon: qué tan adelante predecimos (1 = siguiente)
    step: stride en el barrido
    Retorna X (lista de listas), y (lista)
    """
    x, y = [], []
    n = len(values)
    end = n - horizon
    i = lookback
    while i <= end:
        past = values[i - lookback:i]
        target = values[i + horizon - 1]
        x.append(list(past))
        y.append(float(target))
        i += step
    return x, y
