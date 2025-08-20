import pandas as pd

def unify_index_tz(df, to_tz='America/Argentina/Buenos_Aires', drop_tz=True):
    """
    Normaliza el índice temporal de df:
    - Fuerza a UTC si venía naive
    - Convierte a la zona pedida
    - Opcionalmente elimina información de TZ para evitar conflictos aguas arriba
    """
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, 'tz', None) is None:
        # naive -> hacemos de cuenta que es UTC
        idx = idx.tz_localize('UTC')
    # Convertimos a la TZ de trabajo
    if to_tz:
        idx = idx.tz_convert(to_tz)
    if drop_tz:
        idx = idx.tz_localize(None)
    out.index = idx
    return out
