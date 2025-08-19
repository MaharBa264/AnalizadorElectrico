import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

def comparar_corriente_vs_temp(df):
    """
    Devuelve una tabla HTML comparando corriente (VALUE) vs temperatura (temperature).
    Si no existe la columna 'temperature', retorna mensaje adecuado.
    """
    if "temperature" not in df.columns:
        return "<b>No hay columna de temperatura disponible para comparar.</b>"

    # Solo las filas que tienen ambos valores
    comp_df = df[["TIME_FULL", "VALUE", "temperature"]].dropna()
    if comp_df.empty:
        return "<b>No hay datos de corriente y temperatura coincidentes para comparar.</b>"
    
    comp_df = comp_df.rename(columns={
        "TIME_FULL": "Fecha-Hora",
        "VALUE": "Corriente",
        "temperature": "Temperatura"
    })
    html_table = comp_df.to_html(classes="table table-striped", index=False)
    return html_table

def graficar_corriente_vs_temp(df):
    """
    Devuelve el gráfico corriente vs temperatura en base64 (para insertar en <img src=...>)
    Si falta columna 'temperature', retorna None.
    """
    if "temperature" not in df.columns:
        return None

    valid_df = df.dropna(subset=["VALUE", "temperature"])
    if valid_df.empty:
        return None

    plt.figure(figsize=(7,4))
    plt.scatter(valid_df["temperature"], valid_df["VALUE"], alpha=0.7)
    plt.xlabel("Temperatura [°C]")
    plt.ylabel("Corriente [A]")
    # (en tu archivo original hay una elipsis '...' aquí; la respetamos)
    try:
        z = np.polyfit(valid_df["temperature"], valid_df["VALUE"], 1)
        p = np.poly1d(z)
        plt.plot(valid_df["temperature"], p(valid_df["temperature"]), "--", label="Ajuste lineal")
        plt.legend()
    except Exception:
        pass
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64
