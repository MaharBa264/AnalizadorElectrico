import json
import os
import pandas as pd
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, send_file
from flask_login import login_required
from app.services.weather import forecast_hourly
from . import bp
from .services import get_signal_hierarchy, get_coords_para_equipo, fetch_sql_series, fetch_weather_history, make_trend_payload, fetch_weather_history_influx
from .compare import comparar_corriente_vs_temp, graficar_corriente_vs_temp
from io import StringIO, BytesIO

def _parse_dt(s, default=None):
    if not s:
        return default or datetime.now().replace(minute=0, second=0, microsecond=0)
    s = s.strip()
    fmt = "%Y-%m-%dT%H:%M" if "T" in s else ("%Y-%m-%d %H:%M" if " " in s else "%Y-%m-%d")
    return datetime.strptime(s, fmt)

@bp.route("/", methods=["GET"])
@login_required
def index():
    estructura = get_signal_hierarchy()
    return render_template("analizador/index.html", estructura=estructura)

@bp.route("/analizar_sql", methods=["POST"])
@login_required
def analizar_sql():
    equip_grp = request.form.get("equip_grp","").strip()
    equipment = request.form.get("equipment","").strip()
    signal_id = request.form.get("signal_id","").strip()
    comparar_chk = (request.form.get("comparar_con_clima") == "1"
                    or request.form.get("analizar_sql_vs_temp") == "1")
    fecha_ini = _parse_dt(request.form.get("fecha_ini"), default=datetime.now()-timedelta(days=7))
    fecha_fin = _parse_dt(request.form.get("fecha_fin"), default=datetime.now())

    # 1) SQL
    try:
        df_sql = fetch_sql_series(equip_grp, equipment, signal_id, fecha_ini, fecha_fin)
    except Exception as e:
        flash(f"Error consultando SQL: {e}", "danger")
        return redirect(url_for("analizador.index"))

    # Datos tabulares para la plantilla "resultados_sql.html"
    datos = []
    if not df_sql.empty:
        df_aux = df_sql.copy()
        df_aux["Fecha"] = df_aux["TIME_FULL"].dt.strftime("%Y-%m-%d")
        df_aux["Hora"] = df_aux["TIME_FULL"].dt.strftime("%H:%M:%S")
        for _, r in df_aux.iterrows():
            datos.append({
                "Fecha": r["Fecha"], "Hora": r["Hora"],
                "VALUE": None if pd.isna(r["VALUE"]) else float(r["VALUE"]),
                "OLD": int(r["OLD"]), "BAD": int(r["BAD"])
            })

    corrientes = [d['VALUE'] for d in datos if d['VALUE'] is not None]
    min_value = min(corrientes) if corrientes else None
    max_value = max(corrientes) if corrientes else None
    avg_value = (sum(corrientes)/len(corrientes)) if corrientes else None
    total_muestras = len(corrientes)
    min_time = df_sql.loc[df_sql["VALUE"].idxmin(),"TIME_FULL"].strftime("%Y-%m-%dT%H:%M:%S") if not df_sql.empty else None
    max_time = df_sql.loc[df_sql["VALUE"].idxmax(),"TIME_FULL"].strftime("%Y-%m-%dT%H:%M:%S") if not df_sql.empty else None
    hay_datos_invalidos = bool((df_sql["OLD"]==1).any() or (df_sql["BAD"]==1).any())

    time_data = make_trend_payload(df_sql)

    if not comparar_chk:
        return render_template(
            "analizador/resultados_sql.html",
            datos=datos,
            min_value=min_value, max_value=max_value,
            min_time=min_time, max_time=max_time,
            avg_value=avg_value, total_muestras=total_muestras,
            equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
            time_data=time_data,
            fecha_ini=fecha_ini.strftime("%Y-%m-%d %H:%M"),
            fecha_fin=fecha_fin.strftime("%Y-%m-%d %H:%M"),
            comparar_chk=comparar_chk,
        )

    # 2) Comparación con clima (histórico: Influx; pronóstico: WeatherAPI)
    lat, lon = get_coords_para_equipo(equip_grp, equipment)
    station_info = None
    if lat is None or lon is None:
        flash("No hay coordenadas asociadas al equipo para obtener pronóstico.", "warning")
        # seguimos sin clima: devolvemos solo la vista SQL
        return render_template(
            "analizador/resultados_sql.html",
            datos=datos,
            min_value=min_value, max_value=max_value,
            min_time=min_time, max_time=max_time,
            avg_value=avg_value, total_muestras=total_muestras,
            equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
            time_data=time_data,
            fecha_ini=fecha_ini.strftime("%Y-%m-%d %H:%M"),
            fecha_fin=fecha_fin.strftime("%Y-%m-%d %H:%M"),
            comparar_chk=comparar_chk,
        )
    else:
        station_info = {"latitud": lat, "longitud": lon}

    # Histórico horario de clima → primero Influx, si no hay datos intentamos WeatherAPI (fallback)
    dfw = fetch_weather_history_influx(equip_grp, fecha_ini, fecha_fin)
    if dfw.empty:
        try:
            dfw = fetch_weather_history(lat, lon, fecha_ini, fecha_fin)
        except Exception:
            dfw = pd.DataFrame(columns=["time","temperature","relative_humidity","windspeed","winddirection"])

    # Garantiza datetime (por si el driver devolvió strings/objeto)
    if not pd.api.types.is_datetime64_any_dtype(df_sql["TIME_FULL"]):
        df_sql["TIME_FULL"] = pd.to_datetime(df_sql["TIME_FULL"], errors="coerce", infer_datetime_format=True)

    # Promedio horario de corriente y merge por hora
    sql_hour = (
        df_sql
        .dropna(subset=["VALUE", "TIME_FULL"])
        .assign(TIME_H=lambda x: pd.to_datetime(x["TIME_FULL"], errors="coerce").dt.floor("H"))
        .groupby("TIME_H", as_index=False)
        .agg(VALUE=("VALUE","mean"))
    )
    merged = sql_hour.merge(dfw, left_on="TIME_H", right_on="time", how="left")
    merged.rename(columns={"TIME_H":"time"}, inplace=True)

    # === NUEVO: datos para gráficos D3 ===
    # 1) Scatter (Temp vs Corriente) -> necesita keys: temperature, VALUE
    scatter_df = merged.dropna(subset=["VALUE","temperature"]).copy()
    try:
        scatter_df["VALUE"] = pd.to_numeric(scatter_df["VALUE"], errors="coerce")
        scatter_df["temperature"] = pd.to_numeric(scatter_df["temperature"], errors="coerce")
    except Exception:
        pass
    scatter_df = scatter_df.dropna(subset=["VALUE","temperature"])
    scatter_data = scatter_df[["temperature","VALUE"]].to_dict(orient="records")

    # 2) Serie de tiempo (ya la tenés en time_data)

    # Tabla HTML para "Datos Detallados"
    table_html = (merged[["time","VALUE","temperature","relative_humidity","windspeed","winddirection"]]
                  .rename(columns={
                    "time":"Fecha-Hora","VALUE":"Corriente","temperature":"Temperatura",
                    "relative_humidity":"Humedad","windspeed":"Viento(kph)","winddirection":"Dir.Viento"
                  })
                  .to_html(classes="table table-striped table-sm", index=False, na_rep='-'))

    # Correlación y regresión
    correlation = None
    slope = None
    intercept = None
    if not scatter_df.empty:
        try:
            correlation = float(scatter_df["VALUE"].corr(scatter_df["temperature"]))
        except Exception:
            correlation = None
        try:
            import numpy as np
            slope, intercept = np.polyfit(scatter_df["temperature"], scatter_df["VALUE"], 1)
        except Exception:
            slope = intercept = None

    # Pronóstico próximo 3 días (WeatherAPI) -> requiere WEATHERAPI_KEY
    forecast_list = []
    forecast_debug = None  # <-- para mostrar en la tarjeta de resumen

    if slope is None or intercept is None:
        forecast_debug = "Sin regresión: no hay suficientes pares válidos (temperatura/corriente) para estimar la recta."
    else:
        api_key = os.getenv("WEATHERAPI_KEY", "")
        if not api_key:
            forecast_debug = "Falta WEATHERAPI_KEY en .env (no se puede consultar WeatherAPI)."
        else:
            try:
                fc = forecast_hourly(lat, lon, days=3)
                days = fc.get("forecast", {}).get("forecastday", [])
                if not days:
                    forecast_debug = "WeatherAPI no devolvió 'forecast.forecastday'."
                else:
                    for day in days:
                        date = day.get("date")
                        dayblock = day.get("day", {}) or {}
                        tmax = dayblock.get("maxtemp_c")
                        tmin = dayblock.get("mintemp_c")
                        forecast_list.append({
                            "date": date,
                            "forecast_max": tmax,
                            "predicted_current_max": (slope*tmax + intercept) if tmax is not None else None,
                            "forecast_min": tmin,
                            "predicted_current_min": (slope*tmin + intercept) if tmin is not None else None
                        })
                    if not forecast_list:
                        forecast_debug = "No hubo temperaturas máximas/mínimas en la respuesta de WeatherAPI."
            except Exception as e:
                forecast_debug = "Error al pedir pronóstico a WeatherAPI: {}".format(e)
    # Fuente del histórico usada: Influx si trajo algo; si no, WeatherAPI; si ninguna, 'N/A'
    hist_source = "Influx" if not dfw.empty else ("WeatherAPI" if "dfw" in locals() and isinstance(dfw, pd.DataFrame) else "N/A")

    # LOG para depurar rápido
    from flask import current_app
    current_app.logger.info(
        "ANALIZADOR DEBUG | hist_source=%s | sql_rows=%s | sql_hour_rows=%s | clima_rows=%s | scatter_points=%s | has_reg=%s | forecast_days=%s | reason=%s",
        hist_source, len(df_sql), len(sql_hour), len(dfw) if isinstance(dfw, pd.DataFrame) else 0,
        len(scatter_data), slope is not None, len(forecast_list), forecast_debug
    )


    return render_template(
        "analizador/results.html",
        station_label=f"{equip_grp} / {equipment} / {signal_id}",
        table=table_html,
        time_data=time_data,
        scatter_data=scatter_data,     # <-- para D3
        correlation=correlation,
        forecast_list=forecast_list,
        forecast_debug=forecast_debug, # <-- NUEVO
        hist_source=hist_source,       # <-- NUEVO
        station_info=station_info,
        fecha_ini = fecha_ini.strftime("%Y-%m-%dT%H:%M:%S"),
        fecha_fin = fecha_fin.strftime("%Y-%m-%dT%H:%M:%S"),
        comparar_chk=comparar_chk,
    )




@bp.route("/exportar_xlsx", methods=["POST"])
@login_required
def exportar_xlsx():
    equip_grp = request.form.get("equip_grp","").strip()
    equipment = request.form.get("equipment","").strip()
    signal_id = request.form.get("signal_id","").strip()
    comparar_chk = (request.form.get("comparar_con_clima") == "1"
                    or request.form.get("analizar_sql_vs_temp") == "1")

    def _parse_dt(s, default):
        if not s:
            return default
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return default
    fecha_ini = _parse_dt(request.form.get("fecha_ini"), datetime.now()-timedelta(days=7))
    fecha_fin = _parse_dt(request.form.get("fecha_fin"), datetime.now())

    # SQL
    df_sql = fetch_sql_series(equip_grp, equipment, signal_id, fecha_ini, fecha_fin).copy()
    if not df_sql.empty:
        df_sql["Fecha"] = df_sql["TIME_FULL"].dt.strftime("%Y-%m-%d")
        df_sql["Hora"]  = df_sql["TIME_FULL"].dt.strftime("%H:%M:%S")
        df_sql_out = df_sql[["Fecha","Hora","VALUE","OLD","BAD"]]
    else:
        df_sql_out = df_sql

    # Comparación con clima (opcional)
    merged_out = None
    if comparar_chk:
        lat, lon = get_coords_para_equipo(equip_grp, equipment)
        if lat is not None and lon is not None:
            dfw = fetch_weather_history(lat, lon, fecha_ini, fecha_fin)
            sql_hour = (df_sql.dropna(subset=["VALUE"])
                        .assign(TIME_H=df_sql["TIME_FULL"].dt.floor("H"))
                        .groupby("TIME_H", as_index=False).agg(VALUE=("VALUE","mean")))
            merged = sql_hour.merge(dfw, left_on="TIME_H", right_on="time", how="left")
            merged.rename(columns={"TIME_H":"time"}, inplace=True)
            merged_out = merged[["time","VALUE","temperature","relative_humidity","windspeed","winddirection"]]
            merged_out.rename(columns={
                "time":"Fecha-Hora","VALUE":"Corriente",
                "temperature":"Temperatura","relative_humidity":"Humedad",
                "windspeed":"Viento(kph)","winddirection":"Dir.Viento"
            }, inplace=True)

    # Escribir a Excel en memoria
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_sql_out.to_excel(writer, index=False, sheet_name="SQL")
        if merged_out is not None:
            merged_out.to_excel(writer, index=False, sheet_name="Comparacion")
    bio.seek(0)

    filename = f"analisis_{equip_grp}_{equipment}_{signal_id}.xlsx"
    return send_file(
        bio,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@bp.route("/exportar_csv", methods=["POST"])
@login_required
def exportar_csv():
    equip_grp = request.form.get("equip_grp","").strip()
    equipment = request.form.get("equipment","").strip()
    signal_id = request.form.get("signal_id","").strip()
    comparar_chk = (request.form.get("comparar_con_clima") == "1" or request.form.get("analizar_sql_vs_temp") == "1")

    def _parse_dt(s, default):
        if not s:
            return default
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return default


    fecha_ini = _parse_dt(request.form.get("fecha_ini"), datetime.now()-timedelta(days=7))
    fecha_fin = _parse_dt(request.form.get("fecha_fin"), datetime.now())

    df_sql = fetch_sql_series(equip_grp, equipment, signal_id, fecha_ini, fecha_fin).copy()
    if not df_sql.empty:
        df_sql["Fecha"] = df_sql["TIME_FULL"].dt.strftime("%Y-%m-%d")
        df_sql["Hora"]  = df_sql["TIME_FULL"].dt.strftime("%H:%M:%S")
        df_sql_out = df_sql[["Fecha","Hora","VALUE","OLD","BAD"]]
    else:
        df_sql_out = df_sql

    merged_out = None
    if comparar_chk:
        lat, lon = get_coords_para_equipo(equip_grp, equipment)
        if lat is not None and lon is not None:
            dfw = fetch_weather_history_influx(equip_grp, fecha_ini, fecha_fin)
            if dfw.empty:
                try:
                    dfw = fetch_weather_history(lat, lon, fecha_ini, fecha_fin)
                except Exception:
                    dfw = pd.DataFrame(columns=["time","temperature","relative_humidity","windspeed","winddirection"])
            if not df_sql.empty:
                sql_hour = (df_sql.dropna(subset=["VALUE"])
                            .assign(TIME_H=lambda x: pd.to_datetime(x["TIME_FULL"], errors="coerce").dt.floor("H"))
                            .groupby("TIME_H", as_index=False).agg(VALUE=("VALUE","mean")))
                merged = sql_hour.merge(dfw, left_on="TIME_H", right_on="time", how="left")
                merged.rename(columns={"TIME_H":"time"}, inplace=True)
                merged_out = merged[["time","VALUE","temperature","relative_humidity","windspeed","winddirection"]]
                merged_out.rename(columns={
                    "time":"Fecha-Hora","VALUE":"Corriente","temperature":"Temperatura",
                    "relative_humidity":"Humedad","windspeed":"Viento(kph)","winddirection":"Dir.Viento"
                }, inplace=True)

    sio = StringIO()
    if merged_out is not None:
        merged_out.to_csv(sio, index=False)
        fname = f"analisis_{equip_grp}_{equipment}_{signal_id}_comparacion.csv"
    else:
        df_sql_out.to_csv(sio, index=False)
        fname = f"analisis_{equip_grp}_{equipment}_{signal_id}.csv"

    bio = BytesIO(sio.getvalue().encode("utf-8-sig"))
    return send_file(bio, as_attachment=True, download_name=fname, mimetype="text/csv; charset=utf-8")

@bp.route("/entrenar_lstm", methods=["POST"])
@login_required
def entrenar_lstm():
    from app.services.lstm import train_or_update
    equip_grp = request.form.get("equip_grp","").strip()
    equipment = request.form.get("equipment","").strip()
    signal_id = request.form.get("signal_id","").strip()

    # Por defecto: 120 días de historial
    fecha_ini = _parse_dt(request.form.get("fecha_ini"), datetime.now() - timedelta(days=120))
    fecha_fin = _parse_dt(request.form.get("fecha_fin"), datetime.now())

    try:
        info = train_or_update(equip_grp, equipment, signal_id, fecha_ini, fecha_fin, lookback=24, epochs=8)
        if info.get("trained"):
            backend = info.get('backend','lstm')
            nombre = 'LSTM' if backend=='lstm' else 'MLP (fallback)'
            flash(f"✅ {nombre} entrenada ({info.get('samples')} muestras). Última hora: {info.get('last_ts')}", "success")
        else:
            flash(f"⚠️ No se pudo entrenar: {info.get('reason','')}", "warning")
    except Exception as e:
        flash(f"❌ Error entrenando LSTM/MLP: {e}", "danger")

    # Al terminar, ir a la página de recap 48h
    return redirect(url_for("analizador.ver_lstm_ultimas48",
                            equip_grp=equip_grp, equipment=equipment, signal_id=signal_id))

@bp.route("/proyectar_lstm", methods=["POST"])
@login_required
def proyectar_lstm():
    from app.services.lstm import forecast_next_hours, train_or_update, _paths, _get_forecast_df
    equip_grp = request.form.get("equip_grp","").strip()
    equipment = request.form.get("equipment","").strip()
    signal_id = request.form.get("signal_id","").strip()

    pasos = int(request.form.get("pasos") or 96)
    train_if_missing = (request.form.get("train_if_missing") == "1")

    paths = _paths(equip_grp, equipment, signal_id)
    pmodel = paths["model"]
    if not pmodel.exists():
        if not train_if_missing:
            flash("⚠️ No hay modelo entrenado para esta señal. Marcá 'Entrenar si no hay modelo' o ejecutá el entrenamiento manual.", "warning")
            return redirect(url_for("analizador.index"))
        flash("ℹ️ No había modelo. Entrenando ahora (120 días) antes de proyectar…", "info")
        try:
            _ = train_or_update(
                equip_grp, equipment, signal_id,
                datetime.now() - timedelta(days=120),
                datetime.now(),
                lookback=24, epochs=8
            )
        except Exception as e:
            flash(f"❌ Error entrenando LSTM/MLP previo a la proyección: {e}", "danger")
            return redirect(url_for("analizador.index"))

    try:
        # Serie pronosticada de corriente
        df_pred = forecast_next_hours(equip_grp, equipment, signal_id, steps=pasos)
        labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in df_pred["TIME_FULL"]]
        values = [float(v) for v in df_pred["FORECAST_VALUE"]]

        # Temperatura de pronóstico para las mismas horas (eje derecho)
        try:
            fdf = _get_forecast_df(equip_grp, equipment, pasos)  # tiene 'time' y 'temperature'
            import pandas as pd
            if fdf is not None and not fdf.empty:
                tmap = {pd.to_datetime(t).strftime("%Y-%m-%d %H:%M"): float(v) if pd.notna(v) else None
                        for t, v in zip(fdf["time"], fdf["temperature"])}
                temps = [tmap.get(lbl, None) for lbl in labels]
            else:
                temps = [None]*len(labels)
        except Exception:
            temps = [None]*len(labels)

        return render_template("analizador/proyeccion_lstm.html",
            equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
            labels=json.dumps(labels), values=json.dumps(values), temps=json.dumps(temps),
            tabla=df_pred.to_html(classes="table table-striped table-sm", index=False)
        )
    except Exception as e:
        flash(f"❌ Error en proyección LSTM/MLP: {e}", "danger")
        return redirect(url_for("analizador.index"))


@bp.route("/ver_lstm_ultimas48", methods=["GET"])
@login_required
def ver_lstm_ultimas48():
    from .services import fetch_sql_series, fetch_weather_history_influx
    from app.services.lstm import predict_last_hours, forecast_next_hours
    import pandas as pd

    equip_grp = request.args.get("equip_grp","").strip()
    equipment = request.args.get("equipment","").strip()
    signal_id = request.args.get("signal_id","").strip()

    ahora = datetime.now().replace(minute=0, second=0, microsecond=0)
    ini_48 = ahora - timedelta(hours=48)

    # Serie real últimas 48 h (para ejes y tabla si querés)
    df_sql = fetch_sql_series(equip_grp, equipment, signal_id, ini_48, ahora).copy()
    df_sql = df_sql[(df_sql["BAD"] == 0)].sort_values("TIME_FULL")
    labels_48 = [ts.strftime("%Y-%m-%d %H:%M") for ts in pd.to_datetime(df_sql["TIME_FULL"])]
    real_48 = [None if pd.isna(v) else float(v) for v in df_sql["VALUE"]]

    # Predicción sobre las últimas 48 h (backtest con exógenos observados)
    try:
        df_pred, serie_real_bt = predict_last_hours(equip_grp, equipment, signal_id, hours=48)
        # alineamos a etiquetas de df_pred
        labels_bt = [ts.strftime("%Y-%m-%d %H:%M") for ts in df_pred["TIME_FULL"]]
        pred_48 = [float(v) for v in df_pred["PRED_VALUE"]]
        # real (del backtest) alineado a labels_bt
        real_bt = [float(v) if pd.notna(v) else None for v in serie_real_bt.values]
    except Exception:
        labels_bt, pred_48, real_bt = [], [], []

    # Forecast FUTURO 48 h (segundo gráfico)
    try:
        df_fut = forecast_next_hours(equip_grp, equipment, signal_id, steps=48)
        labels_f = [ts.strftime("%Y-%m-%d %H:%M") for ts in df_fut["TIME_FULL"]]
        values_f = [float(v) for v in df_fut["FORECAST_VALUE"]]
    except Exception:
        labels_f, values_f = [], []

    return render_template("analizador/ultimas48_y_forecast.html",
        equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
        labels_bt=json.dumps(labels_bt), real_bt=json.dumps(real_bt), pred_bt=json.dumps(pred_48),
        labels_f=json.dumps(labels_f), values_f=json.dumps(values_f)
    )


'''
# --- MAPA DE RED ---
@bp.route("/mapa_red", methods=["GET"])
@login_required
def mapa_red():
    """
    Replica la funcionalidad vieja: recibe equipment (y opcional equip_grp),
    filtra el SHP de la red y genera un HTML estático en /static/mapa_red.html,
    luego lo muestra embebido en un iframe.
    """
    import geopandas as gpd
    from app.services.mapa import crear_mapa_red

    equip_grp = request.args.get("equip_grp","").strip()
    equipment = request.args.get("equipment","").strip()

    if not equipment:
        flash("Falta el parámetro 'equipment' para mostrar el mapa.", "warning")
        return redirect(url_for("analizador.index"))

    # Buscar el shapefile en ubicaciones típicas (compatibilidad)
    from flask import current_app
    app_root = Path(current_app.root_path)
    candidates = [
        app_root / "utils" / "shapefile" / "red_electrica.shp",     # como en app vieja
        app_root.parent / "utils" / "shapefile" / "red_electrica.shp",
        app_root / "static" / "geodata" / "red_electrica.shp",
        Path("storage") / "shapefiles" / "red_electrica.shp",
    ]
    shp_path = None
    for p in candidates:
        if p.exists():
            shp_path = p
            break
    if shp_path is None:
        flash("No se encontró el shapefile de la red (red_electrica.shp). Copiá la carpeta 'utils/shapefile' de la app vieja a 'app/utils/shapefile' o a 'app/static/geodata'.", "danger")
        return redirect(url_for("analizador.index"))

    try:
        gdf = gpd.read_file(str(shp_path))
    except Exception as e:
        flash(f"Error leyendo shapefile: {e}", "danger")
        return redirect(url_for("analizador.index"))

    gdf['Alimentado'] = gdf['Alimentado'].astype(str).str.strip()
    eq = equipment.strip().lower()
    # filtro difuso: contiene el texto del equipo
    gdf_filtrado = gdf[gdf['Alimentado'].str.lower().str.contains(eq, na=False)].copy()

    if gdf_filtrado.empty:
        flash("No hay información geográfica para ese equipo/línea.", "warning")
        return redirect(url_for("analizador.index"))

    # columnas auxiliares para popups
    gdf_filtrado['equip_grp'] = gdf_filtrado.get('equip_grp', equip_grp)
    gdf_filtrado['equipment'] = equipment

    # Generar mapa en /static
    out_html = Path(current_app.static_folder) / "mapa_red.html"
    crear_mapa_red(gdf_filtrado, str(out_html))

    return render_template("analizador/mapa_red.html", equip_grp=equip_grp, equipment=equipment)

    '''

# --- MAPA de red (cliente, sin dependencias server) ---
@bp.route("/mapa_red", methods=["GET"])
@login_required
def mapa_red():
    equip_grp = request.args.get("equip_grp","").strip()
    equipment = request.args.get("equipment","").strip()
    if not equipment:
        flash("Falta el parámetro 'equipment' para mostrar el mapa.", "warning")
        return redirect(url_for("analizador.index"))
    # Renderiza el template que carga el ZIP desde /static/geodata/red_electrica.zip
    return render_template(
        "analizador/mapa_red_client.html",
        equip_grp=equip_grp,
        equipment=equipment,
        zip_url=url_for("static", filename="geodata/red_electrica.zip")
    )


@bp.route("/analizador/debug_forecast")
@login_required
def debug_forecast():
    from app.services.lstm import _get_forecast_df
    equip_grp = request.args.get("equip_grp","")
    equipment = request.args.get("equipment","")
    df = _get_forecast_df(equip_grp, equipment, steps=96)
    if df is None or df.empty:
        return "forecast_df: vacío", 200
    return f"""
    filas: {len(df)}<br>
    primera: {df['time'].min()}<br>
    última: {df['time'].max()}<br>
    columnas: {list(df.columns)}
    """, 200