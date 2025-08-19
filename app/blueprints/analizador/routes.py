from datetime import datetime, timedelta
import json
import pandas as pd
from io import BytesIO
from flask import render_template, request, redirect, url_for, flash, send_file
from flask_login import login_required
from . import bp
from .services import get_signal_hierarchy, get_coords_para_equipo, fetch_sql_series, fetch_weather_history, make_trend_payload
from .compare import comparar_corriente_vs_temp, graficar_corriente_vs_temp

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
            time_data=time_data
        )

    # 2) Comparación con clima (WeatherAPI histórico + forecast corto)
    lat, lon = get_coords_para_equipo(equip_grp, equipment)
    station_info = None
    if lat is None or lon is None:
        flash("No hay coordenadas asociadas al equipo para obtener temperatura.", "warning")
        return render_template(
            "analizador/resultados_sql.html",
            datos=datos,
            min_value=min_value, max_value=max_value,
            min_time=min_time, max_time=max_time,
            avg_value=avg_value, total_muestras=total_muestras,
            equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
            time_data=time_data
        )
    else:
        station_info = {"latitud": lat, "longitud": lon}

    dfw = fetch_weather_history(lat, lon, fecha_ini, fecha_fin)

    # Merge por hora (promedio horario de VALUE)
    sql_hour = (df_sql
                .dropna(subset=["VALUE"])
                .assign(TIME_H=df_sql["TIME_FULL"].dt.floor("H"))
                .groupby("TIME_H", as_index=False).agg(VALUE=("VALUE","mean"))
               )
    merged = sql_hour.merge(dfw, left_on="TIME_H", right_on="time", how="left")
    merged.rename(columns={"TIME_H":"time"}, inplace=True)

    # Tabla HTML para detalles en la vista "results.html"
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
    valid = merged.dropna(subset=["VALUE","temperature"])
    if not valid.empty:
        try:
            correlation = float(valid["VALUE"].corr(valid["temperature"]))
        except Exception:
            correlation = None
        try:
            import numpy as np
            slope, intercept = np.polyfit(valid["temperature"], valid["VALUE"], 1)
        except Exception:
            slope = intercept = None

    # Forecast próximo 3 días (si hay regresión)
    forecast_list = []
    if slope is not None and intercept is not None:
        try:
            fc = forecast_hourly(lat, lon, days=3)  # WeatherAPI forecast
            hours = (fc.get("forecast",{}).get("forecastday",[]))
            # Usamos extremos por día (min/max de temp) para proyectar corriente
            for day in hours:
                date = day.get("date")
                try:
                    tmax = day.get("day",{}).get("maxtemp_c")
                    tmin = day.get("day",{}).get("mintemp_c")
                except Exception:
                    tmax = tmin = None
                forecast_list.append({
                    "date": date,
                    "forecast_max": tmax,
                    "predicted_current_max": (slope*tmax + intercept) if tmax is not None else None,
                    "forecast_min": tmin,
                    "predicted_current_min": (slope*tmin + intercept) if tmin is not None else None
                })
        except Exception:
            forecast_list = []

    # Helpers exactamente como en tu app vieja
    tabla_comparacion = comparar_corriente_vs_temp(
        merged.rename(columns={"time":"TIME_FULL"})  # esperan TIME_FULL
    )
    grafico_base64 = graficar_corriente_vs_temp(
        merged.rename(columns={"time":"TIME_FULL"})
    )

    return render_template(
        "analizador/results.html",
        station_label=f"{equip_grp} / {equipment} / {signal_id}",
        table=table_html,
        time_data=time_data,
        correlation=correlation,
        forecast_list=forecast_list,
        station_info=station_info,
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
        fmt = "%Y-%m-%dT%H:%M" if "T" in s else ("%Y-%m-%d %H:%M" if " " in s else "%Y-%m-%d")
        return datetime.strptime(s, fmt)

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
