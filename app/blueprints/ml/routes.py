from flask import render_template, request, flash, redirect, url_for
from . import bp
from datetime import datetime, timedelta
from app.utils.timeutils import LOCAL_TZ, as_local
from .pipeline.data import fetch_signal_series, clean_bad_old, resample_hourly, merge_with_weather

def _dt_parse(s, default_days=7):
    if not s:
        return (datetime.now(LOCAL_TZ) - timedelta(days=default_days)).replace(minute=0, second=0, microsecond=0)
    try:
        # formatos simples: YYYY-MM-DD o YYYY-MM-DD HH:MM
        if len(s.strip()) <= 10:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=LOCAL_TZ)
        return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)
    except Exception:
        return (datetime.now(LOCAL_TZ) - timedelta(days=default_days)).replace(minute=0, second=0, microsecond=0)

@bp.route("/")
def index():
    # Valores por defecto para el form
    today = datetime.now(LOCAL_TZ).replace(minute=0, second=0, microsecond=0)
    seven = (today - timedelta(days=7))
    return render_template("ml/index.html",
                           def_equip_grp="ETSL",
                           def_equipment="LINEA01",
                           def_signal_id="10094",
                           def_start=seven.strftime("%Y-%m-%d"),
                           def_end=today.strftime("%Y-%m-%d"))

@bp.route("/data-preview", methods=["GET"])
def data_preview():
    equip_grp = request.args.get("equip_grp","").strip()
    equipment = request.args.get("equipment","").strip()
    signal_id = request.args.get("signal_id","").strip()
    t_start = _dt_parse(request.args.get("start"))
    t_end   = _dt_parse(request.args.get("end"), default_days=0)

    if not (equip_grp and equipment and signal_id):
        flash("Completá equip_grp, equipment y signal_id.", "warning")
        return redirect(url_for("ml.index"))

    try:
        raw = fetch_signal_series(equip_grp, equipment, signal_id, t_start, t_end)
        cleaned = clean_bad_old(raw)
        hourly = resample_hourly(cleaned, agg="mean")

        # Merge opcional con clima desde Influx (si tenés la lib instalada y bucket poblado)
        weather = None
        try:
            from app.services.influx import query_weather_hourly
            weather = query_weather_hourly(equip_grp, t_start, t_end)
        except Exception as e:
            weather = None  # si no hay Influx/lib, seguimos sin clima

        merged = merge_with_weather(hourly, weather)

        # Stats
        stats = {
            "raw_count": len(raw),
            "cleaned_count": len(cleaned),
            "hourly_count": len(hourly),
            "merged_count": len(merged),
            "range": f"{t_start} → {t_end}",
        }

        # preview primeras 50 filas
        preview = merged[:50]
        return render_template("ml/data_preview.html",
                               equip_grp=equip_grp, equipment=equipment, signal_id=signal_id,
                               stats=stats, rows=preview)
    except Exception as e:
        flash(f"Error obteniendo datos: {e}", "danger")
        return redirect(url_for("ml.index"))

@bp.route("/train", methods=["POST"])
def train():
    # Stub: dejamos el gancho y mensajes claros (TF suele no estar en Py3.6)
    flash("Entrenamiento real pendiente: datos ya consultados correctamente. Faltaría TensorFlow/Keras o actualizar a Py>=3.9.", "info")
    return redirect(url_for("ml.index"))
