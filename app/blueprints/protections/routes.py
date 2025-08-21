# app/blueprints/protections/routes.py (step 3)
import math
from flask import Blueprint, render_template, request, jsonify, current_app
from app.services.sql import query_list as sql_query
from datetime import datetime, timedelta
from app.services.geo_len import kml_line_length_km, shp_line_length_km, zip_shp_line_length_km
from app.blueprints.analizador.services import fetch_sql_series
from app.services.signal_adapter import series

bp = Blueprint("protections", __name__, template_folder="../../templates/protections")

IEC_CURVES = {
    "standard": {"k":0.14, "alpha":0.02},
    "very": {"k":13.5, "alpha":1.0},
    "extremely": {"k":80.0, "alpha":2.0},
    "long": {"k":120.0, "alpha":1.0},
}

def idmt_time(I, Is, TMS, curve="standard"):
    c = IEC_CURVES[curve]
    k, a = c["k"], c["alpha"]
    M = max(I/Is, 1.0000001)
    return TMS * (k / ((M**a) - 1.0))

def percentile(xs, q=0.95):
    if not xs: return None
    xs = sorted(xs)
    i = int(round((len(xs)-1)*q))
    return xs[min(max(i,0), len(xs)-1)]

def last_values_series(equip_grp, equipment, signal_id, hours=168):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=hours)
    df = fetch_sql_series(equip_grp, equipment, signal_id, start_dt, end_dt)
    if df.empty:
        return []
    ok = (df.get("OLD", 0) == 0) & (df.get("BAD", 0) == 0)
    vals = df.loc[ok, "VALUE"].dropna().astype(float).tolist()
    return vals



@bp.route("/protecciones/idmt", methods=["GET"])
def idmt_index():
    return render_template("protections/idmt.html")

@bp.route("/protecciones/idmt", methods=["POST"])
def idmt_calc():
    data = request.get_json(force=True)
    I_fault = float(data.get("I_fault", 1000.0))
    Is = float(data.get("Is", 200.0))
    TMS = float(data.get("TMS", 0.1))
    curve = data.get("curve", "standard")
    t = idmt_time(I_fault, Is, TMS, curve)
    return jsonify({"ok": True, "t_trip_s": t})

@bp.route("/protecciones/sugerencias", methods=["POST"])
def sugerencias():
    d = request.get_json(force=True)
    V_kV = float(d.get("V_kV", 13.2))
    pf = float(d.get("pf", 0.95))
    is_factor = float(d.get("is_factor", 1.2))
    curve = d.get("curve", "standard")
    t_target = float(d.get("t_target", 0.4))

    tag = d.get("tag") or {}
    scale = float(tag.get("scale", 1.0))
    vals = last_values_series(tag.get("equip_grp",""), tag.get("equipment",""), tag.get("signal_id",0), hours=int(d.get("hours",168)))
    if not vals:
        return jsonify({"ok": False, "error": "No hay datos del tag para calcular p95"}), 400
    p95 = percentile(vals, 0.95) * scale  # Supón MW si scale convierte
    I_p95 = (p95*1e6) / (math.sqrt(3.0)*V_kV*1e3*pf)

    Is = is_factor * I_p95

    line = d.get("line") or {}
    Zs = d.get("Zs") or {"r":0.2,"x":0.8}
    frac = float(d.get("fraction", 0.5))
    r_km = float(line.get("r_ohm_km", 0.3))
    x_km = float(line.get("x_ohm_km", 0.9))
    L = None
    shape = d.get("shape") or {}
    if shape.get("path") and shape.get("attr") and (shape.get("value") is not None):
        try:
            L = shp_line_length_km(shape["path"], shape["attr"], shape["value"])
        except Exception:
            L = None
    # NUEVO: zip shapefile (igual que el mapa)
    if L is None and shape.get("zip_path") and shape.get("attr") and (shape.get("value") is not None):
        try:
            zip_path = shape["zip_path"]
            if zip_path.startswith("/static/"):
                zip_path = os.path.join(current_app.root_path, zip_path.lstrip("/"))
            L = zip_shp_line_length_km(zip_path, shape["attr"], shape["value"])
        except Exception:
            L = None
            
    if L is None:
        line = d.get("line") or {}
        if line.get("length_km") is not None:
            L = float(line["length_km"])
        elif line.get("kml_file") and line.get("name"):
            try:
                L = kml_line_length_km(line["kml_file"], line["name"])
            except Exception:
                L = None
    Zline = complex(r_km, x_km) * (L*frac)
    Ztot = complex(Zs.get("r",0.2), Zs.get("x",0.8)) + Zline
    V_phase = (V_kV*1e3)/math.sqrt(3.0)
    I_f = V_phase / abs(Ztot)

    c = IEC_CURVES[curve]; k, a = c["k"], c["alpha"]
    M = max(I_f/Is, 1.0000001)
    TMS = t_target * ((M**a) - 1.0) / k

    return jsonify({"ok": True, "Is_A": Is, "TMS": TMS, "I_f_A": I_f, "p95_power": p95})


prot_bp = Blueprint("protections", __name__, template_folder="../../templates/protections")

@bp.route("/protections/propose_5051", methods=["POST"])
def propose_5051():
    d = request.get_json(force=True)
    cur = d.get("current_tag") or {}
    days = int(d.get("days", 30))

    end = datetime.now()
    start = end - timedelta(days=days)
    df_i = series(cur.get("equip_grp"), cur.get("equipment"), cur.get("signal_id"), start, end)
    if df_i is None or df_i.empty:
        return jsonify({"ok": False, "error": "Sin datos de corriente"}), 400

    ok = (df_i.get("OLD", 0)==0) & (df_i.get("BAD",0)==0)
    df_i = df_i.loc[ok]
    Imax = float(df_i["VALUE"].max())
    Imean = float(df_i["VALUE"].mean())

    I51_pickup = round(1.25 * max(Imean, 0.8*Imax), 2)
    I50_pickup = round(6.0 * I51_pickup, 1)

    L = None
    shape = d.get("shape") or {}
    if shape.get("zip_path") and shape.get("attr") and (shape.get("value") is not None):
        try: L = zip_shp_line_length_km(shape["zip_path"], shape["attr"], shape["value"])
        except Exception: L = None
    elif d.get("length_km") is not None:
        L = float(d["length_km"])

    return jsonify({
        "ok": True,
        "inputs": {"Imax": Imax, "Imean": Imean, "length_km": L},
        "suggestions": {
            "51P": {"pickup_A": I51_pickup, "curve": "IEC SI", "tms": 0.2},
            "50P": {"pickup_A": I50_pickup}
        }
    })
