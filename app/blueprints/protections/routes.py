# app/blueprints/protections/routes.py (step 3)
from flask import Blueprint, render_template, request, jsonify
import math
from app.services.sql import query_list as sql_query
from app.services.geo_len import kml_line_length_km, shp_line_length_km

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
    sql = f"""
    SELECT Time, VALUE
    FROM ANALOG_NUEVA
    WHERE EQUIP_GRP = ? AND EQUIPMENT = ? AND ID = ?
      AND BAD = 0 AND OLD = 0
      AND Time >= DATEADD(hour, -{hours}, GETDATE())
    ORDER BY Time ASC
    """
    rows = sql_query(sql, [equip_grp, equipment, signal_id]) or []
    vals = []
    for r in rows:
        v = r.get("VALUE") or r.get("value")
        try:
            vals.append(float(v))
        except:
            pass
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

    