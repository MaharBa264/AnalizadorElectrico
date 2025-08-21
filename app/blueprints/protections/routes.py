# app/blueprints/protections/routes.py
import os
import math
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, current_app

from app.services.signal_adapter import series
from app.services.geo_len import (
    kml_line_length_km,
    shp_line_length_km,
    zip_shp_line_length_km,
)

DEFAULT_GEODATA_ZIP = os.getenv(
    "GEODATA_ZIP",
    os.path.join("app", "static", "geodata", "red_electrica.zip")
)

prot_bp = Blueprint("protections", __name__, template_folder="../../templates/protections")

# ---- Curvas IEC (IDMT) ----
IEC_CURVES = {
    "standard":   {"k": 0.14,  "alpha": 0.02},  # Standard inverse (SI)
    "very":       {"k": 13.5,  "alpha": 1.0},   # Very inverse (VI)
    "extremely":  {"k": 80.0,  "alpha": 2.0},   # Extremely inverse (EI)
    "long":       {"k": 120.0, "alpha": 1.0},   # Long-time inverse (LTI)
}

def _curve_key(curve: str) -> str:
    return curve if curve in IEC_CURVES else "standard"

def idmt_time(I: float, Is: float, TMS: float, curve: str = "standard") -> float:
    c = IEC_CURVES[_curve_key(curve)]
    k, a = c["k"], c["alpha"]
    M = max(I / max(Is, 1e-9), 1.0000001)
    return TMS * (k / ((M ** a) - 1.0))

def percentile(xs, q=0.95):
    if not xs:
        return None
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    return xs[min(max(i, 0), len(xs) - 1)]

def _values_from_series(equip_grp, equipment, signal_id, start_dt, end_dt):
    """Valores (float) filtrando OLD/BAD, usando el mismo método que /analizador."""
    df = series(equip_grp, equipment, signal_id, start_dt, end_dt)
    if df is None or df.empty:
        return []
    ok = (df.get("OLD", 0) == 0) & (df.get("BAD", 0) == 0)
    vals = df.loc[ok, "VALUE"].dropna().astype(float).tolist()
    return vals

# ---- Vistas ----

@prot_bp.route("/protections")
@prot_bp.route("/protecciones")
def index():
    return render_template("protections/index.html")

@prot_bp.route("/protections/idmt", methods=["GET"])
@prot_bp.route("/protecciones/idmt", methods=["GET"])
def idmt_index():
    return render_template("protections/idmt.html")

@prot_bp.route("/protections/idmt", methods=["POST"])
@prot_bp.route("/protecciones/idmt", methods=["POST"])
def idmt_calc():
    data = request.get_json(force=True)
    I_fault = float(data.get("I_fault", 1000.0))
    Is = float(data.get("Is", 200.0))
    TMS = float(data.get("TMS", 0.1))
    curve = _curve_key(data.get("curve", "standard"))
    t = idmt_time(I_fault, Is, TMS, curve)
    return jsonify({"ok": True, "t_trip_s": t, "curve": curve})

@prot_bp.route("/protections/suggest", methods=["POST"])
@prot_bp.route("/protecciones/sugerencias", methods=["POST"])
def suggest():
    """
    Sugerencia de Is y TMS para 51P:
    - Is: factor * I_p95 (de potencia p95 -> I = P/(√3·V·pf))
    - TMS: tal que la curva dispare en t_target ante I_f calculada (Zs + Zlínea*fraction)
    """
    d = request.get_json(force=True)

    V_kV = float(d.get("V_kV", 13.2))
    pf = float(d.get("pf", 0.95))
    is_factor = float(d.get("is_factor", 1.2))
    curve = _curve_key(d.get("curve", "standard"))
    t_target = float(d.get("t_target", 0.4))

    # --- dato de potencia (para estimar I carga) ---
    tag = d.get("tag") or {}
    scale = float(tag.get("scale", 1.0))
    hours = int(d.get("hours", 168))
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=hours)
    vals = _values_from_series(tag.get("equip_grp", ""), tag.get("equipment", ""), tag.get("signal_id", ""), start_dt, end_dt)
    if not vals:
        return jsonify({"ok": False, "error": "No hay datos del tag para calcular p95"}), 400

    p95 = percentile(vals, 0.95) * scale  # suponemos que scale lleva a MW si corresponde
    I_p95 = (p95 * 1e6) / (math.sqrt(3.0) * V_kV * 1e3 * max(pf, 1e-6))

    Is = is_factor * I_p95

    # --- falta trifásica: Zs + Zlínea*fraction ---
    line = d.get("line") or {}
    Zs = d.get("Zs") or {"r": 0.2, "x": 0.8}
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
    if L is None and shape.get("zip_path") and shape.get("attr") and (shape.get("value") is not None):
        try:
            zip_path = shape["zip_path"]
            if zip_path.startswith("/static/"):
                zip_path = os.path.join(current_app.root_path, zip_path.lstrip("/"))
            L = zip_shp_line_length_km(zip_path, shape["attr"], shape["value"])
        except Exception:
            L = None
    if L is None:
        if line.get("length_km") is not None:
            L = float(line["length_km"])
        elif line.get("kml_file") and line.get("name"):
            try:
                L = kml_line_length_km(line["kml_file"], line["name"])
            except Exception:
                L = None

    Zline = complex(r_km, x_km) * (float(L or 0.0) * frac)
    Ztot = complex(Zs.get("r", 0.2), Zs.get("x", 0.8)) + Zline
    V_phase = (V_kV * 1e3) / math.sqrt(3.0)
    I_f = V_phase / max(abs(Ztot), 1e-9)

    # TMS para t_target en I_f
    c = IEC_CURVES[curve]
    k, a = c["k"], c["alpha"]
    M = max(I_f / max(Is, 1e-9), 1.0000001)
    TMS = t_target * ((M ** a) - 1.0) / k

    return jsonify({
        "ok": True,
        "Is_A": Is,
        "TMS": TMS,
        "I_f_A": I_f,
        "p95_power": p95,
        "curve": curve,
        "length_km": L
    })

@prot_bp.route("/protections/propose_5051", methods=["POST"])
def propose_5051():
    """
    Propuesta inicial de 51P/50P desde histórico de corriente:
    - 51P pickup = 1.25 × max(mean, 0.8×Imax)
    - 50P pickup = 6 × 51P (placeholder; afinar con Icc)
    """
    d = request.get_json(force=True)
    cur = d.get("current_tag") or {}
    days = int(d.get("days", 30))

    end = datetime.now()
    start = end - timedelta(days=days)
    df_i = series(cur.get("equip_grp"), cur.get("equipment"), cur.get("signal_id"), start, end)
    if df_i is None or df_i.empty:
        return jsonify({"ok": False, "error": "Sin datos de corriente"}), 400

    ok = (df_i.get("OLD", 0) == 0) & (df_i.get("BAD", 0) == 0)
    df_i = df_i.loc[ok]
    if df_i.empty:
        return jsonify({"ok": False, "error": "Corriente solo con OLD/BAD=1"}), 400

    Imax = float(df_i["VALUE"].max())
    Imean = float(df_i["VALUE"].mean())

    I51_pickup = round(1.25 * max(Imean, 0.8 * Imax), 2)
    I50_pickup = round(6.0 * I51_pickup, 1)

    L = None
    shape = d.get("shape") or {}
    if shape.get("zip_path") and shape.get("attr") and (shape.get("value") is not None):
        try:
            L = zip_shp_line_length_km(shape["zip_path"], shape["attr"], shape["value"])
        except Exception:
            L = None
    elif d.get("length_km") is not None:
        L = float(d["length_km"])

    return jsonify({
        "ok": True,
        "inputs": {"Imax": Imax, "Imean": Imean, "length_km": L},
        "suggestions": {
            "51P": {"pickup_A": I51_pickup, "curve": "standard", "tms": 0.2},
            "50P": {"pickup_A": I50_pickup}
        }
    })

@prot_bp.route("/protections/wizard5051", methods=["GET"])
def wizard5051():
    return render_template("protections/wizard_5051.html", default_zip=DEFAULT_GEODATA_ZIP)


@prot_bp.route("/protections/zip_line_ids")
def zip_line_ids():
    from app.services.geo_len import zip_shp_unique_values
    zip_path = _resolve_zip_path(request.args.get("zip_path"))
    attr_in = (request.args.get("attr") or "").strip() or None
    try:
        ids, used_attr = zip_shp_unique_values(zip_path, attr_in)
        return jsonify({"ok": True, "ids": ids, "attr": used_attr})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@prot_bp.route("/protections/zip_line_profile", methods=["POST"])
def zip_line_profile():
    from app.services.geo_len import zip_shp_line_profile
    d = request.get_json(force=True)
    zip_path = _resolve_zip_path(d.get("zip_path"))
    attr_in   = (d.get("attr") or "").strip() or None
    value     = d.get("value")
    material  = (d.get("material_attr") or "").strip() or None
    if value is None:
        return jsonify({"ok": False, "error": "value es requerido"}), 400
    try:
        prof = zip_shp_line_profile(zip_path, attr_in, value, material)
        return jsonify({"ok": True, **prof})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
def _resolve_zip_path(p):
    from flask import current_app
    if not p:
        p = DEFAULT_GEODATA_ZIP
    ap = p
    if p.startswith("/static/"):
        ap = os.path.join(current_app.root_path, p.lstrip("/"))
    if not os.path.isabs(ap):
        ap2 = os.path.join(current_app.root_path, ap)
        ap = ap2 if os.path.exists(ap2) else os.path.join(os.getcwd(), ap)
    if not os.path.exists(ap):
        ap = DEFAULT_GEODATA_ZIP
    return ap