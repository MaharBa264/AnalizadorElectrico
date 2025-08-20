# app/blueprints/power/routes.py (step 3 - with tag tools)
from flask import Blueprint, render_template, request, jsonify
import os, requests, math, json
from app.services.sql import query_list as sql_query
from app.services.geo_len import kml_line_length_km

bp = Blueprint("power", __name__, template_folder="../../templates/power")

PYPSA_SVC_URL = os.getenv("PYPSA_SVC_URL", "http://127.0.0.1:8010")
MAP_DIR = os.path.join(os.getcwd(), "storage", "pf_maps")
os.makedirs(MAP_DIR, exist_ok=True)

def last_value(equip_grp, equipment, signal_id):
    sql = """
    SELECT TOP 1 VALUE
    FROM ANALOG_NUEVA
    WHERE EQUIP_GRP = ? AND EQUIPMENT = ? AND ID = ?
      AND BAD = 0 AND OLD = 0
    ORDER BY Time DESC
    """
    rows = sql_query(sql, [equip_grp, equipment, signal_id]) or []
    if not rows:
        return None
    v = rows[0].get("VALUE") or rows[0].get("value")
    try:
        return float(v)
    except Exception:
        return None

def resolve_sources(cfg):
    cfg2 = {"buses":[], "lines":[], "transformers":[], "loads":[], "generators":[]}
    cfg2["buses"] = cfg.get("buses", [])
    cfg2["lines"] = cfg.get("lines", [])
    cfg2["transformers"] = cfg.get("transformers", [])

    for ld in cfg.get("loads", []):
        out = {"name": ld["name"], "bus": ld["bus"], "p_mw": ld.get("p_mw", 0.0), "q_mvar": ld.get("q_mvar", 0.0)}
        src = ld.get("sources", {})
        for k in ("p_mw","q_mvar"):
            if k in src:
                s = src[k]; val = last_value(s["equip_grp"], s["equipment"], s["signal_id"])
                if val is not None: out[k] = val * float(s.get("scale", 1.0))
        cfg2["loads"].append(out)

    for g in cfg.get("generators", []):
        out = {"name": g["name"], "bus": g["bus"],
               "p_set_mw": g.get("p_set_mw", 0.0),
               "v_set_pu": g.get("v_set_pu", 1.0),
               "p_max_mw": g.get("p_max_mw", 1.0),
               "control": g.get("control", "PV")}
        src = g.get("sources", {})
        for k in ("p_set_mw","v_set_pu"):
            if k in src:
                s = src[k]; val = last_value(s["equip_grp"], s["equipment"], s["signal_id"])
                if val is not None: out[k] = val * float(s.get("scale", 1.0))
        cfg2["generators"].append(out)
    return cfg2

@bp.route("/power/unifilar", methods=["GET"])
def unifilar_index():
    return render_template("power/unifilar.html")

@bp.route("/power/unifilar", methods=["POST"])
def unifilar_build():
    cfg = request.get_json(force=True)
    cfg_num = resolve_sources(cfg)  # solo lectura
    try:
        r = requests.post(f"{PYPSA_SVC_URL}/pf", json=cfg_num, timeout=30)
        data = r.json()
        if not data.get("ok"):
            return jsonify({"ok": False, "error": data.get("error","falló microservicio"), "results": data.get("results")}), 500

        # Grafo con % de carga y dirección
        nodes = [{"data":{"id": b["name"], "label": b["name"]}} for b in cfg_num.get("buses", [])]
        edges = []
        line_p = (data.get("results") or {}).get("line_p_mw", {})
        line_defs = {ln["name"]: ln for ln in cfg_num.get("lines", [])}
        for name, ln in line_defs.items():
            p = float(line_p.get(name, 0.0) or 0.0)
            s_nom = float(ln.get("s_nom", 0.0) or 0.0)
            loading = abs(p)/s_nom*100.0 if s_nom>0 else None
            severity = "ok"
            if loading is not None:
                if loading > 90: severity = "crit"
                elif loading > 60: severity = "warn"
            fwd = p >= 0
            source = ln["bus0"] if fwd else ln["bus1"]
            target = ln["bus1"] if fwd else ln["bus0"]
            edges.append({"data":{
                "id": name, "source": source, "target": target,
                "type":"line",
                "p_mw": p, "s_nom": s_nom, "loading": loading, "severity": severity
            }})

        for tr in cfg_num.get("transformers", []):
            edges.append({"data":{"id": tr["name"], "source": tr["bus0"], "target": tr["bus1"], "type":"trafo"}})

        out = {
            "graph":{"nodes":nodes, "edges":edges},
            "results": data["results"],
            "resolved": cfg_num
        }
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": f"No pude contactar al microservicio: {e}"}), 500

# ---- Herramientas de tags (solo lectura) ----
@bp.route("/power/tag_search", methods=["GET"])
def tag_search():
    grp = request.args.get("grp", "%")
    equip = request.args.get("equipment", "%")
    days = int(request.args.get("days","7"))
    sql = f"""
    SELECT DISTINCT TOP 100 EQUIP_GRP, EQUIPMENT, ID
    FROM ANALOG_NUEVA
    WHERE Time >= DATEADD(day, -{days}, GETDATE())
      AND EQUIP_GRP LIKE ? AND EQUIPMENT LIKE ?
    ORDER BY EQUIP_GRP, EQUIPMENT, ID
    """
    rows = sql_query(sql, [grp, equip]) or []
    return jsonify({"ok": True, "items": rows})

@bp.route("/power/probe_tag", methods=["POST"])
def probe_tag():
    data = request.get_json(force=True)
    v = last_value(data["equip_grp"], data["equipment"], data["signal_id"])
    if v is None:
        return jsonify({"ok": False, "error": "sin datos"}), 404
    scale = float(data.get("scale", 1.0))
    return jsonify({"ok": True, "value": v*scale})

@bp.route("/power/map_save", methods=["POST"])
def map_save():
    payload = request.get_json(force=True)
    name = payload.get("name","default").replace("/","_").replace("\\","_")
    data = payload.get("map")
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "map inválido"}), 400
    path = os.path.join(MAP_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "path": path})

@bp.route("/power/map_load", methods=["GET"])
def map_load():
    name = request.args.get("name","default").replace("/","_").replace("\\","_")
    path = os.path.join(MAP_DIR, f"{name}.json")
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "no existe"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"ok": True, "map": data})

# --- Utilidad: Ifalla trifásica (con KML opcional) ---
@bp.route("/power/fault_current", methods=["POST"])
def fault_current():
    data = request.get_json(force=True)
    V_kV = float(data.get("V_kV", 33.0))
    Zs = data.get("Zs") or {"r":0.2,"x":0.8}
    line = data.get("line") or {}
    frac = float(data.get("fraction", 0.5))
    r_km = float(line.get("r_ohm_km", 0.3))
    x_km = float(line.get("x_ohm_km", 0.9))
    L = line.get("length_km", None)
    if L is None and line.get("kml_file") and line.get("name"):
        try:
            L = kml_line_length_km(line["kml_file"], line["name"])
        except Exception:
            L = None
    if L is None:
        return jsonify({"ok": False, "error": "No puedo determinar la longitud: envía length_km o kml_file+name"}), 400

    Zline = complex(r_km, x_km) * (L*frac)
    Ztot = complex(Zs.get("r",0.2), Zs.get("x",0.8)) + Zline
    V_phase = (V_kV*1e3)/math.sqrt(3.0)
    I_fault = V_phase / abs(Ztot)
    return jsonify({"ok": True, "length_km": L, "I_fault_A": I_fault})

# --- Tag combo endpoints (solo lectura) ---
@bp.route("/power/tag_groups", methods=["GET"])
def tag_groups():
    days = int(request.args.get("days","7"))
    sql = f"""
    SELECT DISTINCT EQUIP_GRP
    FROM ANALOG_NUEVA
    WHERE Time >= DATEADD(day, -{days}, GETDATE())
    ORDER BY EQUIP_GRP
    """
    rows = sql_query(sql, []) or []
    return jsonify({"ok": True, "groups": [r["EQUIP_GRP"] for r in rows if r.get("EQUIP_GRP") is not None]})

@bp.route("/power/tag_equipment", methods=["GET"])
def tag_equipment():
    grp = request.args.get("grp","%")
    days = int(request.args.get("days","7"))
    sql = f"""
    SELECT DISTINCT EQUIPMENT
    FROM ANALOG_NUEVA
    WHERE Time >= DATEADD(day, -{days}, GETDATE()) AND EQUIP_GRP = ?
    ORDER BY EQUIPMENT
    """
    rows = sql_query(sql, [grp]) or []
    return jsonify({"ok": True, "equipment": [r["EQUIPMENT"] for r in rows if r.get("EQUIPMENT") is not None]})

@bp.route("/power/tag_ids", methods=["GET"])
def tag_ids():
    grp = request.args.get("grp","%")
    equipment = request.args.get("equipment","%")
    days = int(request.args.get("days","7"))
    sql = f"""
    SELECT DISTINCT ID
    FROM ANALOG_NUEVA
    WHERE Time >= DATEADD(day, -{days}, GETDATE()) AND EQUIP_GRP = ? AND EQUIPMENT = ?
    ORDER BY ID
    """
    rows = sql_query(sql, [grp, equipment]) or []
    ids = [
        int(r["ID"]) if isinstance(r.get("ID"), (int,)) or (isinstance(r.get("ID"), str) and r["ID"].isdigit())
        else r["ID"]
        for r in rows if r.get("ID") is not None
    ]
    return jsonify({"ok": True, "ids": ids})

