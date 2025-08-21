# app/blueprints/power/routes.py  (corregido)
import os, requests, math, json
from app.services.sql import query_list as sql_query
from app.services.geo_len import kml_line_length_km
from flask import Blueprint, render_template, request, jsonify, abort, redirect, url_for, current_app
from werkzeug.urls import url_encode
from app.blueprints.analizador.services import get_signal_hierarchy
from datetime import datetime, timedelta
from app.services.signal_adapter import series, list_hierarchy, snapshot


ANALISIS_BP = os.getenv("ANALISIS_BP", "analisis")  # nombre del blueprint de /analisis
bp = Blueprint("power", __name__, template_folder="../../templates/power")

PYPSA_SVC_URL = os.getenv("PYPSA_SVC_URL", "http://127.0.0.1:8010")
MAP_DIR = os.path.join(os.getcwd(), "storage", "pf_maps")
os.makedirs(MAP_DIR, exist_ok=True)
# --- Detección robusta de tabla/columna tiempo para SCADA (solo lectura) ---
ANALOG_TABLE_ENV = os.getenv("ANALOG_TABLE", "").strip() or None

def _parse_ts(ts_str):
    """
    Soporta valores de <input type='datetime-local'> en Python 3.6:
    'YYYY-MM-DDTHH:MM' o 'YYYY-MM-DDTHH:MM:SS'.
    Devuelve datetime o None si no puede parsear.
    """
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    return None


def _parse_table_ident(ident: str):
    """
    Acepta 'Tabla', 'schema.Tabla', '[db].[schema].[Tabla]' y devuelve (schema, table).
    Si no hay schema, retorna (None, 'Tabla').
    """
    if not ident:
        return (None, None)
    s = ident.replace('[','').replace(']','')
    parts = s.split('.')
    if len(parts) == 1:
        return (None, parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    # db.schema.table -> tomamos los dos últimos
    return (parts[-2], parts[-1])

def _detect_table():
    """
    Devuelve el nombre de la tabla de analógicas.
    Prioridad: env ANALOG_TABLE -> ANALOG_NUEVA -> ANALOG
    """
    candidates = [ANALOG_TABLE_ENV, "ANALOG_NUEVA", "ANALOG"]
    for name in candidates:
        if not name:
            continue
        try:
            sql_query(f"SELECT TOP 1 1 AS X FROM {name}", [])
            return name
        except Exception:
            continue
    # si nada existe, devolvemos algo explícito para que el error sea claro
    return None  # <- hará 500 con mensaje "No se pudo detectar la tabla"

def _detect_time_col(table_name):
    if not table_name:
        return None
    schema, tname = _parse_table_ident(table_name)
    if not tname:
        return None

    params = [tname]
    extra = ""
    if schema:
        extra = " AND TABLE_SCHEMA = ?"
        params.append(schema)

    rows = sql_query(f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?{extra}
    """, params) or []

    names = {r["COLUMN_NAME"] for r in rows}
    for cand in ("TIME_FULL", "Time", "TIME", "TIMESTAMP", "DATE_TIME", "DATETIME", "FECHA_HORA", "FECHA"):
        if cand in names:
            return cand
    return None



_COL_CACHE = {}

def _detect_columns(table_name):
    """
    Devuelve nombres reales de columnas clave en 'table_name'.
    Claves lógicas: grp, equipment, id, value, bad, old.
    """
    if not table_name:
        return {}
    if table_name in _COL_CACHE:
        return _COL_CACHE[table_name]

    schema, tname = _parse_table_ident(table_name)
    if not tname:
        return {}

    params = [tname]
    extra = ""
    if schema:
        extra = " AND TABLE_SCHEMA = ?"
        params.append(schema)

    rows = sql_query(f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?{extra}
    """, params) or []
    names_up = {r["COLUMN_NAME"].upper(): r["COLUMN_NAME"] for r in rows}

    def pick(cands):
        for c in cands:
            uc = c.upper()
            if uc in names_up:
                return names_up[uc]
        return None

    meta = {
        "grp":       pick(["EQUIP_GRP","EQUIPGRP","TAG_GROUP","SITE_TAG","EQUIP_GROUP","GRUPO","GRP"]),
        "equipment": pick(["EQUIPMENT","EQUIP","EQUIPO"]),
        "id":        pick(["ID","SIGNAL_ID","TAG_ID","POINT_ID"]),
        "value":     pick(["VALUE","VALOR","VAL"]),
        "bad":       pick(["BAD","IS_BAD","Q_BAD","QUALITY_BAD"]),
        "old":       pick(["OLD","IS_OLD","Q_OLD"]),
    }
    _COL_CACHE[table_name] = meta
    return meta



ANALOG_KEY_TABLE_ENV = os.getenv("ANALOG_KEY_TABLE", "").strip() or None

def _detect_tag_table():
    """
    Detecta la tabla 'key' que expone grp/equipment/id.
    Prioridad: env ANALOG_KEY_TABLE -> ANALOG_NUEVA_KEY -> ANALOG_KEY -> V_ANALOG_KEY
    Solo acepta tablas donde se detecten grp/equipment/id vía INFORMATION_SCHEMA.
    """
    candidates = [ANALOG_KEY_TABLE_ENV, "ANALOG_NUEVA_KEY", "ANALOG_KEY", "V_ANALOG_KEY", "dbo.ANALOG_KEY"]
    last_err = None
    for name in candidates:
        if not name:
            continue
        try:
            # Verificamos que existe
            sql_query(f"SELECT TOP 1 1 AS X FROM {name}", [])
            # Verificamos columnas
            meta = _detect_columns(name)
            if meta.get("grp") and meta.get("equipment") and meta.get("id"):
                return name
        except Exception as e:
            last_err = e
            continue
    # No encontrada: devolvemos None para que el endpoint avise
    return None



def _q(name):
    """Pone corchetes para SQL Server si hay nombre de columna."""
    return f"[{name}]" if name else None




# --- Config tabla y detección de columna de tiempo ---
TABLE_NAME = os.getenv("ANALOG_TABLE", "ANALOG_NUEVA")

_TIME_COL_CACHE = {"expr": None, "name": None}
def _time_col_expr():
    """
    Devuelve el nombre de la columna de tiempo entre corchetes para SQL Server (ej. [Time], [TIME_FULL]).
    Se detecta una sola vez con INFORMATION_SCHEMA.
    """
    if _TIME_COL_CACHE["expr"]:
        return _TIME_COL_CACHE["expr"]

    rows = sql_query("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?
    """, [TABLE_NAME]) or []

    names = [r["COLUMN_NAME"] for r in rows]
    # candidatos por orden de preferencia (case-insensitive)
    candidates = {"TIME_FULL","TIME","TIMESTAMP","DATE_TIME","DATETIME","FECHA_HORA","FECHA"}
    chosen = None
    for n in names:
        if n.upper() in candidates:
            chosen = n
            break
    # fallback razonable
    if not chosen:
        chosen = "Time"  # muy común en SCADA

    _TIME_COL_CACHE["name"] = chosen
    _TIME_COL_CACHE["expr"] = f"[{chosen}]"
    return _TIME_COL_CACHE["expr"]


def last_value(equip_grp, equipment, signal_id):
    end = datetime.now()
    start = end - timedelta(hours=6)
    df = series(equip_grp, equipment, signal_id, start, end)
    if df is None or df.empty:
        return None
    ok = (df.get("OLD",0)==0) & (df.get("BAD",0)==0)
    df = df.loc[ok]
    if df.empty:
        return None
    # ordenar por tiempo si está la columna
    for c in ("TIME_FULL","Time","TIME","timestamp"):
        if c in df.columns:
            df = df.sort_values(c)
            break
    try:
        return float(df["VALUE"].iloc[-1])
    except Exception:
        return None


def resolve_sources(cfg, snapshot_dt=None):
    cfg2 = {"buses":[], "lines":[], "transformers":[], "loads":[], "generators":[]}
    cfg2["buses"] = cfg.get("buses", [])
    cfg2["lines"] = cfg.get("lines", [])
    cfg2["transformers"] = cfg.get("transformers", [])

    def maybe_from_tag(s):
        if not s: return None
        try:
            if snapshot_dt is not None:
                v = snapshot(s["equip_grp"], s["equipment"], s["signal_id"], snapshot_dt, tolerance_minutes=10)
            else:
                v = None  # si no mandan timestamp, podés mantener last_value si querés
        except Exception:
            v = None
        return None if v is None else v * float(s.get("scale", 1.0))

    for ld in cfg.get("loads", []):
        out = {"name": ld["name"], "bus": ld["bus"], "p_mw": ld.get("p_mw", 0.0), "q_mvar": ld.get("q_mvar", 0.0)}
        for k in ("p_mw","q_mvar"):
            mv = maybe_from_tag((ld.get("sources") or {}).get(k))
            if mv is not None: out[k] = mv
        cfg2["loads"].append(out)

    for g in cfg.get("generators", []):
        out = {"name": g["name"], "bus": g["bus"], "p_set_mw": g.get("p_set_mw", 0.0),
               "v_set_pu": g.get("v_set_pu", 1.0), "p_max_mw": g.get("p_max_mw", 1.0),
               "control": g.get("control", "PV")}
        for k in ("p_set_mw","v_set_pu"):
            mv = maybe_from_tag((g.get("sources") or {}).get(k))
            if mv is not None: out[k] = mv
        cfg2["generators"].append(out)
    return cfg2

# ---------------------------------------------------------------------------

@bp.route("/power/unifilar", methods=["GET"])
def unifilar_index():
    return render_template("power/unifilar.html")

@bp.route("/power/unifilar", methods=["POST"])
def unifilar_build():
    cfg = request.get_json(force=True)
    ts_iso = cfg.get("timestamp")  # ej. "2025-08-20T10:00:00"
    snapshot_dt = None
    if ts_iso:
        # interpretamos en zona local -3 si viene naive
        try:
            snapshot_dt = datetime.fromisoformat(ts_iso)
        except Exception:
            snapshot_dt = None
    cfg_num = resolve_sources(cfg, snapshot_dt=snapshot_dt)
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

# ---------- Herramientas de tags (solo lectura) ----------

@bp.route("/power/tag_search", methods=["GET"])
def tag_search():
    table = _detect_table()
    if not table:
        return jsonify({"ok": False, "error": "No se detectó la tabla de analógicas"}), 500
    grp   = request.args.get("grp", "%")
    equip = request.args.get("equipment", "%")

    sql = f"""
    SELECT DISTINCT TOP 100 EQUIP_GRP, EQUIPMENT, ID
    FROM {table}
    WHERE EQUIP_GRP LIKE ? AND EQUIPMENT LIKE ?
    ORDER BY EQUIP_GRP, EQUIPMENT, ID
    """
    rows = sql_query(sql, [grp, equip]) or []
    return jsonify({"ok": True, "items": rows})

@bp.route("/power/probe_tag", methods=["POST"])
def probe_tag():
    data = request.get_json(force=True)
    equip_grp = data.get("equip_grp")
    equipment = data.get("equipment")
    signal_id = data.get("signal_id")   # puede ser string o número -> NO castear aquí
    scale = float(data.get("scale", 1.0))

    # 1) Si vino timestamp, buscamos el valor más cercano a T (±30 min)
    ts = data.get("timestamp")
    when_dt = _parse_ts(ts) if ts else None
    if ts and when_dt is None:
        # En 3.6 fromisoformat no existe; si no parsea, en vez de 400, hacemos fallback a "último bueno"
        when_dt = None

    if when_dt is not None:
        val = snapshot(equip_grp, equipment, signal_id, when_dt, tolerance_minutes=30)
        if val is not None:
            return jsonify({"ok": True, "value": val * scale})
        # si no hay datos cercanos a esa hora, caemos al último valor bueno reciente

    # 2) Último valor BUENO en ventana reciente (6 horas)
    end = datetime.now()
    start = end - timedelta(hours=6)
    df = series(equip_grp, equipment, signal_id, start, end)
    if df is None or df.empty:
        return jsonify({"ok": False, "error": "sin datos recientes"}), 404

    # filtro de calidad
    ok = (df.get("OLD", 0) == 0) & (df.get("BAD", 0) == 0)
    df = df.loc[ok]
    if df.empty:
        return jsonify({"ok": False, "error": "solo hay datos con OLD/BAD=1"}), 404

    # ordenar por alguna columna de tiempo conocida
    for c in ("TIME_FULL", "Time", "TIME", "timestamp"):
        if c in df.columns:
            df = df.sort_values(c)
            break

    try:
        val = float(df["VALUE"].iloc[-1])
    except Exception:
        return jsonify({"ok": False, "error": "valor inválido"}), 500

    return jsonify({"ok": True, "value": val * scale})

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
    shape = data.get("shape") or {}
    if L is None and shape.get("zip_path") and shape.get("attr") and (shape.get("value") is not None):
        try:
            from app.services.geo_len import zip_shp_line_length_km
            zip_path = shape["zip_path"]
            # si te llega como "/static/...", convertí a path real si hace falta
            if zip_path.startswith("/static/"):
                zip_path = os.path.join(bp.root_path, "..", "..", zip_path.lstrip("/"))
            L = zip_shp_line_length_km(zip_path, shape["attr"], shape["value"])
        except Exception:
            L = None
    # (dejas tu fallback KML como está)
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

# --- Endpoints para combos (GRUPO / EQUIPO / ID) ---
@bp.route("/power/tag_groups")
def tag_groups():
    est = list_hierarchy()
    return jsonify(sorted(list(est.keys())))

@bp.route("/power/tag_equipment", methods=["GET"])
def tag_equipment():
    grp = (request.args.get("grp") or "").strip()
    if not grp:
        return jsonify([])  # ⬅️ lista vacía

    est = list_hierarchy()  # mismo método que /analizador
    equipos = sorted(list((est.get(grp) or {}).keys()))
    return jsonify(equipos)  # ⬅️ devolver array

@bp.route("/power/tag_ids", methods=["GET"])
def tag_ids():
    grp   = (request.args.get("grp") or "").strip()
    equip = (request.args.get("equipment") or "").strip()
    if not grp or not equip:
        return jsonify([])

    est = list_hierarchy()
    raw = (est.get(grp, {}).get(equip, []))
    ids = sorted([str(x) for x in raw])   # ← todo string
    return jsonify(ids)

@bp.route("/power/debug_tag_meta")
def debug_tag_meta():
    t = _detect_table()
    c = _detect_time_col(t) if t else None
    return jsonify({"table": t, "time_col": c})