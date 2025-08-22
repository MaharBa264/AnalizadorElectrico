# app/services/geo_len.py
import math, xml.etree.ElementTree as ET
import os, io, zipfile, unicodedata

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def _coords_length_km(coords_str: str) -> float:
    pts = []
    for token in coords_str.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon = float(parts[0]); lat = float(parts[1])
            pts.append((lat, lon))
    L = 0.0
    for (lat1,lon1),(lat2,lon2) in zip(pts, pts[1:]):
        L += _haversine_km(lat1,lon1,lat2,lon2)
    return L

def kml_line_length_km(kml_path: str, name: str) -> float:
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"k":"http://www.opengis.net/kml/2.2"}
    for pm in root.findall(".//k:Placemark", ns):
      nm = pm.find("k:name", ns)
      if nm is not None and (nm.text or "").strip() == name:
          coords = pm.find(".//k:LineString/k:coordinates", ns)
          if coords is not None and coords.text:
              return _coords_length_km(coords.text)
    raise ValueError(f"No encontré {name} en {kml_path}")


def shp_line_length_km(shp_path: str, attr_name: str, attr_value):
    """
    Calcula la longitud de una polilínea en un shapefile .shp filtrando por atributo.
    Requiere el paquete 'pyshp' (shapefile).
    """
    try:
        import shapefile  # pip install pyshp
    except Exception as e:
        raise RuntimeError("Falta dependencia 'pyshp'. Instala con: pip install pyshp") from e

    sf = shapefile.Reader(shp_path)
    fields = [f[0] for f in sf.fields[1:]]  # salta DeletionFlag
    if attr_name not in fields:
        raise ValueError(f"Campo {attr_name} no encontrado en {shp_path}. Campos: {fields}")
    idx = fields.index(attr_name)
    L_total = 0.0
    for sr in sf.iterShapeRecords():
        if sr.record[idx] == attr_value:
            shp = sr.shape
            pts = shp.points
            parts = list(shp.parts) + [len(pts)]
            for i in range(len(parts)-1):
                seg = pts[parts[i]:parts[i+1]]
                for (lon1, lat1), (lon2, lat2) in zip(seg, seg[1:]):
                    L_total += _haversine_km(lat1, lon1, lat2, lon2)
    if L_total == 0.0:
        raise ValueError(f"No hallé geometría con {attr_name}={attr_value} en {shp_path}")
    return L_total

def zip_shp_line_length_km(zip_path, attr_name, attr_value):
    import zipfile, io, os
    try:
        import shapefile
    except Exception as e:
        raise RuntimeError("Instala 'pyshp'") from e

    if not os.path.isabs(zip_path):
        zip_path = os.path.join(os.getcwd(), zip_path.lstrip("/"))

    with zipfile.ZipFile(zip_path, "r") as z:
        shp = next((n for n in z.namelist() if n.lower().endswith(".shp")), None)
        shx = next((n for n in z.namelist() if n.lower().endswith(".shx")), None)
        dbf = next((n for n in z.namelist() if n.lower().endswith(".dbf")), None)
        if not (shp and shx and dbf):
            raise ValueError("ZIP sin .shp/.shx/.dbf")

        r = shapefile.Reader(
            shp=io.BytesIO(z.read(shp)),
            shx=io.BytesIO(z.read(shx)),
            dbf=io.BytesIO(z.read(dbf)),
        )

    fields = [f[0] for f in r.fields[1:]]
    if attr_name not in fields:
        raise ValueError("Campo {} no encontrado".format(attr_name))
    idx = fields.index(attr_name)

    L = 0.0
    for sr in r.iterShapeRecords():
        if sr.record[idx] == attr_value:
            pts = sr.shape.points
            parts = list(sr.shape.parts) + [len(pts)]
            for i in range(len(parts)-1):
                seg = pts[parts[i]:parts[i+1]]
                for (lon1, lat1), (lon2, lat2) in zip(seg, seg[1:]):
                    L += _haversine_km(lat1, lon1, lat2, lon2)
    if L == 0.0:
        raise ValueError("No encontré geometría con {}={}".format(attr_name, attr_value))
    return L


def _ensure_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path.lstrip("/"))
'''
def zip_shp_unique_values(zip_path: str, attr_name: str):
    """
    Devuelve lista ordenada de valores únicos del atributo 'attr_name' dentro del SHP del ZIP.
    """
    try:
        import shapefile
    except Exception as e:
        raise RuntimeError("Falta dependencia 'pyshp' (pip install pyshp)") from e

    zip_path = _ensure_abs(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        shp = next((n for n in z.namelist() if n.lower().endswith(".shp")), None)
        shx = next((n for n in z.namelist() if n.lower().endswith(".shx")), None)
        dbf = next((n for n in z.namelist() if n.lower().endswith(".dbf")), None)
        if not (shp and shx and dbf):
            raise ValueError("El ZIP no contiene .shp/.shx/.dbf")
        r = shapefile.Reader(
            shp=io.BytesIO(z.read(shp)),
            shx=io.BytesIO(z.read(shx)),
            dbf=io.BytesIO(z.read(dbf)),
        )

    fields = [f[0] for f in r.fields[1:]]
    if attr_name not in fields:
        raise ValueError(f"Campo {attr_name} no encontrado. Campos: {fields}")
    idx = fields.index(attr_name)

    vals = set()
    for sr in r.iterShapeRecords():
        vals.add(sr.record[idx])
    # devuelvo como strings para el front (más simple)
    out = sorted([str(v) for v in vals if v is not None])
    return out

def zip_shp_line_profile(zip_path: str, group_attr: str, group_value, material_attr: str = None):
    """
    Suma la longitud de TODOS los segmentos con group_attr == group_value
    y (si material_attr) descompone por material, devolviendo porcentajes.
    Estructura:
    {
      "length_km": 12.345,
      "materials": [{"material": "ACSR 120", "length_km": 7.89, "percent": 63.9}, ...]
    }
    """
    try:
        import shapefile
    except Exception as e:
        raise RuntimeError("Falta dependencia 'pyshp' (pip install pyshp)") from e

    zip_path = _ensure_abs(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        shp = next((n for n in z.namelist() if n.lower().endswith(".shp")), None)
        shx = next((n for n in z.namelist() if n.lower().endswith(".shx")), None)
        dbf = next((n for n in z.namelist() if n.lower().endswith(".dbf")), None)
        if not (shp and shx and dbf):
            raise ValueError("El ZIP no contiene .shp/.shx/.dbf")
        r = shapefile.Reader(
            shp=io.BytesIO(z.read(shp)),
            shx=io.BytesIO(z.read(shx)),
            dbf=io.BytesIO(z.read(dbf)),
        )

    fields = [f[0] for f in r.fields[1:]]
    if group_attr not in fields:
        raise ValueError(f"Campo {group_attr} no encontrado. Campos: {fields}")
    idx_group = fields.index(group_attr)

    idx_mat = None
    if material_attr:
        if material_attr not in fields:
            # si no existe, devolvemos sin composición
            material_attr = None
        else:
            idx_mat = fields.index(material_attr)

    # haversine
    from math import radians, sin, cos, sqrt, atan2
    def hav(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2.0)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2.0)**2
        return R * 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    L_total = 0.0
    comp = {}  # material -> length_km

    # normalizo group_value a string para comparar de forma robusta
    gv = str(group_value)

    for sr in r.iterShapeRecords():
        if str(sr.record[idx_group]) != gv:
            continue
        pts = sr.shape.points
        if not pts:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        seg_len = 0.0
        for i in range(len(parts)-1):
            seg = pts[parts[i]:parts[i+1]]
            for (lon1, lat1), (lon2, lat2) in zip(seg, seg[1:]):
                seg_len += hav(lat1, lon1, lat2, lon2)

        L_total += seg_len

        if idx_mat is not None:
            mat = sr.record[idx_mat]
            mat = str(mat) if mat is not None else "SIN_MATERIAL"
            comp[mat] = comp.get(mat, 0.0) + seg_len

    materials = []
    if comp and L_total > 0:
        for m, lk in comp.items():
            materials.append({"material": m, "length_km": lk, "percent": (lk / L_total) * 100.0})
        materials.sort(key=lambda x: x["length_km"], reverse=True)

    if L_total == 0.0 and not materials:
        # No se encontró la línea; devolvemos estructura vacía
        return {"length_km": 0.0, "materials": []}

    return {"length_km": L_total, "materials": materials}
'''
def _zip_open_reader(zip_path: str):
    """Abre SHP del ZIP. Usa .cpg si existe; si no, prueba utf-8 -> latin-1 -> cp1252."""
    try:
        import shapefile
    except Exception as e:
        raise RuntimeError("Falta dependencia 'pyshp' (pip install pyshp)") from e

    zip_path = _ensure_abs(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        shp = next((n for n in z.namelist() if n.lower().endswith(".shp")), None)
        shx = next((n for n in z.namelist() if n.lower().endswith(".shx")), None)
        dbf = next((n for n in z.namelist() if n.lower().endswith(".dbf")), None)
        cpg = next((n for n in z.namelist() if n.lower().endswith(".cpg")), None)
        if not (shp and shx and dbf):
            raise ValueError("El ZIP no contiene .shp/.shx/.dbf")

        enc_order = []
        if cpg:
            try:
                enc_guess = z.read(cpg).decode("ascii", errors="ignore").strip()
                if enc_guess:
                    enc_order.append(enc_guess)
            except Exception:
                pass
        enc_order += ["utf-8", "latin-1", "cp1252"]

        last_err = None
        for enc in enc_order:
            try:
                import shapefile
                r = shapefile.Reader(
                    shp=io.BytesIO(z.read(shp)),
                    shx=io.BytesIO(z.read(shx)),
                    dbf=io.BytesIO(z.read(dbf)),
                    encoding=enc
                )
                return r, enc
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("No se pudo abrir el SHP del ZIP")

def zip_shp_fields(zip_path: str):
    r, _ = _zip_open_reader(zip_path)
    return [f[0] for f in r.fields[1:]]

def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.lower().replace(" ", "").replace("_", "")

def _choose_id_field(fields):
    """Elige un campo 'ID de línea' razonable."""
    norm_map = { _norm(f): f for f in fields }
    for key in ["lineaid","idlinea","linea","alimentador","etiqueta","nombre","id","codigo","tag","name"]:
        if key in norm_map:
            return norm_map[key]
    # fallback heurístico: cualquier campo que contenga 'eti' (etiqueta)
    for f in fields:
        if "ETI" in f.upper(): return f
    return fields[0]

def zip_shp_unique_values(zip_path: str, attr_name: str = None):
    """Valores únicos del atributo (si no se pasa, se autodetecta)."""
    r, _ = _zip_open_reader(zip_path)
    fields = [f[0] for f in r.fields[1:]]
    if not attr_name or attr_name not in fields:
        attr_name = _choose_id_field(fields)
    idx = fields.index(attr_name)
    vals = set()
    for sr in r.iterShapeRecords():
        vals.add(sr.record[idx])
    return sorted([str(v) for v in vals if v is not None]), attr_name

def zip_shp_line_profile(zip_path: str, group_attr: str, group_value, material_attr: str = None):
    """Suma longitudes por group_attr==group_value y descompone por material (si existe)."""
    r, _ = _zip_open_reader(zip_path)
    fields = [f[0] for f in r.fields[1:]]
    if not group_attr or group_attr not in fields:
        group_attr = _choose_id_field(fields)
    idx_group = fields.index(group_attr)
    idx_mat = fields.index(material_attr) if (material_attr and material_attr in fields) else None

    from math import radians, sin, cos, sqrt, atan2
    def hav(lat1, lon1, lat2, lon2):
        R=6371.0
        dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
        a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R*2*atan2(sqrt(a), sqrt(1-a))

    L_total, comp = 0.0, {}
    gv = str(group_value)

    for sr in r.iterShapeRecords():
        if str(sr.record[idx_group]) != gv: continue
        pts = sr.shape.points
        if not pts: continue
        parts = list(sr.shape.parts) + [len(pts)]
        seg_len = 0.0
        for i in range(len(parts)-1):
            seg = pts[parts[i]:parts[i+1]]
            for (lon1,lat1),(lon2,lat2) in zip(seg, seg[1:]):
                seg_len += hav(lat1,lon1,lat2,lon2)
        L_total += seg_len
        if idx_mat is not None:
            mat = sr.record[idx_mat]
            mat = str(mat) if mat is not None else "SIN_MATERIAL"
            comp[mat] = comp.get(mat, 0.0) + seg_len

    materials = []
    if comp and L_total>0:
        for m,lk in comp.items():
            materials.append({"material":m,"length_km":lk,"percent":(lk/L_total)*100.0})
        materials.sort(key=lambda x: x["length_km"], reverse=True)

    return {"length_km": L_total, "materials": materials, "used_attr": group_attr}


# Sugerencias de IDs (filtrando por texto y con límite)
def zip_shp_suggest(zip_path: str, attr_name: str = None, q: str = "", limit: int = 50):
    r, _ = _zip_open_reader(zip_path)
    fields = [f[0] for f in r.fields[1:]]
    if not attr_name or attr_name not in fields:
        attr_name = _choose_id_field(fields)
    idx = fields.index(attr_name)

    def _norm(s: str) -> str:
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        return s.lower()

    qn = _norm(q or "")
    seen, items = set(), []
    truncated = False

    for sr in r.iterShapeRecords():
        v = sr.record[idx]
        if v is None:
            continue
        s = str(v)
        if qn and qn not in _norm(s):
            continue
        if s in seen:
            continue
        seen.add(s)
        items.append(s)
        if len(items) >= max(1, int(limit)):
            truncated = True
            break

    return {"items": items, "attr": attr_name, "truncated": truncated}


def _norm_txt(s):
    if s is None: return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().upper()

def _choose_group_field(fields):
    cand = ["RED","GRUPO","SISTEMA","ZONA","SUBRED"]
    upp = {f.upper(): f for f in fields}
    for c in cand:
        if c in upp: return upp[c]
    return fields[0]

def _choose_feeder_field(fields):
    cand = ["ALIMENTADO","ALIMENTADOR","ETIQUETA","FEEDER","CIRCUITO","LINEA","ID","NOMBRE"]
    upp = {f.upper(): f for f in fields}
    for c in cand:
        if c in upp: return upp[c]
    return fields[0]

def zip_shp_feeder_profile(zip_path: str, group_value: str, feeder_value: str,
                           group_attr: str = None, feeder_attr: str = None,
                           material_attr: str = None):
    """
    Suma TODAS las geometrías cuyo (group_attr == group_value) y
    (feeder_attr == feeder_value)   [comparación normalizada, case/acentos-insensible].
    Devuelve longitud total y % por material si existe.
    """
    r, _ = _zip_open_reader(zip_path)
    fields = [f[0] for f in r.fields[1:]]
    group_attr  = group_attr  if group_attr  in fields else _choose_group_field(fields)
    feeder_attr = feeder_attr if feeder_attr in fields else _choose_feeder_field(fields)
    idx_g = fields.index(group_attr)
    idx_f = fields.index(feeder_attr)
    idx_m = fields.index(material_attr) if (material_attr and material_attr in fields) else None

    G = _norm_txt(group_value)
    F = _norm_txt(feeder_value)

    from math import radians, sin, cos, sqrt, atan2
    def hav(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    L_total = 0.0
    comp = {}
    segs = 0

    for sr in r.iterShapeRecords():
        g = _norm_txt(sr.record[idx_g])
        f = _norm_txt(sr.record[idx_f])
        # igualdad estricta; si no matchea nada, adelante lo relajamos a 'contains' en el endpoint
        if g != G or f != F:
            continue
        pts = sr.shape.points
        if not pts:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        seg_len = 0.0
        for i in range(len(parts)-1):
            seg = pts[parts[i]:parts[i+1]]
            for (lon1,lat1),(lon2,lat2) in zip(seg, seg[1:]):
                seg_len += hav(lat1,lon1,lat2,lon2)
        L_total += seg_len
        segs += 1
        if idx_m is not None:
            m = _norm_txt(sr.record[idx_m]) or "SIN_MATERIAL"
            comp[m] = comp.get(m, 0.0) + seg_len

    # Si no encontramos nada por igualdad, reintentamos con 'contains'
    if L_total == 0.0:
        for sr in r.iterShapeRecords():
            g = _norm_txt(sr.record[idx_g])
            f = _norm_txt(sr.record[idx_f])
            if (G and G not in g) or (F and F not in f):
                continue
            pts = sr.shape.points
            if not pts: continue
            parts = list(sr.shape.parts) + [len(pts)]
            seg_len = 0.0
            for i in range(len(parts)-1):
                seg = pts[parts[i]:parts[i+1]]
                for (lon1,lat1),(lon2,lat2) in zip(seg, seg[1:]):
                    seg_len += hav(lat1,lon1,lat2,lon2)
            L_total += seg_len
            segs += 1
            if idx_m is not None:
                m = _norm_txt(sr.record[idx_m]) or "SIN_MATERIAL"
                comp[m] = comp.get(m, 0.0) + seg_len

    materials = []
    if comp and L_total > 0:
        for m, lk in comp.items():
            materials.append({"material": m, "length_km": lk, "percent": (lk/L_total)*100.0})
        materials.sort(key=lambda x: x["length_km"], reverse=True)

    return {
        "length_km": L_total,
        "materials": materials,
        "used_fields": {"group": group_attr, "feeder": feeder_attr},
        "segments": segs
    }