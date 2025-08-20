# app/services/geo_len.py
import math, xml.etree.ElementTree as ET

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
