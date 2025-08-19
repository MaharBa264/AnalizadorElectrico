
import os
import folium
import geopandas as gpd

def crear_mapa_red(gdf_in, archivo_html):
    """
    Genera un HTML (Leaflet/Folium) con la red filtrada y popups básicos.
    - gdf_in: GeoDataFrame (líneas/geom) ya filtrado
    - archivo_html: ruta destino (por ej. app/static/mapa_red.html)
    """
    if not isinstance(gdf_in, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf_in, geometry="geometry")
    else:
        gdf = gdf_in

    gdf = gdf[gdf["geometry"].notnull()]
    if gdf.empty:
        # centro por defecto si no hay datos
        centro = [-33.3, -65.3]
        m = folium.Map(location=centro, zoom_start=5, tiles="OpenStreetMap")
    else:
        # centro en el bounding box de las geometrías
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        cy = (bounds[1] + bounds[3]) / 2.0
        cx = (bounds[0] + bounds[2]) / 2.0
        m = folium.Map(location=[cy, cx], zoom_start=12, tiles="OpenStreetMap")

        # estilo líneas
        def style_fn(_):
            return {"color": "#3388ff", "weight": 3, "opacity": 0.9}

        # Popups/tooltip
        for _, row in gdf.iterrows():
            info = f"""
            <b>Alimentado:</b> {row.get('Alimentado','')}<br>
            <b>Equipo:</b> {row.get('equip_grp','')} [{row.get('equipment','')}]<br>
            <b>Tensión nominal:</b> {row.get('Tensión n','') or row.get('TensiÃ³n n','')} kV<br>
            <b>Material:</b> {row.get('Material','')}<br>
            <b>Tipo:</b> {row.get('Tipo','')}<br>
            """
            try:
                geom = row["geometry"]
                folium.GeoJson(geom, style_function=style_fn,
                    tooltip=row.get('Etiqueta', row.get('Alimentado','')),
                    popup=folium.Popup(info, max_width=260)
                ).add_to(m)
            except Exception:
                continue

    os.makedirs(os.path.dirname(archivo_html), exist_ok=True)
    m.save(archivo_html)
    return archivo_html
