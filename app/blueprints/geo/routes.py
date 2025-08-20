
from flask import render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required
from pathlib import Path
from . import bp
from app.blueprints.analizador.services import get_signal_hierarchy

@bp.route("/mapa_red", methods=["GET"])
@login_required
def mapa_red():
    """
    Página de selección (dos combos, uno arriba del otro).
    Al enviar, redirige a /geo/ver con los parámetros.
    """
    estructura = get_signal_hierarchy()
    return render_template("geo/mapa_red.html", estructura=estructura)

@bp.route("/ver", methods=["GET"])
@login_required
def ver_mapa():
    equip_grp = request.args.get("equip_grp","").strip()
    equipment = request.args.get("equipment","").strip()
    if not equipment:
        flash("Seleccioná un grupo y un equipo para ver el mapa.", "warning")
        return redirect(url_for(".mapa_red"))

    # Shapefile comprimido en /static/geodata/red_electrica.zip
    zip_path = Path(current_app.static_folder) / "geodata" / "red_electrica.zip"
    if not zip_path.exists():
        flash("No se encontró /static/geodata/red_electrica.zip con el shapefile.", "danger")
        return redirect(url_for(".mapa_red"))

    return render_template("geo/mapa_red_client.html",
                           equip_grp=equip_grp,
                           equipment=equipment,
                           zip_url=url_for("static", filename="geodata/red_electrica.zip"))
