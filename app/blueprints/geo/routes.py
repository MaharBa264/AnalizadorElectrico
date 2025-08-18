from flask import render_template
from . import bp

@bp.route("/mapa_red")
def mapa_red():
    return render_template("geo/mapa_red.html")
