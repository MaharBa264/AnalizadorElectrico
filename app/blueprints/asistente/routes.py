from flask import render_template
from . import bp

@bp.route("/asistente")
def index():
    return render_template("asistente/index.html")
