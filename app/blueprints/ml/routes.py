from flask import render_template, request, flash, redirect, url_for
from . import bp

@bp.route("/")
def index():
    return render_template("ml/index.html")

@bp.route("/train", methods=["POST"])
def train():
    """
    Placeholder: aún no conectamos a tus datos reales.
    Muestra mensaje indicando qué falta.
    """
    flash("Entrenamiento no implementado aún: migrá primero data/features.", "warning")
    return redirect(url_for("ml.index"))

@bp.route("/forecast", methods=["POST"])
def forecast():
    flash("Forecast no implementado aún: migrá primero model/forecast.", "warning")
    return redirect(url_for("ml.index"))
