from flask import render_template
from . import bp

@bp.route("/")
def home():
    return render_template("main/home.html")

@bp.route("/health")
def health():
    return {"status": "ok"}
