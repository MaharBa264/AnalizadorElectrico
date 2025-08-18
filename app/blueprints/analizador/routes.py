from flask import render_template, request
from . import bp
from .services import sql_demo_rows

@bp.route("/")
def index():
    # Home del analizador: links a demos/tools
    return render_template("analizador/index.html")

@bp.route("/sql-demo")
def sql_demo():
    ok, rows = sql_demo_rows()
    return render_template("analizador/sql_demo.html", ok=ok, rows=rows)
