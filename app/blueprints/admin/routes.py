from flask import render_template, abort, current_app
from flask_login import login_required, current_user
from . import bp

@bp.route("/")
@login_required
def index():
    if getattr(current_user, "role", "user") != "admin":
        abort(403)
    return render_template("admin/index.html")

@bp.route("/diag")
@login_required
def diag():
    if getattr(current_user, "role", "user") != "admin":
        abort(403)
    results = {}

    # SQL
    try:
        from app.services import sql
        ok, msg = sql.test_ping()
        results["sql"] = {"ok": ok, "msg": msg}
    except Exception as e:
        results["sql"] = {"ok": False, "msg": str(e)}

    # Influx
    try:
        from app.services import influx
        ok, msg = influx.test_ping()
        results["influx"] = {"ok": ok, "msg": msg}
    except Exception as e:
        results["influx"] = {"ok": False, "msg": str(e)}

    current_app.logger.info("DIAG: %s", results)
    return render_template("admin/diag.html", results=results)
