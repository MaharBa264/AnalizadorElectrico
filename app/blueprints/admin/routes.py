from flask import render_template, abort
from flask_login import login_required, current_user
from . import bp

@bp.route("/")
@login_required
def index():
    if getattr(current_user, "role", "user") != "admin":
        abort(403)
    return render_template("admin/index.html")
