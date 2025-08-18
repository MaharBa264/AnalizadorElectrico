from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from . import bp
from ...extensions import login_manager
from .services import User, USER_DB, get_user

@login_manager.user_loader
def load_user(user_id):
    return get_user(user_id)

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            flash("Ingrese un usuario.", "warning")
            return redirect(url_for("auth.login"))
        role = "admin" if username.lower() == "admin" else "user"
        user = get_user(username) or User(username, role)
        USER_DB[username] = user
        login_user(user)
        flash("Sesión iniciada.", "success")
        return redirect(url_for("main.home"))
    return render_template("auth/login.html")

@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Sesión cerrada.", "info")
    return redirect(url_for("main.home"))
