from flask import render_template, request, make_response, abort, flash, redirect, url_for
from flask_login import login_required, current_user
from . import bp
from .services import sql_demo_rows, list_tables, table_preview, export_preview_csv

@bp.route("/")
def index():
    return render_template("analizador/index.html")

@bp.route("/sql-demo")
def sql_demo():
    ok, rows = sql_demo_rows()
    return render_template("analizador/sql_demo.html", ok=ok, rows=rows)

@bp.route("/sql-browser")
@login_required
def sql_browser():
    if getattr(current_user, "role", "user") != "admin":
        abort(403)
    table = request.args.get("table")
    top = request.args.get("top", "50")
    try:
        top = int(top)
        if top <= 0 or top > 1000:  # límite sano
            top = 50
    except ValueError:
        top = 50

    tables = []
    rows = []
    error = None
    try:
        tables = list_tables()
        if table:
            rows = table_preview(table, top=top)
    except Exception as e:
        error = str(e)

    return render_template("analizador/sql_browser.html",
                           tables=tables, table=table, top=top, rows=rows, error=error)

@bp.route("/sql-browser/export")
@login_required
def sql_browser_export():
    if getattr(current_user, "role", "user") != "admin":
        abort(403)
    table = request.args.get("table")
    if not table:
        flash("Elegí una tabla primero.", "warning")
        return redirect(url_for("analizador.sql_browser"))
    try:
        rows = table_preview(table, top=1000)
        # Exporto a CSV en disco Y devuelvo attachment directo
        import io, csv
        si = io.StringIO()
        if rows:
            hdr = list(rows[0].keys())
            w = csv.DictWriter(si, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        csv_bytes = si.getvalue().encode("utf-8-sig")
        resp = make_response(csv_bytes)
        resp.headers["Content-Type"] = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = f'attachment; filename="preview_{table}.csv"'
        return resp
    except Exception as e:
        flash(f"Error exportando: {e}", "danger")
        return redirect(url_for("analizador.sql_browser"))
