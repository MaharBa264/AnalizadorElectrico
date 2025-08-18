from app.services import sql
from app.services.files import EXPORTS
import os, csv


def sql_demo_rows():
    """
    Demo segura que no depende de tu schema:
    lista las primeras tablas de la DB (sys.tables).
    """
    try:
        rows = sql.query_list("SELECT TOP 20 name AS table_name FROM sys.tables ORDER BY name")
        return True, rows
    except Exception as e:
        return False, [{"error": str(e)}]



def list_tables():
    q = "SELECT name AS table_name FROM sys.tables ORDER BY name"
    return [r["table_name"] for r in sql.query_list(q)]

def table_preview(table: str, top: int = 50):
    # Whitelist: el nombre debe existir en sys.tables
    tables = set(list_tables())
    if table not in tables:
        raise ValueError("Tabla no permitida.")
    # Consulta segura (escapando con [])
    q = f"SELECT TOP {int(top)} * FROM [{table}]"
    return sql.query_list(q)

def export_preview_csv(table: str, rows: list):
    os.makedirs(EXPORTS, exist_ok=True)
    path = os.path.join(EXPORTS, f"preview_{table}.csv")
    if not rows:
        # crear CSV vacío con solo BOM para evitar problemas de Excel
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("") 
        return path
    hdr = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
