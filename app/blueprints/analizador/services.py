from app.services import sql

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
