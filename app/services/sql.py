import os

def _conn_str():
    server = os.getenv("SQL_SERVER", "127.0.0.1")
    port   = os.getenv("SQL_PORT", "1433")
    db     = os.getenv("SQL_DB", "master")
    uid    = os.getenv("SQL_UID", "")
    pwd    = os.getenv("SQL_PWD", "")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    return (
        "DRIVER={" + driver + "};"
        "SERVER=" + server + "," + str(port) + ";"
        "DATABASE=" + db + ";"
        "UID=" + uid + ";"
        "PWD=" + pwd + ";"
        "TrustServerCertificate=yes;"
    )

def connect(timeout=10):
    import pyodbc
    return pyodbc.connect(_conn_str(), timeout=timeout)

def query_dataframe(sql, params=None):
    import pandas as pd
    with connect() as cn:
        return pd.read_sql(sql, cn, params=params)

def test_ping():
    """Devuelve (ok:bool, detalle:str)"""
    try:
        with connect() as cn:
            cur = cn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        return True, "SQL OK"
    except Exception as e:
        return False, str(e)

def query_list(sql, params=None):
    """Devuelve lista de dicts sin requerir pandas."""
    with connect() as cn:
        cur = cn.cursor()
        cur.execute(sql, params or [])
        cols = [c[0] for c in cur.description]
        out = []
        for row in cur.fetchall():
            out.append({c: v for c, v in zip(cols, row)})
        return out
