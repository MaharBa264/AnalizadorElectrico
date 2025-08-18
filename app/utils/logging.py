import logging, os
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    # Nivel por config
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    app.logger.setLevel(level)

    # Archivo rotativo en storage/logs/app.log (fuera del repo)
    log_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    # Evitar dobles handlers si se recarga
    for h in list(app.logger.handlers):
        app.logger.removeHandler(h)
    app.logger.addHandler(file_handler)

    # También a consola (útil en dev)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    app.logger.addHandler(console)
