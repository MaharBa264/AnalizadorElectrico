import os
from flask import Flask
from .extensions import login_manager

def create_app(config_object=None):
    # Cargar .env temprano (compat. python-dotenv 0.19.x)
    try:
        from dotenv import load_dotenv
        load_dotenv()  # lee ./.env y también instance/.env si existe
    except Exception:
        pass

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object or "config.DevelopmentConfig")

    # Asegura carpetas
    os.makedirs(app.instance_path, exist_ok=True)

    # === Logging unificado (a storage/logs/app.log) ===
    try:
        from .utils.logging import setup_logging
        setup_logging(app)
    except Exception as e:
        app.logger.warning("No se pudo inicializar logging: %s", e)

        # Crear carpetas storage/*
    try:
        from .services.files import ensure_dirs
        ensure_dirs()
    except Exception as e:
        app.logger.warning("No se pudieron crear carpetas de storage: %s", e)


    # Extensiones
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Blueprints (igual que antes)
    from .blueprints.main import bp as main_bp
    app.register_blueprint(main_bp)

    from .blueprints.auth import bp as auth_bp
    app.register_blueprint(auth_bp)

    from .blueprints.analizador import bp as analizador_bp
    app.register_blueprint(analizador_bp, url_prefix="/analizador")

    from .blueprints.asistente import bp as asistente_bp
    app.register_blueprint(asistente_bp)

    from .blueprints.admin import bp as admin_bp
    app.register_blueprint(admin_bp, url_prefix="/admin")

    from .blueprints.ml import bp as ml_bp
    app.register_blueprint(ml_bp, url_prefix="/ml")

    from .blueprints.geo import bp as geo_bp
    app.register_blueprint(geo_bp, url_prefix="/geo")

    return app
