import os

class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me")
    TIMEZONE = "America/Argentina/San_Luis"
    DEBUG = False

class DevelopmentConfig(BaseConfig):
    DEBUG = True

class ProductionConfig(BaseConfig):
    pass

def resolve_config():
    env = os.getenv("APP_CONFIG", "development").lower()
    return DevelopmentConfig if env.startswith("dev") else ProductionConfig
