from app import create_app
from config import ProductionConfig

app = create_app(ProductionConfig)

# Gunicorn apunta a: wsgi:app
