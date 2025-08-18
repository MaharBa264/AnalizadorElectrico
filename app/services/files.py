import os

BASE = os.path.abspath(os.getcwd())
STORAGE = os.path.join(BASE, "storage")
LOGS = os.path.join(STORAGE, "logs")
MODELS = os.path.join(STORAGE, "models", "lstm")
EXPORTS = os.path.join(STORAGE, "exports")
UPLOADS = os.path.join(STORAGE, "uploads")
TEMP = os.path.join(STORAGE, "temp")

INSTANCE = os.path.join(BASE, "instance")  # CSV canónicos y secretos

def ensure_dirs():
    for p in (LOGS, MODELS, EXPORTS, UPLOADS, TEMP, INSTANCE):
        os.makedirs(p, exist_ok=True)
