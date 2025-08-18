from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, username: str, role: str = "user"):
        self.id = username
        self.role = role

# Minimal "DB" en memoria (solo dev)
USER_DB = {}

def get_user(username: str):
    return USER_DB.get(username)
