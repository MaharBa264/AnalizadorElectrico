from flask import Blueprint
bp = Blueprint("main", __name__, template_folder="templates")  # <-- clave
from . import routes  # noqa
