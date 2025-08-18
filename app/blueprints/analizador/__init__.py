from flask import Blueprint
bp = Blueprint("analizador", __name__, template_folder="templates")
from . import routes  # noqa
