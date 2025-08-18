from flask import Blueprint
bp = Blueprint("ml", __name__, template_folder="templates")
from . import routes  # noqa
