from flask import Blueprint
bp = Blueprint("geo", __name__, template_folder="templates")
from . import routes  # noqa
