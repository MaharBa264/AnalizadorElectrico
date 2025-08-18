from datetime import datetime, timedelta, timezone

# Offset fijo -3 (requisito del proyecto)
LOCAL_TZ = timezone(timedelta(hours=-3), name="America/Argentina/-03")

def now_local():
    """Datetime timezone-aware en -03."""
    return datetime.now(LOCAL_TZ)

def as_local(dt):
    """Convierte naive o tz-aware a -03."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)
