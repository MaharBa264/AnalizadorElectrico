import os, requests

API_KEY = os.getenv("WEATHERAPI_KEY", "")

def _get(url, params):
    if not API_KEY:
        raise RuntimeError("Falta WEATHERAPI_KEY en .env")
    params = dict(params or {})
    params["key"] = API_KEY
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def history_hourly(lat, lon, date_str):
    """
    date_str: YYYY-MM-DD (UTC)
    """
    url = "https://api.weatherapi.com/v1/history.json"
    return _get(url, {"q": f"{lat},{lon}", "dt": date_str, "aqi": "no", "alerts": "no"})

def forecast_hourly(lat, lon, days=3):
    url = "https://api.weatherapi.com/v1/forecast.json"
    return _get(url, {"q": f"{lat},{lon}", "days": days, "aqi": "no", "alerts": "no"})
