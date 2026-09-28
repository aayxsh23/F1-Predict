"""Race-start weather forecast from Open-Meteo (free, no API key).

Past races use Open-Meteo's archive of forecasts, so the model trains on the
same kind of input a live prediction gets (a forecast), not on the weather the
race actually had, which nobody knows before Sunday."""
import numpy as np
import pandas as pd
import requests

LIVE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
RACE_HOURS = 2
EMPTY = {"air_temp_forecast": np.nan, "rain_mm_forecast": np.nan, "wind_kph_forecast": np.nan}


def race_forecast(lat: float, lon: float, start_utc) -> dict:
    """Mean air temperature (C), total rain (mm) and mean wind (km/h) over the
    two hours from race start. NaNs when the race is too far out to forecast
    (Open-Meteo covers 16 days) or the request fails."""
    start = pd.Timestamp(start_utc)
    if pd.isna(start) or pd.isna(lat) or pd.isna(lon):
        return dict(EMPTY)
    start = start.tz_localize(None) if start.tzinfo else start
    end = start + pd.Timedelta(hours=RACE_HOURS)
    url = ARCHIVE_URL if start < pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=2) else LIVE_URL
    params = {
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
    }
    try:
        hourly = requests.get(url, params=params, timeout=15).json()["hourly"]
    except (requests.RequestException, KeyError, ValueError):
        return dict(EMPTY)
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    w = h[(h["time"] >= start.floor("h")) & (h["time"] <= end)]
    if w.empty:
        return dict(EMPTY)
    return {
        "air_temp_forecast": w["temperature_2m"].mean(),
        "rain_mm_forecast": w["precipitation"].sum(min_count=1),
        "wind_kph_forecast": w["wind_speed_10m"].mean(),
    }
