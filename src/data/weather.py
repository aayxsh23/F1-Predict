"""Race-start weather forecast from Open-Meteo (free, no API key).

Past races use Open-Meteo's archive of forecasts, so the model trains on the
same kind of input a live prediction gets (a forecast), not on the weather the
race actually had, which nobody knows before Sunday."""
from pathlib import Path

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


# --- forecasts as they stood days before the race (train/serve match) ---
#
# The archive above is stitched from the first hours of each forecast run, so
# for a past race it is nearly a same-day forecast. A live forecast made on
# the Thursday is about three days out and much less sure about rain. Open-
# Meteo's Previous Runs API keeps what each day's forecast said: rain and wind
# from about March 2024, temperature from 2022. features.mask_for_stage puts
# the 3-day-ahead values into the before-practice stages and the 1-day-ahead
# ones into the Saturday stages, where they exist.
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEAD_DAYS = (3, 1)
LEADS_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather_leads.parquet"


def lead_forecast(lat: float, lon: float, start_utc, days: int) -> dict:
    """The race-window forecast as it stood `days` days before the race:
    {air_temp_forecast_d<days>, rain_mm_forecast_d<days>, wind_kph_forecast_d<days>}, NaN where not archived."""
    keys = {f"{k}_d{days}": np.nan for k in EMPTY}
    start = pd.Timestamp(start_utc)
    if pd.isna(start) or pd.isna(lat) or pd.isna(lon):
        return keys
    start = start.tz_localize(None) if start.tzinfo else start
    end = start + pd.Timedelta(hours=RACE_HOURS)
    names = {f"temperature_2m_previous_day{days}": "air_temp_forecast", f"precipitation_previous_day{days}": "rain_mm_forecast",
             f"wind_speed_10m_previous_day{days}": "wind_kph_forecast"}
    params = {"latitude": lat, "longitude": lon, "timezone": "UTC", "hourly": ",".join(names),
              "start_date": start.date().isoformat(), "end_date": end.date().isoformat()}
    try:
        hourly = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=15).json()["hourly"]
    except (requests.RequestException, KeyError, ValueError):
        return keys
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    w = h[(h["time"] >= start.floor("h")) & (h["time"] <= end)]
    for src, dst in names.items():
        if src in w and w[src].notna().any():
            keys[f"{dst}_d{days}"] = w[src].sum(min_count=1) if dst == "rain_mm_forecast" else w[src].mean()
    return keys


def update_leads(raw: pd.DataFrame) -> pd.DataFrame:
    """Fetch the lead-day forecasts for every race in `raw` (finished races,
    2022 on) not yet in data/raw/weather_leads.parquet, and save the table."""
    from src.features.circuit_reference import coords

    have = pd.read_parquet(LEADS_PATH) if LEADS_PATH.exists() else pd.DataFrame(columns=["season", "round"])
    done = set(zip(have["season"], have["round"]))
    races = raw[raw["season"] >= 2022].drop_duplicates(["season", "round"])
    rows = []
    for _, r in races.iterrows():
        if (r["season"], r["round"]) in done or pd.isna(r.get("race_start_utc")):
            continue
        lat, lon = coords(r["location"], int(r["season"]))
        row = {"season": int(r["season"]), "round": int(r["round"])}
        for d in LEAD_DAYS:
            row.update(lead_forecast(lat, lon, r["race_start_utc"], d))
        rows.append(row)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).sort_values(["season", "round"])
        LEADS_PATH.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(LEADS_PATH, index=False)
    return have


if __name__ == "__main__":
    from src.features.build_dataset import load_raw

    table = update_leads(load_raw())
    print(f"{len(table)} races in {LEADS_PATH}; with 3-day rain forecast: {int(table['rain_mm_forecast_d3'].notna().sum())}")
