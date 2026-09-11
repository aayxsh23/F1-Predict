"""Pull one row per driver per race (grid, quali gap, practice pace, weather,
race outcome) from FastF1 and cache it to data/raw/<season>/<round>_<location>.parquet.

Resumable: re-running skips races that already have a cached raw file.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fastf1_client import event_schedule, load_session

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "races"
DEFAULT_SEASONS = [2022, 2023, 2024]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _clean_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag, non-in/out laps only — used for race-pace and pit-stop stats."""
    if laps is None or laps.empty:
        return laps
    mask = (
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["TrackStatus"].astype(str).eq("1")
        & laps["LapTime"].notna()
    )
    return laps[mask]


def _practice_pace(season: int, round_number: int) -> pd.DataFrame:
    """Best clean lap per driver across whichever FP sessions exist (FP1-FP3),
    normalized to % delta over the fastest lap among all of them combined
    (comparable across circuits)."""
    frames = []
    for code in ("FP1", "FP2", "FP3"):
        try:
            s = load_session(season, round_number, code, laps=True, weather=False)
            laps = s.laps
            if laps is not None and not laps.empty:
                frames.append(laps[["Driver", "LapTime"]].dropna())
        except Exception as exc:  # sprint weekends lack FP2/FP3
            log.info("  no %s (%s)", code, exc)
    if not frames:
        return pd.DataFrame(columns=["Driver", "practice_pace"])
    all_laps = pd.concat(frames, ignore_index=True)
    best = all_laps.groupby("Driver")["LapTime"].min()
    fastest = best.min()
    pct = ((best - fastest) / fastest * 100).rename("practice_pace")
    return pct.reset_index()


def _quali_features(season: int, round_number: int) -> pd.DataFrame:
    """Full Q1/Q2/Q3 times (seconds) plus the derived gap-to-pole, so the raw
    per-session times are available on disk even though only quali_gap_to_pole
    feeds the Phase 1 feature table."""
    q = load_session(season, round_number, "Q", laps=False, weather=False)
    res = q.results.copy()
    for col in ("Q1", "Q2", "Q3"):
        if col not in res.columns:
            res[col] = pd.NaT
    res["quali_best"] = res[["Q1", "Q2", "Q3"]].min(axis=1)
    pole = res["quali_best"].min()
    res["quali_gap_to_pole"] = (res["quali_best"] - pole).dt.total_seconds()
    res["q1_time"] = res["Q1"].dt.total_seconds()
    res["q2_time"] = res["Q2"].dt.total_seconds()
    res["q3_time"] = res["Q3"].dt.total_seconds()
    return res[["Abbreviation", "quali_gap_to_pole", "q1_time", "q2_time", "q3_time"]]


def _weather_features(session) -> dict:
    w = session.weather_data
    if w is None or w.empty:
        return dict(air_temp=np.nan, track_temp=np.nan, rain_probability=np.nan,
                     wind_speed=np.nan, wet_track_probability=np.nan)
    return dict(
        air_temp=w["AirTemp"].mean(),
        track_temp=w["TrackTemp"].mean(),
        rain_probability=w["Rainfall"].mean(),
        wind_speed=w["WindSpeed"].mean(),
        wet_track_probability=((w["Rainfall"]) | (w["Humidity"] > 85)).mean(),
    )


def ingest_race(season: int, round_number: int, location: str, race_date) -> pd.DataFrame | None:
    log.info("Ingesting %s round %s (%s)", season, round_number, location)
    try:
        race = load_session(season, round_number, "R", laps=True, weather=True)
    except Exception as exc:
        log.warning("  skipping, race session failed to load: %s", exc)
        return None

    results = race.results.copy()
    if results.empty:
        log.warning("  skipping, no results")
        return None

    laps = race.laps
    clean = _clean_race_laps(laps)
    if clean is not None and not clean.empty:
        fastest = clean["LapTime"].min()
        race_pace = (
            clean.groupby("Driver")["LapTime"].mean().sub(fastest).div(fastest).mul(100)
        ).rename("race_pace_pct")
        stops = (laps.groupby("Driver")["Stint"].max() - 1).rename("num_pit_stops")
    else:
        race_pace = pd.Series(dtype=float, name="race_pace_pct")
        stops = pd.Series(dtype=float, name="num_pit_stops")

    starting_compound = (
        laps[laps["LapNumber"] == 1][["Driver", "Compound"]]
        .drop_duplicates("Driver")
        .set_index("Driver")["Compound"]
        .rename("starting_compound")
        if laps is not None and not laps.empty
        else pd.Series(dtype=object, name="starting_compound")
    )

    df = results[[
        "Abbreviation", "DriverNumber", "TeamName", "GridPosition", "Position",
        "Points", "Status", "Time",
    ]].rename(columns={
        "Abbreviation": "driver", "DriverNumber": "driver_number", "TeamName": "team",
        "GridPosition": "grid_position", "Position": "finish_position",
        "Points": "points", "Status": "status",
    })
    # FastF1's Time is the winner's absolute race duration, and everyone else's
    # gap to that winner -- both leak straight to 0 for the winner, the target
    # a "predicted race time" model actually wants. NaN (DNF/not classified)
    # stays NaN, not imputed -- same honest-gap philosophy as every other target.
    df["gap_to_winner_seconds"] = df["Time"].dt.total_seconds()
    df.loc[df["finish_position"] == 1, "gap_to_winner_seconds"] = 0.0
    df = df.drop(columns=["Time"])
    classified_not_dnf = df["status"].isin(["Finished", "Lapped"]) | df["status"].str.contains(r"^\+", regex=True, na=False)
    df["dnf"] = ~classified_not_dnf

    df = df.merge(race_pace, left_on="driver", right_index=True, how="left")
    df = df.merge(stops, left_on="driver", right_index=True, how="left")
    df = df.merge(starting_compound, left_on="driver", right_index=True, how="left")

    quali = _quali_features(season, round_number)
    df = df.merge(quali, left_on="driver", right_on="Abbreviation", how="left").drop(columns=["Abbreviation"])

    practice = _practice_pace(season, round_number)
    df = df.merge(practice, left_on="driver", right_on="Driver", how="left").drop(columns=["Driver"])

    weather = _weather_features(race)
    for k, v in weather.items():
        df[k] = v

    df["season"] = season
    df["round"] = round_number
    df["location"] = location
    df["race_date"] = pd.Timestamp(race_date)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true", help="re-pull races even if cached")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        sched = event_schedule(season)
        for _, row in sched.iterrows():
            round_number = int(row["RoundNumber"])
            location = row["Location"]
            out_path = RAW_DIR / f"{season}_{round_number:02d}_{location}.parquet"
            if out_path.exists() and not args.force:
                log.info("Skipping cached %s", out_path.name)
                continue
            try:
                df = ingest_race(season, round_number, location, row["EventDate"])
            except Exception as exc:
                log.warning("  skipping %s round %s, unexpected error: %s", season, round_number, exc)
                continue
            if df is not None:
                df.to_parquet(out_path, index=False)
                log.info("  wrote %s (%d rows)", out_path.name, len(df))


if __name__ == "__main__":
    main()
