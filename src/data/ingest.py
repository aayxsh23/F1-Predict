"""Pull one row per driver per race from FastF1 to
data/raw/races/<season>_<round>_<location>.parquet, plus that race's lap-by-lap
stint data to data/raw/laps/ (the strategy model's input).

Per driver: grid, qualifying times, practice pace (best lap and long run),
outcome and time gap to the winner. Per race, repeated on every row: winner's
race duration, lap count, safety-car laps, pole time, fastest practice lap, and
the Open-Meteo forecast for race start.

Resumable: re-running skips races that already have a cached raw file. Races
that haven't started yet are always skipped, so --force never chases future
sessions into FastF1's rate limit.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fastf1_client import event_schedule, load_session
from src.data.weather import race_forecast
from src.features.circuit_reference import coords

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "races"
LAPS_DIR = RAW_DIR.parent / "laps"
DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
LONG_RUN_MIN_LAPS = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _secs(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()


def _clean_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag, non-in/out laps only — used for race-pace stats."""
    if laps is None or laps.empty:
        return laps
    mask = (
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["TrackStatus"].astype(str).eq("1")
        & laps["LapTime"].notna()
    )
    return laps[mask]


def _long_runs(laps: pd.DataFrame) -> pd.Series:
    """Each driver's best race-simulation run: median lap of a stint with at
    least LONG_RUN_MIN_LAPS laps within 107% of that stint's fastest lap (so
    cool-down laps don't count). A one-lap qualifying sim says little about
    Sunday; a fuelled run on one set of tyres says more."""
    if (laps["session"] == "FP2").any():  # FP2 is where teams run race simulations
        laps = laps[laps["session"] == "FP2"]
    ok = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna() & laps["LapTime"].notna()].copy()
    ok["t"] = _secs(ok["LapTime"])
    ok = ok[ok["t"] <= ok.groupby(["session", "Driver", "Stint"])["t"].transform("min") * 1.07]
    runs = ok.groupby(["session", "Driver", "Stint"])["t"].agg(["median", "count"])
    return runs[runs["count"] >= LONG_RUN_MIN_LAPS]["median"].groupby(level="Driver").min()


def practice_features(season: int, round_number: int) -> tuple[pd.DataFrame, float, list[str]]:
    """Per driver across whichever FP sessions exist: best lap and long-run
    pace, each as % over the field's best. Also returns the fastest practice
    lap in seconds and which sessions were found."""
    frames, found = [], []
    for code in ("FP1", "FP2", "FP3"):
        try:
            laps = load_session(season, round_number, code, laps=True, weather=False).laps
        except Exception as exc:  # sprint weekends lack FP2/FP3; future sessions don't exist yet
            log.info("  no %s (%s)", code, exc)
            continue
        if laps is not None and not laps.empty:
            frames.append(laps.assign(session=code))
            found.append(code)
    empty = pd.DataFrame(columns=["Driver", "practice_pace", "practice_long_run_pace"])
    if not frames:
        return empty, np.nan, found
    laps = pd.concat(frames, ignore_index=True)
    best = _secs(laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min())
    if best.empty:
        return empty, np.nan, found
    runs = _long_runs(laps)
    out = pd.DataFrame({"practice_pace": (best / best.min() - 1) * 100})
    out["practice_long_run_pace"] = (runs / runs.min() - 1) * 100 if not runs.empty else np.nan
    return out.rename_axis("Driver").reset_index(), float(best.min()), found


def quali_features(season: int, round_number: int) -> pd.DataFrame:
    """Q1/Q2/Q3 times (s), gap to pole in seconds and as % of the pole lap
    (the % travels between a 70 s Monaco lap and a 104 s Spa lap), pole time."""
    res = load_session(season, round_number, "Q", laps=False, weather=False).results.copy()
    for col in ("Q1", "Q2", "Q3"):
        if col not in res.columns:
            res[col] = pd.NaT
    best = _secs(res[["Q1", "Q2", "Q3"]].min(axis=1))
    pole = best.min()
    return pd.DataFrame({
        "Abbreviation": res["Abbreviation"],
        "quali_position": res["Position"],
        "quali_gap_to_pole": best - pole,
        "quali_gap_pct": (best / pole - 1) * 100,
        "quali_pole_s": pole,
        "q1_time": _secs(res["Q1"]), "q2_time": _secs(res["Q2"]), "q3_time": _secs(res["Q3"]),
    })


def _race_gaps(results: pd.DataFrame, laps: pd.DataFrame) -> tuple[pd.Series, float, int]:
    """Gap to the winner in seconds for every classified car, laps down
    included: a car k laps down is (its finish time - winner's finish time)
    + k average winner laps behind. FastF1's own `Time` for a lapped car is
    only its gap within the final lap (Bottas, Melbourne 2024: "+42 s" while a
    lap down), so it can't be used as-is."""
    winner = results.loc[results["Position"] == 1].iloc[0]
    duration, race_laps = winner["Time"].total_seconds(), int(winner["Laps"])
    avg_lap = duration / race_laps
    gap = pd.Series(np.nan, index=results.index)
    lead_lap = results["Laps"] == race_laps
    gap[lead_lap] = _secs(results.loc[lead_lap, "Time"])
    gap[results["Position"] == 1] = 0.0
    if laps is not None and not laps.empty:
        finish = laps.groupby("Driver")["Time"].max()
        t_w = finish.get(winner["Abbreviation"])
        for i, r in results[~lead_lap & results["Laps"].notna()].iterrows():
            if t_w is not None and r["Abbreviation"] in finish.index:
                gap[i] = (finish[r["Abbreviation"]] - t_w).total_seconds() + (race_laps - r["Laps"]) * avg_lap
    # a race ending under a safety car can make a lapped car's estimate smaller
    # than a lead-lap car's; never let a gap contradict the official order
    order = results["Position"].sort_values().index
    gap[order] = gap[order].cummax().where(gap[order].notna())
    return gap, duration, race_laps


def _lap_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Lap-by-lap stint record for the strategy model."""
    return pd.DataFrame({
        "driver": laps["Driver"], "team": laps["Team"], "lap": laps["LapNumber"],
        "lap_time_s": _secs(laps["LapTime"]), "compound": laps["Compound"], "tyre_life": laps["TyreLife"],
        "stint": laps["Stint"], "pit_in": laps["PitInTime"].notna(), "pit_out": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"].astype(str), "position": laps["Position"],
    })


def race_start(event: pd.Series):
    """UTC start of the Grand Prix itself, from the schedule's session slots."""
    for n in range(5, 0, -1):
        if event.get(f"Session{n}") == "Race":
            return event.get(f"Session{n}DateUtc")
    return pd.NaT


def ingest_race(season: int, round_number: int, location: str, race_date, start_utc) -> tuple[pd.DataFrame, pd.DataFrame] | None:
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
        race_pace = (clean.groupby("Driver")["LapTime"].mean().sub(fastest).div(fastest).mul(100)).rename("race_pace_pct")
        stops = (laps.groupby("Driver")["Stint"].max() - 1).rename("num_pit_stops")
    else:
        race_pace = pd.Series(dtype=float, name="race_pace_pct")
        stops = pd.Series(dtype=float, name="num_pit_stops")

    starting_compound = (
        laps[laps["LapNumber"] == 1][["Driver", "Compound"]].drop_duplicates("Driver").set_index("Driver")["Compound"].rename("starting_compound")
        if laps is not None and not laps.empty
        else pd.Series(dtype=object, name="starting_compound")
    )

    gap, duration, race_laps = _race_gaps(results, laps)
    df = results[["Abbreviation", "DriverNumber", "TeamName", "GridPosition", "Position", "Points", "Status", "Laps"]].rename(columns={
        "Abbreviation": "driver", "DriverNumber": "driver_number", "TeamName": "team",
        "GridPosition": "grid_position", "Position": "finish_position",
        "Points": "points", "Status": "status", "Laps": "laps_completed",
    })
    classified_not_dnf = df["status"].isin(["Finished", "Lapped"]) | df["status"].str.contains(r"^\+", regex=True, na=False)
    df["dnf"] = ~classified_not_dnf
    # NaN (not imputed) for retirements: they have no finishing time
    df["gap_to_winner_seconds"] = gap.where(~df["dnf"])
    df["race_gap_pct"] = df["gap_to_winner_seconds"] / duration * 100

    df = df.merge(race_pace, left_on="driver", right_index=True, how="left")
    df = df.merge(stops, left_on="driver", right_index=True, how="left")
    df = df.merge(starting_compound, left_on="driver", right_index=True, how="left")

    quali = quali_features(season, round_number)
    df = df.merge(quali, left_on="driver", right_on="Abbreviation", how="left").drop(columns=["Abbreviation"])

    practice, practice_fastest, _ = practice_features(season, round_number)
    df = df.merge(practice, left_on="driver", right_on="Driver", how="left").drop(columns=["Driver"])

    # measured race weather, kept as raw data; the model uses the forecast below
    w = race.weather_data
    if w is not None and not w.empty:
        df["air_temp"], df["track_temp"] = w["AirTemp"].mean(), w["TrackTemp"].mean()
        df["rain_probability"], df["wind_speed"] = w["Rainfall"].mean(), w["WindSpeed"].mean()
        df["wet_track_probability"] = (w["Rainfall"] | (w["Humidity"] > 85)).mean()
    for k, v in race_forecast(*coords(location, season), start_utc).items():
        df[k] = v

    status = laps["TrackStatus"].astype(str) if laps is not None and not laps.empty else pd.Series(dtype=str)
    df["sc_laps"] = laps.loc[status.str.contains("[46]"), "LapNumber"].nunique() if len(status) else np.nan
    df["red_flag"] = bool(status.str.contains("5").any())
    df["race_winner_time_s"] = duration
    df["race_laps"] = race_laps
    df["practice_fastest_s"] = practice_fastest
    df["season"] = season
    df["round"] = round_number
    df["location"] = location
    df["race_date"] = pd.Timestamp(race_date)
    df["race_start_utc"] = pd.Timestamp(start_utc)

    lap_table = _lap_table(laps) if laps is not None and not laps.empty else pd.DataFrame()
    return df, lap_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true", help="re-pull races even if cached")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LAPS_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize(None)

    for season in args.seasons:
        sched = event_schedule(season)
        for _, row in sched.iterrows():
            round_number, location = int(row["RoundNumber"]), row["Location"]
            start = race_start(row)
            if pd.isna(start) or start > now:
                continue
            name = f"{season}_{round_number:02d}_{location}.parquet"
            if (RAW_DIR / name).exists() and not args.force:
                log.info("Skipping cached %s", name)
                continue
            try:
                out = ingest_race(season, round_number, location, row["EventDate"], start)
            except Exception as exc:
                log.warning("  skipping %s round %s, unexpected error: %s", season, round_number, exc)
                continue
            if out is not None:
                df, lap_table = out
                df.to_parquet(RAW_DIR / name, index=False)
                if not lap_table.empty:
                    lap_table.to_parquet(LAPS_DIR / name, index=False)
                log.info("  wrote %s (%d rows, %d laps)", name, len(df), len(lap_table))


if __name__ == "__main__":
    main()
