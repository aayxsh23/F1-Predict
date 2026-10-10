"""Pull one row per driver per race from FastF1 to
data/raw/races/<season>_<round>_<location>.parquet, plus that race's lap-by-lap
stint data to data/raw/laps/ (the strategy model's input).

Per driver: grid, qualifying times, practice pace (best lap and long run),
outcome and time gap to the winner, and on sprint weekends the Sprint
Qualifying gap, Sprint result and Sprint pace. Per race, repeated on every row: winner's
race duration, lap count, safety-car laps, pole time, fastest practice lap, and
the Open-Meteo forecast for race start.

Resumable: re-running skips races that already have a cached raw file. Races
that haven't started yet are always skipped, so --force never chases future
sessions into FastF1's rate limit. `--sprint-backfill` adds the sprint columns
to raw files written before they existed, without re-pulling the races.

Each run also records every listed season's pre-season testing pace once the
test has run (data/raw/testing/<season>.parquet): per team, the best lap of
the season's last test as % over the fastest team.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fastf1_client import enable_cache, event_schedule, load_session
from src.data.weather import race_forecast
from src.features.build_dataset import SPRINT_COLS
from src.features.circuit_reference import coords

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "races"
LAPS_DIR = RAW_DIR.parent / "laps"
TESTING_DIR = RAW_DIR.parent / "testing"
DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
LONG_RUN_MIN_LAPS = 6
# FastF1 EventFormat of a sprint weekend where Sprint Qualifying and the Sprint
# stand on their own (2023 on): one practice session, and neither sprint
# session sets the Grand Prix grid. 2022's format ("sprint": Friday's
# qualifying set the Sprint grid, the Sprint set Sunday's) is left out.
SPRINT_FORMATS = {"sprint_shootout": "SS", "sprint_qualifying": "SQ"}
SPRINT_QUALI_MAX_GAP = 7.0  # % of the best lap: the 107% rule

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


def _race_pace(laps: pd.DataFrame) -> pd.Series:
    """Mean green-flag lap per driver, as % over the fastest such lap."""
    clean = _clean_race_laps(laps)
    if clean is None or clean.empty:
        return pd.Series(dtype=float)
    fastest = clean["LapTime"].min()
    return clean.groupby("Driver")["LapTime"].mean().sub(fastest).div(fastest).mul(100)


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
    (the % travels between a 70 s Monaco lap and a 104 s Spa lap), pole time.
    Laps and race-control messages are loaded because for a session that has
    only just run, Jolpica (Ergast) has no result yet and FastF1 derives the
    classification and Q1/Q2/Q3 times from them; without them a live forecast
    sat at "after the Sprint" for hours after qualifying (2026-10-10)."""
    res = load_session(season, round_number, "Q", laps=True, weather=False, messages=True).results.copy()
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


def sprint_features(season: int, round_number: int, event_format: str) -> pd.DataFrame:
    """Per driver on a sprint weekend: Sprint Qualifying gap to its pole (% of
    the pole lap), Sprint finishing position, and Sprint green-flag pace (% over
    the fastest lap). A session that hasn't run yet leaves its columns NaN."""
    out = pd.DataFrame({"driver": pd.Series(dtype=object)})
    try:
        # FastF1 has no segment times for Sprint Qualifying (Ergast never covered
        # it), so read each driver's best valid lap off the laps instead
        laps = load_session(season, round_number, SPRINT_FORMATS[event_format], laps=True, messages=True).laps
        if "Deleted" in laps:  # track-limits laps don't count, as in real qualifying
            laps = laps[~laps["Deleted"].fillna(False).astype(bool)]
        best = _secs(laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min())
        if not best.empty:
            gap = (best / best.min() - 1) * 100
            # outside 107% is no representative lap (a crash, an out-lap only), not pace
            sq = gap.where(gap <= SPRINT_QUALI_MAX_GAP).rename("sprint_quali_gap_pct").rename_axis("driver").reset_index()
            out = out.merge(sq, on="driver", how="outer")
    except Exception as exc:
        log.info("  no sprint qualifying (%s)", exc)
    try:
        sprint = load_session(season, round_number, "S", laps=True, messages=True)
        res = sprint.results[["Abbreviation", "Position"]].rename(columns={"Abbreviation": "driver", "Position": "sprint_finish_position"})
        res = res.merge(_race_pace(sprint.laps).rename("sprint_race_pace_pct"), left_on="driver", right_index=True, how="left")
        out = out.merge(res[res["sprint_finish_position"].notna()], on="driver", how="outer")
    except Exception as exc:
        log.info("  no sprint (%s)", exc)
    return out.reindex(columns=["driver", *SPRINT_COLS]).astype({c: float for c in SPRINT_COLS})


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


def ingest_race(season: int, round_number: int, location: str, race_date, start_utc,
                event_format: str = "conventional") -> tuple[pd.DataFrame, pd.DataFrame] | None:
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
        race_pace = _race_pace(laps).rename("race_pace_pct")
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

    df = add_sprint(df, season, round_number, event_format)

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


def add_sprint(df: pd.DataFrame, season: int, round_number: int, event_format: str) -> pd.DataFrame:
    """df with `sprint_weekend` and the SPRINT_COLS (NaN on other weekends)."""
    df = df.drop(columns=["sprint_weekend", *SPRINT_COLS], errors="ignore").assign(sprint_weekend=event_format in SPRINT_FORMATS)
    if event_format not in SPRINT_FORMATS:
        return df.assign(**{c: np.nan for c in SPRINT_COLS})
    return df.merge(sprint_features(season, round_number, event_format), on="driver", how="left")


def sprint_backfill(seasons: list[int]) -> None:
    """Add the sprint columns to raw files written before they existed."""
    for season in seasons:
        for _, row in event_schedule(season).iterrows():
            path = RAW_DIR / f"{season}_{int(row['RoundNumber']):02d}_{row['Location']}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if "sprint_weekend" in df.columns:
                continue
            log.info("Sprint columns for %s", path.name)
            df = add_sprint(df, season, int(row["RoundNumber"]), row["EventFormat"])
            if df["sprint_weekend"].any() and df["sprint_finish_position"].isna().all():
                log.warning("  sprint weekend but no sprint result loaded, leaving %s for a later run", path.name)
                continue
            df.to_parquet(path, index=False)


def testing_pace(season: int) -> pd.DataFrame | None:
    """Per team: best lap across the days of the season's last pre-season
    test, as % over the fastest team, and that test's last day. None if FastF1
    has no test for the season (it starts in 2020)."""
    import fastf1
    from fastf1.exceptions import RateLimitExceededError

    from src.models.catalog import TEAM_LINEAGE

    enable_cache()
    for test in (3, 2, 1):
        try:
            fastf1.get_testing_event(season, test)
        except Exception:
            continue
        frames, last_day = [], None
        for day in (1, 2, 3):
            try:
                s = fastf1.get_testing_session(season, test, day)
                for _ in range(400):  # same courtesy-limit patience as load_session
                    try:
                        s.load(laps=True, telemetry=False, weather=False, messages=False)
                        break
                    except RateLimitExceededError:
                        import time
                        time.sleep(9)
                laps = s.laps
            except Exception as exc:
                log.info("  no testing %s test %s day %s (%s)", season, test, day, exc)
                continue
            if laps is not None and not laps.empty:
                frames.append(laps)
                last_day = pd.Timestamp(s.date)
        if frames:
            laps = pd.concat(frames, ignore_index=True).dropna(subset=["LapTime"])
            laps = laps[laps["Team"].fillna("").str.strip() != ""]
            best = _secs(laps.groupby("Team")["LapTime"].min())
            out = pd.DataFrame({"season": season, "team": best.index.map(lambda t: TEAM_LINEAGE.get(t, t)),
                                "testing_gap_pct": ((best / best.min() - 1) * 100).to_numpy(), "date": last_day.normalize()})
            return out
    return None


def testing_backfill(seasons: list[int]) -> None:
    TESTING_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize(None)
    for season in seasons:
        path = TESTING_DIR / f"{season}.parquet"
        first_race = event_schedule(season)["Session1DateUtc"].min()
        if path.exists() or not first_race < now:
            continue
        pace = testing_pace(season)
        if pace is not None:
            pace.to_parquet(path, index=False)
            log.info("testing %s: %s", season, pace.sort_values("testing_gap_pct")[["team", "testing_gap_pct"]].round(2).values.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true", help="re-pull races even if cached")
    parser.add_argument("--sprint-backfill", action="store_true", help="add sprint columns to existing raw files")
    parser.add_argument("--testing-only", action="store_true", help="only record pre-season testing pace")
    args = parser.parse_args()
    if args.sprint_backfill:
        return sprint_backfill(args.seasons)
    if args.testing_only:
        return testing_backfill(args.seasons)

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
                out = ingest_race(season, round_number, location, row["EventDate"], start, row["EventFormat"])
            except Exception as exc:
                log.warning("  skipping %s round %s, unexpected error: %s", season, round_number, exc)
                continue
            if out is not None:
                df, lap_table = out
                df.to_parquet(RAW_DIR / name, index=False)
                if not lap_table.empty:
                    lap_table.to_parquet(LAPS_DIR / name, index=False)
                log.info("  wrote %s (%d rows, %d laps)", name, len(df), len(lap_table))
    testing_backfill(args.seasons)


if __name__ == "__main__":
    main()
