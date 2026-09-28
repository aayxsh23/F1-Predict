"""Turn the models' relative predictions (gap to pole in %, gap to the winner
in %) into absolute times a fan can read: a qualifying lap of 1:41.2, a race
of 1h 32m. Plus the chance of a safety car.

These are historical ratios at each circuit, not trained models:
  pole lap      = fastest practice lap x (pole / fastest practice lap) at this track
  race duration = laps x fastest practice lap x (winner time / (laps x fastest practice lap))
A circuit with no history (a new venue) falls back to the median ratio
across all circuits. Without any practice yet, last year's figure here is used.
"""
import numpy as np
import pandas as pd

from src.features.circuit_reference import LOCATION_ALIASES, load as load_circuits

RACE_DISTANCE_KM = 305  # FIA minimum race distance; Monaco is the exception, handled by its own lap history


def _races(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per historical race with its race-level facts."""
    cols = ["season", "round", "location", "practice_fastest_s", "quali_pole_s", "race_winner_time_s", "race_laps", "sc_laps"]
    races = raw.reindex(columns=cols).drop_duplicates(["season", "round"]).copy()
    races["location"] = races["location"].replace(LOCATION_ALIASES)
    return races


def _at(races: pd.DataFrame, location: str) -> pd.DataFrame:
    return races[races["location"] == LOCATION_ALIASES.get(location, location)]


def _ratio(races: pd.DataFrame, location: str, num: pd.Series, den: pd.Series) -> float:
    r = (num / den).replace([np.inf, -np.inf], np.nan)
    here = r[races["location"] == LOCATION_ALIASES.get(location, location)].dropna()
    return float(here.median()) if len(here) else float(r.median())


def pole_time(raw: pd.DataFrame, location: str, practice_fastest_s: float) -> float:
    races = _races(raw)
    if pd.isna(practice_fastest_s):
        past = _at(races, location)["quali_pole_s"].dropna()
        return float(past.iloc[-1]) if len(past) else np.nan
    return practice_fastest_s * _ratio(races, location, races["quali_pole_s"], races["practice_fastest_s"])


def race_laps(raw: pd.DataFrame, location: str, season: int) -> int:
    past = _at(_races(raw), location)["race_laps"].dropna()
    if len(past):
        return int(past.mode().iloc[-1])
    c = load_circuits()
    length = c.loc[c["location"] == LOCATION_ALIASES.get(location, location), "track_length_km"]
    return int(np.ceil(RACE_DISTANCE_KM / length.iloc[0])) if len(length) else 57


def race_duration(raw: pd.DataFrame, location: str, season: int, practice_fastest_s: float) -> float:
    """Expected winner's race time in seconds (pit stops and typical safety
    cars included, because the historical ratio includes them)."""
    races = _races(raw)
    laps = race_laps(raw, location, season)
    if pd.isna(practice_fastest_s):
        past = _at(races, location)["race_winner_time_s"].dropna()
        return float(past.median()) if len(past) else np.nan
    ratio = _ratio(races, location, races["race_winner_time_s"], races["race_laps"] * races["practice_fastest_s"])
    return laps * practice_fastest_s * ratio


def safety_car_probability(raw: pd.DataFrame, location: str) -> float:
    """Share of past races here with at least one safety car or VSC lap;
    shrunk toward the all-circuit rate when there are only a few races here."""
    races = _races(raw).dropna(subset=["sc_laps"])
    here = _at(races, location)
    overall = float((races["sc_laps"] > 0).mean())
    n = len(here)
    return (float((here["sc_laps"] > 0).sum()) + 2 * overall) / (n + 2)


if __name__ == "__main__":
    from src.features.build_dataset import load_raw

    raw = load_raw()
    for loc in ("Monaco", "Monza", "Baku", "Madrid"):
        fp = raw.loc[raw["location"].replace(LOCATION_ALIASES) == loc, "practice_fastest_s"].dropna()
        fp = float(fp.iloc[-1]) if len(fp) else np.nan
        print(f"{loc:8s} practice {fp:7.2f}  pole est {pole_time(raw, loc, fp):7.2f}  "
              f"race {race_duration(raw, loc, 2026, fp) / 60:6.1f} min over {race_laps(raw, loc, 2026)} laps  "
              f"SC {safety_car_probability(raw, loc):.0%}")
