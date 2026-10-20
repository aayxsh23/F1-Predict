"""Odds added to a cached forecast when it is served: win / podium / top-10,
likely finishing range, retirement risk, chance of beating the teammate and,
until qualifying has happened, pole / Q3 / out-in-Q1 odds.

Done at read time rather than in refresh_job.py so every forecast already in
data/predictions/ picks up a recalibration without a CI re-run. ~30 ms per
race, memoised per (season, round, generated_at).

Circuit facts measured from F1 data (features/circuit_telemetry.py) are added
the same way, so every stored forecast shows them."""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.models.catalog import LOCATION_ALIASES
from src.models.probabilities import position_probabilities, quali_probabilities, race_probabilities

_cache: dict[tuple, dict] = {}
CIRCUITS_CSV = Path(__file__).resolve().parents[1] / "features" / "circuit_reference.csv"


def stage_of_payload(payload: dict) -> str:
    if payload.get("stage"):
        return payload["stage"]
    known = payload.get("known_sessions", {})
    return "post_quali" if known.get("qualifying") else "post_practice" if known.get("practice") else "pre_weekend"


def _num(v) -> float:
    return np.nan if v is None else float(v)


def _simulate(payload: dict) -> dict | None:
    key = (payload["season"], payload["round"], payload.get("generated_at"))
    if key in _cache:
        return _cache[key]
    drivers = payload["drivers"]
    predicted = [d.get("predicted_finish_position") for d in drivers]
    if len(drivers) < 3 or any(p is None for p in predicted):
        return None
    stage = stage_of_payload(payload)
    seed = payload["season"] * 100 + payload["round"]  # the same race always samples the same draws
    race = race_probabilities(predicted, stage=stage, teams=[d["team"] for d in drivers], seed=seed,
                              grid=[_num(d.get("grid_position")) for d in drivers])
    quali = None
    gaps = [d.get("predicted_qualifying_gap_pct") for d in drivers]
    if not payload.get("known_sessions", {}).get("qualifying") and all(g is not None for g in gaps):
        quali = quali_probabilities(gaps, stage, seed=seed, nolap_rate=[_num(d.get("quali_nolap_rate")) for d in drivers])
    if len(_cache) >= 16:
        _cache.clear()
    _cache[key] = {"race": race, "quali": quali}
    return _cache[key]


def with_probabilities(payload: dict) -> dict:
    sim = _simulate(payload)
    if sim is None:
        return payload
    race, quali = sim["race"], sim["quali"]
    dist = position_probabilities(race["samples"])
    out = []
    for i, d in enumerate(payload["drivers"]):
        extra = {
            "probabilities": {k: round(float(race[k][i]), 4) for k in ("win", "podium", "top10")},
            "position_band": [int(race["band"][i][0]), int(race["band"][i][1])],
            # the chance of each finishing place, P1 first (phase 10 item 6.4: chances, not just "Finish 3.42")
            "finish_distribution": [round(float(x), 3) for x in dist[i]],
            "expected_position": round(float(race["expected_position"][i]), 2),
            "retire_risk": round(float(race["retire"][i]), 3),
            "beats_teammate": None if np.isnan(race["beats_teammate"][i]) else round(float(race["beats_teammate"][i]), 3),
        }
        if quali is not None:
            extra["quali_odds"] = {k: round(float(quali[k][i]), 4) for k in ("pole", "q3", "q1_out")}
        out.append({**d, **extra})
    return {**payload, "drivers": out}


@lru_cache(maxsize=1)
def _circuit_rows() -> tuple[dict, ...]:
    with open(CIRCUITS_CSV, encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def circuit_facts(location: str, season: int) -> dict:
    """The measured facts of the layout raced at `location` that season; {} if unmapped."""
    loc = LOCATION_ALIASES.get(location, location)
    for r in _circuit_rows():
        if r["location"] == loc and float(r["valid_from_year"]) <= season <= float(r["valid_to_year"] or 9999):
            return {**{k: float(r[f"measured_{k}"]) if r.get(f"measured_{k}") else None
                       for k in ("flat_out_m", "braking_zones", "pit_loss_s")},
                    "measured_from": r.get("measured_from") or None}
    return {}


def with_circuit(payload: dict) -> dict:
    return {**payload, "race": {**(payload.get("race") or {}), "circuit": circuit_facts(payload["location"], payload["season"])}}


def head_to_head(payload: dict, a: str, b: str) -> float | None:
    """Chance driver `a` finishes ahead of driver `b` in this race."""
    sim = _simulate(payload)
    codes = [d["driver"] for d in payload["drivers"]]
    if sim is None or a not in codes or b not in codes:
        return None
    s = sim["race"]["samples"]
    return round(float((s[:, codes.index(a)] < s[:, codes.index(b)]).mean()), 3)
