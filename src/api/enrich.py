"""Response-time additions to the cached prediction/backtest JSON: car numbers
(from the committed raw race cache) and, for live forecasts, win/podium/top-10
probabilities plus a P10-P90 finishing band sampled from the saved
calibration in src/models/saved/pl_calibration.json (see models/probabilities.py).

Done here rather than in refresh_job.py so it applies to every payload already
sitting in data/predictions/ without waiting on a CI re-run, and so the
sampling constants can be recalibrated without regenerating history. It costs
~20 ms per race, memoised per (season, round, generated_at)."""
from functools import lru_cache

import pandas as pd

from src.features.build_dataset import RAW_DIR
from src.models.probabilities import race_probabilities

_probability_cache: dict[tuple, list[dict]] = {}


@lru_cache(maxsize=1)
def _car_numbers() -> dict[tuple[int, str], int]:
    """(season, driver code) -> car number, read straight from the raw cache."""
    frames = [pd.read_parquet(f, columns=["season", "driver", "driver_number"]) for f in sorted(RAW_DIR.glob("*.parquet"))]
    if not frames:
        return {}
    latest = pd.concat(frames).drop_duplicates(["season", "driver"], keep="last")
    return {
        (int(season), driver): int(number)
        for season, driver, number in zip(latest["season"], latest["driver"], latest["driver_number"])
        if str(number).isdigit()
    }


def with_car_numbers(payload: dict) -> dict:
    numbers = _car_numbers()
    drivers = [{**d, "driver_number": numbers.get((payload["season"], d["driver"]))} for d in payload["drivers"]]
    return {**payload, "drivers": drivers}


def _sampled(payload: dict) -> list[dict] | None:
    drivers = payload["drivers"]
    predicted = [d.get("predicted_finish_position") for d in drivers]
    if len(drivers) < 3 or any(p is None for p in predicted):
        return None
    rows = [d.get("feature_row", {}) for d in drivers]
    nan = float("nan")
    pr = race_probabilities(
        predicted,
        [r.get("driver_dnf_rate") if r.get("driver_dnf_rate") is not None else nan for r in rows],
        [r.get("team_reliability") if r.get("team_reliability") is not None else nan for r in rows],
        seed=payload["season"] * 100 + payload["round"],  # same race always samples the same draws
    )
    return [
        {
            "probabilities": {"win": round(float(pr["win"][i]), 4), "podium": round(float(pr["podium"][i]), 4), "top10": round(float(pr["top10"][i]), 4)},
            "position_band": [int(pr["band"][i][0]), int(pr["band"][i][1])],
            "expected_position": round(float(pr["expected_position"][i]), 2),
        }
        for i in range(len(drivers))
    ]


def with_probabilities(payload: dict) -> dict:
    payload = with_car_numbers(payload)
    key = (payload["season"], payload["round"], payload.get("generated_at"))
    if key not in _probability_cache:
        if len(_probability_cache) >= 16:
            _probability_cache.clear()
        _probability_cache[key] = _sampled(payload)
    extra = _probability_cache[key]
    if extra is None:
        return payload
    return {**payload, "drivers": [{**d, **e} for d, e in zip(payload["drivers"], extra)]}
