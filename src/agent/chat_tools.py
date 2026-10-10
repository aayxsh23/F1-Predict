"""Chat-model-sized versions of tools whose full output is too big for a small
model's context window. The deterministic router keeps using the full
versions in tools.py; these are what the model sees (registered in
toolspec.py under the same tool names)."""
from src.agent import tools

_COMPACT_KEYS = ("predicted_rank", "driver", "name", "team", "grid", "win", "podium", "points", "likely_range", "pole")


def forecast(season: int | None = None, round: int | None = None, driver: str | None = None) -> dict:
    """The forecast for a race weekend: the predicted top 10 with win, podium,
    points and pole chances, plus race facts. Pass `driver` (3-letter code) for
    that one driver's full forecast: qualifying lap and gap, likely finishing
    range, retirement risk, beating the teammate, gap to the winner."""
    full = tools.forecast(season, round)
    head = {k: v for k, v in full.items() if k != "drivers"}
    if driver:
        code = driver.strip().upper()
        match = next((d for d in full["drivers"] if d["driver"] == code or (d.get("name") or "").lower().endswith(driver.strip().lower())), None)
        if not match:
            raise ValueError(f"no driver '{driver}' in this forecast; drivers: {', '.join(d['driver'] for d in full['drivers'])}")
        return {**head, "field_size": len(full["drivers"]), "driver": match}
    return {**head, "field_size": len(full["drivers"]),
            "top10": [{k: d.get(k) for k in _COMPACT_KEYS} for d in full["drivers"][:10]],
            "rest_of_field": [d["driver"] for d in full["drivers"][10:]],
            "note": "pass driver=<code> for one driver's full forecast"}


_PLAIN_NAMES = {"finish_position": "finishing_position", "qualifying": "qualifying_gap", "quali_delta": "places_gained", "race_time": "gap_to_winner"}


def model_track_record() -> dict:
    """How accurate the prediction models have been on races they never saw:
    average error per prediction against a simple guess, before the weekend and
    once the grid is known (qualifying: after practice), plus this season."""
    full = tools.model_track_record()
    out = {}
    for target, m in full["held_out"].items():
        stages = m["error_by_stage"]
        late = stages.get("post_quali") or stages.get("post_practice")
        seasons = full["by_season"].get(target, {}).get("by_season", {})
        latest = max(seasons) if seasons else None
        out[_PLAIN_NAMES.get(target, target)] = {
            "unit": m["unit"], "simple_guess": m["baseline"],
            "average_error_once_grid_known" if "post_quali" in stages else "average_error_after_practice": late["mae"],
            "simple_guess_error": late["baseline_mae"], "average_error_before_the_weekend": stages["pre_weekend"]["mae"],
            **({f"{latest}_season": seasons[latest]} if latest else {}),
        }
    any_m = next(iter(full["held_out"].values()))
    return {"how_measured": any_m["what"], "lower_is_better": True, **out}
