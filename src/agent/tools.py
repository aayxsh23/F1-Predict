"""The data the chat router answers from. Every one of these is a plain
Python function returning a plain dict (or list) built from real numbers --
cached forecasts and backtests (src/api/cache.py), the odds sampler, the
strategy simulator, the regulations index and live standings. None of them
calls an LLM: see src/agent/router.py for how a question is matched to one
of these, and src/agent/chat.py for the two places (and only two) a local
model is used to turn a result into prose.
"""
import json
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from src.agent import f1_api
from src.agent.scenarios import closest_rival, remaining_rounds, title_odds, title_scenario
from src.api.cache import get_json
from src.api.enrich import head_to_head as _h2h
from src.api.enrich import with_probabilities
from src.models.catalog import CANONICAL_PRED_COLS, FEATURE_LABELS, MODEL_DIR
from src.models.probabilities import load_calibration
from src.rag.corpus import circuit_summary, search
from src.strategy.model import simulate

Target = Literal["finish_position", "qualifying", "quali_delta", "race_time"]
TARGET_MEANING = {
    "finish_position": "finishing position (lower is better)",
    "qualifying": "qualifying gap to pole as % of the pole lap (lower is better)",
    "quali_delta": "places gained from the grid (higher is better)",
    "race_time": "gap to the winner as % of race time (lower is better)",
}


def pct(p) -> float | None:
    return None if p is None else round(float(p), 4)


def forecast(season: int | None = None, round: int | None = None) -> dict:
    """The model's forecast for one race: every driver's predicted finishing
    order, win/podium/points chances, likely finishing range, retirement
    risk, qualifying prediction (lap time, gap, pole/Q3 chances) and gap to
    the winner, plus race length and safety-car chance. season/round=None
    means the current race weekend. Raises for a past race with no stored
    forecast -- use past_race_prediction for those."""
    payload = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    p = with_probabilities(payload)
    race = p.get("race", {})
    drivers = []
    for rank, d in enumerate(p["drivers"], 1):
        pr = d.get("probabilities", {})
        q = d.get("quali_odds", {})
        drivers.append({
            "predicted_rank": rank, "driver": d["driver"], "team": d["team"], "grid": d.get("grid_position"),
            "win": pct(pr.get("win")), "podium": pct(pr.get("podium")), "points": pct(pr.get("top10")),
            "likely_range": d.get("position_band"), "retire_risk": pct(d.get("retire_risk")),
            "beats_teammate": pct(d.get("beats_teammate")),
            "predicted_quali_lap_s": d.get("predicted_quali_lap_s"), "predicted_quali_gap_s": d.get("predicted_qualifying_gap"),
            "actual_quali_gap_s": d.get("quali_gap_to_pole"), "pole": pct(q.get("pole")), "q3": pct(q.get("q3")),
            "out_in_q1": pct(q.get("q1_out")), "gap_to_winner_s": d.get("predicted_race_time_gap"),
        })
    return {
        "season": p["season"], "round": p["round"], "race": p.get("event_name") or p["location"], "circuit": p["location"],
        "race_start_utc": p.get("race_start_utc"), "forecast_as_of": p.get("session_label") or p.get("stage"),
        "generated_at": p.get("generated_at"), "laps": race.get("laps"),
        "expected_race_duration_min": None if race.get("expected_duration_s") is None else round(race["expected_duration_s"] / 60, 1),
        "predicted_pole_lap_s": race.get("pole_time_estimate_s"), "safety_car_chance": pct(race.get("safety_car_probability")),
        "drivers": drivers,
    }


def driver_code(payload: dict, driver: str) -> str:
    d = driver.strip().upper()
    codes = [x["driver"] for x in payload["drivers"]]
    if d in codes:
        return d
    raise ValueError(f"unknown driver '{driver}'; this race has: {', '.join(codes)}")


def explain_prediction(driver: str, target: Target = "finish_position", season: int | None = None, round: int | None = None) -> dict:
    """Why the model predicts what it does for one driver: the inputs that
    pushed the prediction up or down most (SHAP contributions, in the
    target's own units), with each input's value. driver is the 3-letter code."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    code = driver_code(p, driver)
    d = next(x for x in p["drivers"] if x["driver"] == code)
    exp = d.get("explanations", {}).get(target)
    if exp is None:
        raise ValueError("this forecast has no stored breakdown")
    return {
        "season": p["season"], "round": p["round"], "circuit": p["location"],
        "driver": code, "team": d["team"], "target": target, "prediction_of": TARGET_MEANING[target],
        "predicted": exp["predicted_value"], "average_prediction": exp["base_value"],
        "contributions": [{"feature": c["feature"], "input": FEATURE_LABELS.get(c["feature"], c["feature"]),
                           "value": c["value"], "effect": c["shap"]} for c in exp["top_contributions"]],
    }


def head_to_head(driver_a: str, driver_b: str, season: int | None = None, round: int | None = None) -> dict:
    """Chance driver_a finishes ahead of driver_b in a race, from 10,000 simulated races."""
    payload = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    p = with_probabilities(payload)
    a, b = driver_code(p, driver_a), driver_code(p, driver_b)
    chance = _h2h(p, a, b)
    exp = {d["driver"]: d.get("expected_position") for d in p["drivers"]}
    return {"season": p["season"], "round": p["round"], "driver_a": a, "driver_b": b,
            "a_ahead_of_b": pct(chance), "expected_finish_a": exp.get(a), "expected_finish_b": exp.get(b)}


def championship_state() -> dict:
    """Standings, remaining rounds and simulated title odds (shared with the API)."""
    standings = f1_api.get_driver_standings()
    remaining = remaining_rounds(f1_api.get_season_schedule(), f1_api.get_current_round())
    forecast_payload = with_probabilities(get_json("latest.json"))
    strength = {d["driver"]: d["predicted_finish_position"] for d in forecast_payload["drivers"]}
    worst = max(strength.values()) + 2
    retire = [d.get("retire_risk") or 0.1 for d in forecast_payload["drivers"]]
    odds = title_odds(
        [s["points"] for s in standings], [strength.get(s["code"], worst) for s in standings], remaining,
        tau=load_calibration()["race"]["pre_weekend"]["tau_field"], p_dnf=float(np.mean(retire)),
        groups=[s["team"] for s in standings],
    )
    return {"standings": standings, "remaining": remaining, "odds": odds}


def championship(kind: Literal["drivers", "constructors"] = "drivers") -> dict:
    """Live championship standings (points, wins) plus each contender's
    chance of winning the title, from simulating the rest of the season
    5,000 times with every driver's current forecast form."""
    c = championship_state()
    left = {"races": len(c["remaining"]), "sprints": sum(r["is_sprint"] for r in c["remaining"])}
    if kind == "constructors":
        teams = f1_api.get_constructor_standings()
        return {"remaining": left, "constructors": [
            {**t, "title_chance": pct(c["odds"]["team_champion"].get(t["name"]))} for t in teams]}
    return {"remaining": left, "drivers": [
        {"position": s["position"], "driver": s["code"], "name": f"{s['given_name']} {s['family_name']}", "team": s["team"],
         "points": s["points"], "wins": s["wins"], "title_chance": pct(c["odds"]["champion"][i]),
         "expected_final_points": round(float(c["odds"]["expected_points"][i]))}
        for i, s in enumerate(c["standings"])]}


def title_scenario_for(driver: str) -> dict:
    """What one driver needs to win the championship: points gap to their
    closest rival, points still available, whether they're mathematically
    alive, points needed to draw level if the rival scored nothing more, and
    their simulated title chance. 3-letter code or surname."""
    c = championship_state()
    standings = c["standings"]
    q = driver.strip().lower()
    i = next((k for k, s in enumerate(standings) if q in (s["code"].lower(), s["family_name"].lower())), None)
    if i is None:
        raise ValueError(f"no driver '{driver}' in the standings")
    me = standings[i]
    rival = closest_rival(standings, me["code"])
    return {"driver": me["code"], "driver_name": f"{me['given_name']} {me['family_name']}", "rival": rival["code"],
            "rival_name": f"{rival['given_name']} {rival['family_name']}",
            **title_scenario(me["points"], rival["points"], c["remaining"]), "title_chance": pct(c["odds"]["champion"][i])}


def race_strategy(safety_car_lap: int | None = None, season: int | None = None, round: int | None = None) -> dict:
    """Tyre strategy for a race: the fastest plans (compounds, pit laps, pit
    windows), how much slower each alternative is, and each plan's chance of
    being fastest once random safety cars are simulated. Pass
    safety_car_lap to see how a safety car on that lap changes the best plan."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    laps = p.get("race", {}).get("laps") or 57
    return {"season": p["season"], "round": p["round"], **simulate(p["location"], int(laps), sc_lap=safety_car_lap)}


def forecast_timeline(driver: str | None = None, season: int | None = None, round: int | None = None) -> list[dict]:
    """How the forecast changed through the weekend (before practice, after
    each practice, after qualifying): predicted finishing position per
    driver at each point. Pass a driver code to follow one driver."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    timeline = get_json(f"timeline/{p['season']}_{p['round']}.json")
    code = driver_code(p, driver) if driver else None
    return [{"when": s["label"], "drivers": [
        {"driver": d["driver"], "predicted_finish": d["predicted_finish_position"]}
        for d in s["drivers"] if code is None or d["driver"] == code]} for s in timeline]


def search_rules(query: str, k: int = 5) -> list[dict]:
    """Search the 2026 FIA Sporting Regulations, penalty guidelines, driving
    standards and real steward decisions. Use specific keywords ("unsafe
    release", "grid penalty power unit", "safety car restart"). Each hit
    carries its document and article number, so it can be cited exactly."""
    return search(query, k=k, doc_types={"regulation", "steward_decision"})


def circuit_guide(circuit: str) -> str:
    """What a circuit is like: layout, overtaking, tyre wear, safety cars,
    weather, history. circuit is the location name (Monaco, Monza, Baku,
    Silverstone, Marina Bay, Yas Island, ...)."""
    text = circuit_summary(circuit)
    if not text:
        hits = search(circuit, k=1, doc_types={"circuit_summary"})
        text = hits[0]["text"] if hits else ""
    return text


def driver_history(driver: str, circuit: str | None = None, last_n: int = 10) -> list[dict]:
    """A driver's recent real results since 2022 (grid, qualifying position,
    finish, status, points), optionally only at one circuit."""
    rows = [r for r in get_json("backtest/results.json") if r["driver"] == driver.strip().upper()]
    if circuit:
        rows = [r for r in rows if circuit.lower() in r["location"].lower()]
    return rows[-last_n:]


def past_race_prediction(season: int, round: int) -> dict:
    """For a race that has already happened: what the model predicted
    beforehand (from a model trained only on earlier races) next to the real
    result, per driver."""
    b = get_json(f"backtest/{season}_{round}.json")
    rows = sorted(b["drivers"], key=lambda d: d["finish_position"]["actual"] or 99)
    return {"season": season, "round": round, "circuit": b["location"], "method": b.get("method"), "drivers": [
        {"driver": d["driver"], "actual_finish": d["finish_position"]["actual"], "predicted_finish": d["finish_position"]["predicted"],
         "actual_places_gained": d["quali_delta"]["actual"], "predicted_places_gained": d["quali_delta"]["predicted"]}
        for d in rows]}


def model_track_record() -> dict:
    """How accurate the models have been on races they had never seen, per
    season and per weekend stage, against simple baselines (e.g. "finish
    where you start"), and how often they called the winner and pole."""
    summary = get_json("backtest/summary.json")
    held_out = {t: json.loads((MODEL_DIR / f"{t}_metrics.json").read_text()) for t in CANONICAL_PRED_COLS}
    return {"by_season": summary, "held_out": {
        t: {"what": m["evaluation"], "unit": m["unit"], "baseline": m["baseline"], "error_by_stage": m["stages"]}
        for t, m in held_out.items()}}


def season_schedule(season: int | None = None) -> dict:
    """The season calendar: rounds, venues, race start times (UTC), sprint
    weekends, and which race is next."""
    season = season or datetime.now(timezone.utc).year
    sched = get_json(f"schedule/{season}.json")
    now = datetime.now(timezone.utc).isoformat()
    upcoming = [r for r in sched if (r.get("RaceStartUtc") or "") > now[:19]]
    return {"season": season, "next_round": upcoming[0]["RoundNumber"] if upcoming else None, "rounds": [
        {"round": r["RoundNumber"], "event": r["EventName"], "location": r["Location"], "race_start_utc": r.get("RaceStartUtc"),
         "sprint_weekend": "sprint" in (r.get("EventFormat") or "")} for r in sched]}
