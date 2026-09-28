"""The chat assistant's tools. Every number the assistant may state comes from
one of these: cached forecasts and backtests (src/api/cache.py), the odds
sampler, the strategy simulator, the regulations index and live standings.

Each returns compact JSON text for the model to read. search_rules also
returns its sources as an artifact, which the API forwards to the app as
clickable citations.
"""
import json
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from langchain_core.tools import tool

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


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))


def _pct(p) -> str | None:
    return None if p is None else f"{p:.0%}" if p >= 0.01 or p == 0 else "<1%"


def _forecast(season: int | None, round: int | None) -> dict:
    payload = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    return with_probabilities(payload)


def _code(payload: dict, driver: str) -> str:
    d = driver.strip().upper()
    codes = [x["driver"] for x in payload["drivers"]]
    if d in codes:
        return d
    raise ValueError(f"unknown driver '{driver}'; this race has: {', '.join(codes)}")


@tool
def race_forecast(season: int | None = None, round: int | None = None) -> str:
    """The model's forecast for one race: every driver's predicted finishing
    order, chances to win / finish on the podium / score points, likely
    finishing range, retirement risk, qualifying prediction (lap time, gap,
    pole and Q3 chances) and gap to the winner, plus race length and
    safety-car chance. Omit season/round for the current race weekend. Past
    races without a stored forecast raise an error: use past_race_prediction."""
    p = _forecast(season, round)
    race = p.get("race", {})
    drivers = []
    for rank, d in enumerate(p["drivers"], 1):
        pr = d.get("probabilities", {})
        q = d.get("quali_odds", {})
        drivers.append({
            "predicted_rank": rank, "driver": d["driver"], "team": d["team"], "grid": d.get("grid_position"),
            "win": _pct(pr.get("win")), "podium": _pct(pr.get("podium")), "points": _pct(pr.get("top10")),
            "likely_range": d.get("position_band"), "retire_risk": _pct(d.get("retire_risk")),
            "beats_teammate": _pct(d.get("beats_teammate")),
            "predicted_quali_lap_s": d.get("predicted_quali_lap_s"), "predicted_quali_gap_s": d.get("predicted_qualifying_gap"),
            "actual_quali_gap_s": d.get("quali_gap_to_pole"), "pole": _pct(q.get("pole")), "q3": _pct(q.get("q3")),
            "out_in_q1": _pct(q.get("q1_out")), "gap_to_winner_s": d.get("predicted_race_time_gap"),
        })
    return _dump({
        "race": f"{p['season']} round {p['round']}, {p.get('event_name') or p['location']}",
        "race_start_utc": p.get("race_start_utc"), "forecast_as_of": p.get("session_label") or p.get("stage"),
        "generated_at": p.get("generated_at"), "laps": race.get("laps"),
        "expected_race_duration_min": None if race.get("expected_duration_s") is None else round(race["expected_duration_s"] / 60, 1),
        "predicted_pole_lap_s": race.get("pole_time_estimate_s"), "safety_car_chance": _pct(race.get("safety_car_probability")),
        "drivers": drivers,
    })


@tool
def explain_prediction(driver: str, target: Target = "finish_position", season: int | None = None, round: int | None = None) -> str:
    """Why the model predicts what it does for one driver: the inputs that
    pushed the prediction up or down most (SHAP contributions, in the
    target's own units), with each input's value. `driver` is the 3-letter
    code (e.g. NOR). Omit season/round for the current race weekend."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    code = _code(p, driver)
    d = next(x for x in p["drivers"] if x["driver"] == code)
    exp = d.get("explanations", {}).get(target)
    if exp is None:
        raise ValueError("this forecast has no stored breakdown")
    return _dump({
        "driver": code, "prediction_of": TARGET_MEANING[target], "predicted": exp["predicted_value"],
        "average_prediction": exp["base_value"],
        "contributions": [{"input": FEATURE_LABELS.get(c["feature"], c["feature"]), "value": c["value"], "effect": c["shap"]}
                          for c in exp["top_contributions"]],
        "how_to_read": "effect is how much this input moved the prediction from the average, in the prediction's units; "
                       "the effects plus the average add up to the prediction",
    })


@tool
def head_to_head(driver_a: str, driver_b: str, season: int | None = None, round: int | None = None) -> str:
    """Chance driver_a finishes ahead of driver_b in a race, from 10,000
    simulated races. 3-letter codes."""
    p = _forecast(season, round)
    a, b = _code(p, driver_a), _code(p, driver_b)
    pa = _h2h(p, a, b)
    exp = {d["driver"]: d.get("expected_position") for d in p["drivers"]}
    return _dump({"race": f"{p['season']} round {p['round']}", f"{a}_ahead_of_{b}": _pct(pa),
                  "expected_finish": {a: exp.get(a), b: exp.get(b)}})


def championship_state() -> dict:
    """Standings, remaining rounds and simulated title odds (shared with the API)."""
    standings = f1_api.get_driver_standings()
    remaining = remaining_rounds(f1_api.get_season_schedule(), f1_api.get_current_round())
    forecast = with_probabilities(get_json("latest.json"))
    strength = {d["driver"]: d["predicted_finish_position"] for d in forecast["drivers"]}
    worst = max(strength.values()) + 2
    retire = [d.get("retire_risk") or 0.1 for d in forecast["drivers"]]
    odds = title_odds(
        [s["points"] for s in standings], [strength.get(s["code"], worst) for s in standings], remaining,
        tau=load_calibration()["race"]["pre_weekend"]["tau_field"], p_dnf=float(np.mean(retire)),
        groups=[s["team"] for s in standings],
    )
    return {"standings": standings, "remaining": remaining, "odds": odds}


@tool
def championship(kind: Literal["drivers", "constructors"] = "drivers") -> str:
    """Live championship standings (points, wins) plus each contender's
    chance of winning the title, from simulating the rest of the season
    5,000 times with every driver's current forecast form."""
    c = championship_state()
    left = {"races": len(c["remaining"]), "sprints": sum(r["is_sprint"] for r in c["remaining"])}
    if kind == "constructors":
        teams = f1_api.get_constructor_standings()
        return _dump({"remaining": left, "constructors": [
            {**t, "title_chance": _pct(c["odds"]["team_champion"].get(t["name"]))} for t in teams]})
    return _dump({"remaining": left, "drivers": [
        {"position": s["position"], "driver": s["code"], "name": f"{s['given_name']} {s['family_name']}", "team": s["team"],
         "points": s["points"], "wins": s["wins"], "title_chance": _pct(c["odds"]["champion"][i]),
         "expected_final_points": round(float(c["odds"]["expected_points"][i]))}
        for i, s in enumerate(c["standings"])]})


@tool
def title_scenario_for(driver: str) -> str:
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
    return _dump({"driver": me["code"], "rival": rival["code"],
                  **title_scenario(me["points"], rival["points"], c["remaining"]),
                  "title_chance": _pct(c["odds"]["champion"][i])})


@tool
def race_strategy(safety_car_lap: int | None = None, season: int | None = None, round: int | None = None) -> str:
    """Tyre strategy for a race: the fastest plans (compounds, pit laps,
    pit windows), how much slower each alternative is, and each plan's chance
    of being fastest once random safety cars are simulated. Pass
    safety_car_lap to see how a safety car on that lap changes the best
    plan. Omit season/round for the current race weekend."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    laps = p.get("race", {}).get("laps") or 57
    return _dump(simulate(p["location"], int(laps), sc_lap=safety_car_lap))


@tool
def forecast_timeline(driver: str | None = None, season: int | None = None, round: int | None = None) -> str:
    """How the forecast changed through the weekend (before practice, after
    each practice, after qualifying): predicted finishing position per
    driver at each point. Pass a driver code to follow one driver."""
    p = get_json("latest.json" if season is None or round is None else f"{season}_{round}.json")
    timeline = get_json(f"timeline/{p['season']}_{p['round']}.json")
    code = _code(p, driver) if driver else None
    return _dump([{"when": s["label"], "drivers": [
        {"driver": d["driver"], "predicted_finish": d["predicted_finish_position"]}
        for d in s["drivers"] if code is None or d["driver"] == code]} for s in timeline])


@tool(response_format="content_and_artifact")
def search_rules(query: str) -> tuple[str, list[dict]]:
    """Search the 2026 FIA Sporting Regulations, penalty guidelines, driving
    standards and real steward decisions. Use specific keywords ("unsafe
    release", "grid penalty power unit", "safety car restart"). Cite results
    by article number and document."""
    hits = search(query, k=5, doc_types={"regulation", "steward_decision"})
    sources = [{"filename": h["source"], "title": h["title"], "article": h["article"], "doc_type": h["doc_type"]} for h in hits]
    return _dump([{"document": h["source"], "article": h["article"], "text": h["text"][:1200]} for h in hits]), sources


@tool
def circuit_guide(circuit: str) -> str:
    """What a circuit is like: layout, overtaking, tyre wear, safety cars,
    weather, history. `circuit` is the location name (Monaco, Monza, Baku,
    Silverstone, Marina Bay, Yas Island, ...)."""
    text = circuit_summary(circuit)
    if not text:
        hits = search(circuit, k=1, doc_types={"circuit_summary"})
        text = hits[0]["text"] if hits else ""
    return text or f"No write-up for '{circuit}'."


@tool
def driver_history(driver: str, circuit: str | None = None, last_n: int = 10) -> str:
    """A driver's recent real results since 2022 (grid, qualifying position,
    finish, status, points), optionally only at one circuit."""
    rows = [r for r in get_json("backtest/results.json") if r["driver"] == driver.strip().upper()]
    if circuit:
        rows = [r for r in rows if circuit.lower() in r["location"].lower()]
    return _dump(rows[-last_n:]) if rows else f"No results for {driver}{' at ' + circuit if circuit else ''}."


@tool
def past_race_prediction(season: int, round: int) -> str:
    """For a race that has already happened: what the model predicted
    beforehand (from a model trained only on earlier races) next to the real
    result, per driver."""
    b = get_json(f"backtest/{season}_{round}.json")
    rows = sorted(b["drivers"], key=lambda d: d["finish_position"]["actual"] or 99)
    return _dump({"race": f"{season} round {round}, {b['location']}", "method": b.get("method"), "drivers": [
        {"driver": d["driver"], "actual_finish": d["finish_position"]["actual"], "predicted_finish": d["finish_position"]["predicted"],
         "actual_places_gained": d["quali_delta"]["actual"], "predicted_places_gained": d["quali_delta"]["predicted"]}
        for d in rows]})


@tool
def model_track_record() -> str:
    """How accurate the models have been on races they had never seen, per
    season and per weekend stage, against simple baselines (e.g. "finish where
    you start"), and how often they called the winner and pole."""
    summary = get_json("backtest/summary.json")
    held_out = {t: json.loads((MODEL_DIR / f"{t}_metrics.json").read_text()) for t in CANONICAL_PRED_COLS}
    return _dump({"by_season": summary, "held_out": {
        t: {"what": m["evaluation"], "unit": m["unit"], "baseline": m["baseline"], "error_by_stage": m["stages"]}
        for t, m in held_out.items()}})


@tool
def season_schedule(season: int | None = None) -> str:
    """The season calendar: rounds, venues, race start times (UTC), sprint
    weekends, and which race is next."""
    season = season or datetime.now(timezone.utc).year
    sched = get_json(f"schedule/{season}.json")
    now = datetime.now(timezone.utc).isoformat()
    upcoming = [r for r in sched if (r.get("RaceStartUtc") or "") > now[:19]]
    return _dump({"next_round": upcoming[0]["RoundNumber"] if upcoming else None, "rounds": [
        {"round": r["RoundNumber"], "event": r["EventName"], "location": r["Location"], "race_start_utc": r.get("RaceStartUtc"),
         "sprint_weekend": "sprint" in (r.get("EventFormat") or "")} for r in sched]})


TOOLS = [race_forecast, explain_prediction, head_to_head, championship, title_scenario_for, race_strategy,
         forecast_timeline, search_rules, circuit_guide, driver_history, past_race_prediction, model_track_record,
         season_schedule]
