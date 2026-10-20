"""The data the chat router answers from. Every one of these is a plain
Python function returning a plain dict (or list) built from real numbers --
cached forecasts and backtests (src/api/cache.py), the odds sampler, the
strategy simulator, the regulations index and live standings. None of them
calls an LLM: see src/agent/router.py for how a question is matched to one
of these, and src/agent/chat.py for the two places (and only two) a local
model is used to turn a result into prose.
"""
import builtins
import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import numpy as np

from src.agent import context, f1_api
from src.agent.scenarios import closest_rival, remaining_rounds, title_odds, title_scenario
from src.api.cache import LOCAL_DATA_DIR, get_json
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


def _payload_name(season: int | None = None, round: int | None = None, folder: str = "") -> str:
    """Which forecast file a tool reads: the one asked for, else the current
    weekend: the live latest.json, or the replayed snapshot (src/agent/context.py)."""
    if season is not None and round is not None:
        return f"{folder}{season}_{round}.json"
    snap = context.snapshot()
    if not snap:
        return "latest.json"
    variant = context.variant()
    name = f"{snap[0]}_{snap[1]}{'_' + variant if variant else ''}.json"
    return f"snapshots/{folder}{name}" if (LOCAL_DATA_DIR / "snapshots" / folder / name).exists() else f"{folder}{name}"


def _completed_round() -> int | None:
    """The last finished round the standings should reflect when replaying a weekend."""
    snap = context.snapshot()
    return snap[1] - 1 if snap else None


def pct(p) -> float | None:
    return None if p is None else round(float(p), 4)


@lru_cache(maxsize=1)
def _people() -> dict:
    """Every F1 driver's name and code (data/history/people.json, which ships
    with the serverless API: 200 KB, unlike the results table)."""
    from src.agent.history import HISTORY_DIR

    path = HISTORY_DIR / "people.json"
    return json.loads(path.read_text(encoding="utf-8"))["drivers"] if path.exists() else {}


def grid_names(codes) -> dict[str, dict]:
    """code -> {given, family, name} for the drivers in one race: the most
    recent driver to carry each code (codes get reused: MSC, VER)."""
    best: dict[str, dict] = {}
    for p in _people().values():
        c = p.get("code")
        if c in codes and (c not in best or (p.get("born") or "") > (best[c].get("born") or "")):
            best[c] = p
    return best


def driver_name(code: str) -> str | None:
    """A driver's full name from his 3-letter code, so nothing (a person or a
    model) has to guess who "ANT" is."""
    return (grid_names({code}).get(code) or {}).get("name")


# lowercase words that are also driver codes: only count them typed in capitals
_WORD_CODES = {"had", "nor", "per", "law", "ant", "gas", "col", "bot", "mag", "sit", "max"}
NICKNAMES = {"checo": "perez", "alex": "albon", "ollie": "bearman", "hulk": "hulkenberg"}


def drivers_in(text: str, codes) -> list[str]:
    """The drivers of one race named in `text`, in the order they appear: by
    code ("NOR", or "nor"), surname, first name (when only one driver on the
    grid has it) or nickname ("Checo"). The one resolver behind every
    driver argument, so "Norris vs Verstappen" works wherever "NOR vs VER" does."""
    from src.agent.history import _fold

    q = _fold(text)
    people = grid_names(set(codes))
    given_count = Counter(w for p in people.values() for w in _fold(p["given"]).split())
    found: list[tuple[int, str]] = []
    for code in codes:
        p = people.get(code, {})
        names = [_fold(p["family"])] if p else []
        names += [w for w in (_fold(p["given"]).split() if p else []) if given_count[w] == 1]
        names += [nick for nick, fam in NICKNAMES.items() if names and fam == names[0]]
        spots = [m.start() for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", q)]
        spots += [m.start() for m in re.finditer(rf"\b{code}\b", text)]  # capitals: always a code
        if code.lower() not in _WORD_CODES:
            spots += [m.start() for m in re.finditer(rf"\b{code.lower()}\b", q)]
        if spots:
            found.append((min(spots), code))
    return [c for _, c in sorted(found)]


def forecast(season: int | None = None, round: int | None = None) -> dict:
    """The model's forecast for one race: every driver's predicted finishing
    order, win/podium/points chances, likely finishing range, retirement
    risk, qualifying prediction (lap time, gap, pole/Q3 chances) and gap to
    the winner, plus race length and safety-car chance. season/round=None
    means the current race weekend. Raises for a past race with no stored
    forecast -- use past_race_prediction for those."""
    payload = get_json(_payload_name(season, round))
    p = with_probabilities(payload)
    race = p.get("race", {})
    drivers = []
    for rank, d in enumerate(p["drivers"], 1):
        pr = d.get("probabilities", {})
        q = d.get("quali_odds", {})
        drivers.append({
            "predicted_rank": rank, "driver": d["driver"], "name": driver_name(d["driver"]), "team": d["team"], "grid": d.get("grid_position"),
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
        "expected_race_duration_min": None if race.get("expected_duration_s") is None else builtins.round(race["expected_duration_s"] / 60, 1),  # the `round` param shadows the builtin
        "predicted_pole_lap_s": race.get("pole_time_estimate_s"), "safety_car_chance": pct(race.get("safety_car_probability")),
        "drivers": drivers,
    }


def driver_code(payload: dict, driver: str) -> str:
    """The 3-letter code of the one driver `driver` names (code, surname,
    first name or nickname) in this payload's field."""
    codes = [x["driver"] for x in payload["drivers"]]
    d = driver.strip().upper()
    found = [d] if d in codes else drivers_in(driver, codes)
    if len(found) == 1:
        return found[0]
    raise ValueError(f"unknown driver '{driver}'; this race has: {', '.join(codes)}")


def explain_prediction(driver: str, target: Target = "finish_position", season: int | None = None, round: int | None = None) -> dict:
    """Why the model predicts what it does for one driver: the inputs that
    pushed the prediction up or down most (SHAP contributions, in the
    target's own units), with each input's value. driver is the 3-letter code."""
    p = get_json(_payload_name(season, round))
    code = driver_code(p, driver)
    d = next(x for x in p["drivers"] if x["driver"] == code)
    exp = d.get("explanations", {}).get(target)
    if exp is None:
        raise ValueError("this forecast has no stored breakdown")
    return {
        "season": p["season"], "round": p["round"], "race": p.get("event_name") or p["location"], "circuit": p["location"],
        "driver": code, "name": driver_name(code), "team": d["team"], "target": target, "prediction_of": TARGET_MEANING[target],
        "predicted": exp["predicted_value"], "average_prediction": exp["base_value"],
        "contributions": [{"feature": c["feature"], "input": FEATURE_LABELS.get(c["feature"], c["feature"]),
                           "value": c["value"], "effect": c["shap"]} for c in exp["top_contributions"]],
    }


def head_to_head(driver_a: str, driver_b: str, season: int | None = None, round: int | None = None) -> dict:
    """Chance driver_a finishes ahead of driver_b in a race, from 10,000 simulated races."""
    payload = get_json(_payload_name(season, round))
    p = with_probabilities(payload)
    a, b = driver_code(p, driver_a), driver_code(p, driver_b)
    chance = _h2h(p, a, b)
    exp = {d["driver"]: d.get("expected_position") for d in p["drivers"]}
    return {"season": p["season"], "round": p["round"], "driver_a": a, "name_a": driver_name(a), "driver_b": b, "name_b": driver_name(b),
            "a_ahead_of_b": pct(chance), "expected_finish_a": exp.get(a), "expected_finish_b": exp.get(b)}


def season_strength(payload: dict) -> dict[str, float]:
    """Each driver's strength for the rest of the season (predicted finish,
    lower = stronger) with nothing specific to this weekend in it: the
    refresh job's pre-weekend prediction, else this weekend's "Before
    practice" snapshot. The shown forecast after qualifying includes the grid,
    so one weekend's pole or grid penalty would otherwise count in every
    remaining round (Singapore 2026: Russell's back-of-grid start cut his
    expected final points from 299 to 255)."""
    stored = {d["driver"]: d["season_strength"] for d in payload["drivers"] if d.get("season_strength") is not None}
    if len(stored) == len(payload["drivers"]):
        return stored
    try:
        timeline = get_json(f"timeline/{payload['season']}_{payload['round']}.json")
        pre = next((s for s in timeline if s.get("stage") == "pre_weekend"), None)
    except FileNotFoundError:
        pre = None
    if pre:
        snap = {d["driver"]: d["predicted_finish_position"] for d in pre["drivers"] if d.get("predicted_finish_position") is not None}
        if snap:
            return {d["driver"]: snap.get(d["driver"], d["predicted_finish_position"]) for d in payload["drivers"]}
    return {d["driver"]: d["predicted_finish_position"] for d in payload["drivers"]}


def championship_state() -> dict:
    """Standings, remaining rounds and simulated title odds (shared with the API)."""
    done, snap = _completed_round(), context.snapshot()
    season = str(snap[0]) if snap else "current"
    standings = f1_api.get_driver_standings(season, done)
    remaining = remaining_rounds(f1_api.get_season_schedule(season), done if done is not None else f1_api.get_current_round())
    forecast_payload = with_probabilities(get_json(_payload_name()))
    strength = season_strength(forecast_payload)
    worst = max(strength.values()) + 2
    retire = [d.get("retire_risk") or 0.1 for d in forecast_payload["drivers"]]
    odds = title_odds(
        [s["points"] for s in standings], [strength.get(s["code"], worst) for s in standings], remaining,
        tau=load_calibration()["race"]["pre_weekend"]["tau_field"], p_dnf=float(np.mean(retire)),
        groups=[s["team"] for s in standings],
    )
    return {"standings": standings, "remaining": remaining, "odds": odds}


def championship_payload() -> dict:
    """Standings with every driver's and team's title chance and expected
    points: what GET /championship serves. The refresh job precomputes it
    into data/predictions/championship.json, so a request needs neither
    Jolpica nor 5,000 simulated seasons."""
    c = championship_state()
    odds = c["odds"]
    snap = context.snapshot()
    return {
        "remaining": {"races": len(c["remaining"]), "sprints": sum(r["is_sprint"] for r in c["remaining"])},
        "drivers": [{**s, "title_chance": round(float(odds["champion"][i]), 4),
                     "expected_points": round(float(odds["expected_points"][i]), 1)} for i, s in enumerate(c["standings"])],
        "constructors": [{**t, "title_chance": round(float(odds["team_champion"].get(t["name"], 0.0)), 4)}
                         for t in f1_api.get_constructor_standings(str(snap[0]) if snap else "current", _completed_round())],
    }


def championship(kind: Literal["drivers", "constructors"] = "drivers") -> dict:
    """Live championship standings (points, wins) plus each contender's
    chance of winning the title, from simulating the rest of the season
    5,000 times with every driver's current forecast form."""
    c = championship_state()
    left = {"races": len(c["remaining"]), "sprints": sum(r["is_sprint"] for r in c["remaining"])}
    if kind == "constructors":
        snap = context.snapshot()
        teams = f1_api.get_constructor_standings(str(snap[0]) if snap else "current", _completed_round())
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
    p = get_json(_payload_name(season, round))
    laps = p.get("race", {}).get("laps") or 57
    return {"season": p["season"], "round": p["round"], "race": p.get("event_name") or p["location"], **simulate(p["location"], int(laps), sc_lap=safety_car_lap)}


def forecast_timeline(driver: str | None = None, season: int | None = None, round: int | None = None) -> list[dict]:
    """How the forecast changed through the weekend (before practice, after
    each practice, after qualifying): predicted finishing position per
    driver at each point. Pass a driver code to follow one driver."""
    p = get_json(_payload_name(season, round))
    timeline = get_json(_payload_name(folder="timeline/") if context.snapshot() and season is None
                        else f"timeline/{p['season']}_{p['round']}.json")
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


def search_knowledge(query: str, k: int = 4, only: str | None = None) -> dict:
    """Search everything written down: Wikipedia articles on drivers, teams,
    circuits, seasons and Grands Prix, rules explainers, the FIA regulations,
    steward decisions and circuit notes. For explanations, background and
    stories; use `records`/`driver_career`/`history_results` for any number.
    Hits are interleaved best-first from the two indexes (BM25 scores are not
    comparable across them) and carry their source, so they can be cited."""
    from src.rag.corpus import WIKI_INDEX_PATH

    main_types = {"rules": {"regulation", "steward_decision"}, "circuit_notes": {"circuit_summary"}}.get(only)
    main = [] if only == "wikipedia" else search(query, k=k, doc_types=main_types)
    wiki = search(query, k=k, index_path=WIKI_INDEX_PATH) if WIKI_INDEX_PATH.exists() and only in (None, "wikipedia") else []
    # BM25 always returns something; keep only hits within half the best score of their own index
    main, wiki = (([h for h in hs if h["score"] >= 0.5 * hs[0]["score"]] if hs else []) for hs in (main, wiki))
    hits, i = [], 0
    while len(hits) < k and (i < len(main) or i < len(wiki)):
        hits += [h for h in (wiki[i:i + 1] + main[i:i + 1])]
        i += 1
    return {"hits": [{"title": h["title"], "source": h["source"], "doc_type": h["doc_type"], "article": h.get("article"),
                      "license": h.get("license"), "text": h["text"][:650]} for h in hits[:k]],
            "note": "Wikipedia text is CC BY-SA 4.0" if wiki else None}


def circuit_guide(circuit: str) -> str:
    """What a circuit is like: layout, overtaking, tyre wear, safety cars,
    weather, history. circuit is the location name (Monaco, Monza, Baku,
    Silverstone, Marina Bay, Yas Island, ...)."""
    text = circuit_summary(circuit)
    if not text:
        hits = search(circuit, k=1, doc_types={"circuit_summary"})
        text = hits[0]["text"] if hits else ""
    return text


def past_race_prediction(season: int, round: int) -> dict:
    """For a race that has already happened: what the model predicted
    beforehand (from a model trained only on earlier races) next to the real
    result, per driver."""
    snap = context.snapshot()
    if snap and (season, round) >= snap:
        raise ValueError(f"{season} round {round} hasn't been raced yet")
    b = get_json(f"backtest/{season}_{round}.json")
    rows = sorted(b["drivers"], key=lambda d: d["finish_position"]["actual"] or 99)
    return {"season": season, "round": round, "circuit": b["location"], "method": b.get("method"), "drivers": [
        {"driver": d["driver"], "name": driver_name(d["driver"]), "actual_finish": d["finish_position"]["actual"], "predicted_finish": d["finish_position"]["predicted"],
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
    season = season or context.today().year
    sched = get_json(f"schedule/{season}.json")
    # live: the current UTC time, so today's race stops being "next" once it
    # has started; a replayed weekend only has a date
    now = context.today().isoformat() if context.snapshot() else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    upcoming = [r for r in sched if (r.get("RaceStartUtc") or "")[:19] > now]
    return {"season": season, "next_round": upcoming[0]["RoundNumber"] if upcoming else None, "rounds": [
        {"round": r["RoundNumber"], "event": r["EventName"], "location": r["Location"], "race_start_utc": r.get("RaceStartUtc"),
         "sprint_weekend": "sprint" in (r.get("EventFormat") or "")} for r in sched]}


def driver_ratings(driver: str | None = None) -> dict:
    """Drivers ranked by qualifying pace with the car taken out (from
    teammate head-to-heads, src/features/ratings.py): % of a lap faster (-) or
    slower (+) than an average driver, and the gap to the teammate. Pass a
    driver for that one driver and teammate."""
    if context.snapshot():  # only the current weekend's ratings are published: a replayed weekend would see the future
        raise ValueError("driver ratings are only published for the current race weekend")
    r = get_json("ratings.json")
    head = {"season": r["season"], "as_of": r.get("event_name"), "method": r["method"], "field_size": len(r["drivers"])}
    rows = [{**d, "name": driver_name(d["driver"])} for d in r["drivers"]]
    if not driver:
        return {**head, "drivers": rows}
    code = driver_code(r, driver)
    me = next(d for d in rows if d["driver"] == code)
    return {**head, "driver": me, "teammate": next((d for d in rows if d["team"] == me["team"] and d["driver"] != code), None)}
