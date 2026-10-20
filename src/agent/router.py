"""Deterministic chat router.

A LangGraph StateGraph with two real nodes: `classify` puts the latest
question into one of a fixed set of intents with a small learned classifier
(deterministic: same question, same intent), and `dispatch` calls the one
matching function in tools.py and renders a plain-English answer from its
real, already-correct result. Neither node calls an LLM, and every number in
every answer traces to a real lookup.

Two intents go on to the local model afterwards, outside this module (see
chat.py): `explain_prediction`, to write the "why" in words, and `general`,
everything the fixed intents can't answer (history, records, background),
which the model answers by looking things up. Every other intent works where
no model runs at all.
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent import tools as t

_CODE_RE = re.compile(r"\b[A-Z]{3}\b")  # driver codes are conventionally typed uppercase
_LAP_RE = re.compile(r"\blap\s*(\d{1,2})\b", re.I)
_ROUND_RE = re.compile(r"\bround\s*(\d{1,2})\b", re.I)
_YEAR_RE = re.compile(r"\b(20[12]\d)\b")
_ARTICLE_RE = re.compile(r"\b[a-z]?\d{1,2}\.\d{1,2}(?:\.\d{1,2})*\b", re.I)
CONTINUATION = ("and ", "also ", "what about", "how about", "what's his", "what is his", "what's her", "what is her")


# --- small formatting helpers, shared by every handler ---

def _pctfmt(p) -> str:
    return "—" if p is None else "<1%" if 0 < p < 0.01 else f"{p:.0%}"


def _lapfmt(s) -> str:
    if s is None:
        return "—"
    m = int(s // 60)
    return f"{m}:{s - m * 60:06.3f}"


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_No data._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _codes_in(question: str) -> list[str]:
    found = []
    for tok in _CODE_RE.findall(question):
        if tok not in found:
            found.append(tok)
    return found


@lru_cache(maxsize=1)
def _known_circuits() -> list[str]:
    from src.rag.corpus import load_index

    return sorted({c["circuit"] for c in load_index()["chunks"] if c.get("circuit")})


# What fans actually type for a circuit whose corpus name is its location.
# Only unambiguous ones: "Spain", "Italy", "USA" each have several venues.
CIRCUIT_NICKNAMES = {
    "singapore": "Marina Bay", "sepang": "Kuala Lumpur", "malaysia": "Kuala Lumpur", "bahrain": "Sakhir",
    "abu dhabi": "Yas Island", "yas marina": "Yas Island", "qatar": "Lusail", "brazil": "São Paulo",
    "interlagos": "São Paulo", "cota": "Austin", "hungary": "Budapest", "hungaroring": "Budapest",
    "austria": "Spielberg", "red bull ring": "Spielberg", "japan": "Suzuka", "china": "Shanghai",
    "australia": "Melbourne", "albert park": "Melbourne", "canada": "Montréal", "montreal": "Montréal",
    "mexico": "Mexico City", "netherlands": "Zandvoort", "dutch": "Zandvoort", "azerbaijan": "Baku",
    "saudi": "Jeddah", "vegas": "Las Vegas", "british": "Silverstone", "belgium": "Spa-Francorchamps",
    "belgian": "Spa-Francorchamps", "spa": "Spa-Francorchamps", "sao paulo": "São Paulo", "paul ricard": "Le Castellet",
}
_NICKNAME_RE = re.compile(r"\b(" + "|".join(map(re.escape, CIRCUIT_NICKNAMES)) + r")\b")


def _circuit_in(question_lower: str) -> str | None:
    for name in _known_circuits():
        if name.lower() in question_lower:
            return name
    m = _NICKNAME_RE.search(question_lower)
    return CIRCUIT_NICKNAMES[m.group(1)] if m else None


def _target_in(question_lower: str) -> str:
    if any(w in question_lower for w in ("qualify", "quali", "pole", "grid slot")):
        return "qualifying"
    if any(w in question_lower for w in ("places gained", "place gained", "grid to finish", "overtake", "positions gained")):
        return "quali_delta"
    if any(w in question_lower for w in ("gap to the winner", "gap to winner", "race time", "behind the winner", "time gap")):
        return "race_time"
    return "finish_position"


def _named(drivers: list[dict], question: str) -> dict | None:
    """The first of `drivers` (dicts with a "driver" code) the question names."""
    found = t.drivers_in(question, [d["driver"] for d in drivers])
    return next(d for d in drivers if d["driver"] == found[0]) if found else None


def _field(ctx: dict) -> list[str]:
    return [d["driver"] for d in t.forecast(ctx.get("season"), ctx.get("round"))["drivers"]]


# --- intent classification: a small learned classifier, no network, no LLM ---

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token pattern


@lru_cache(maxsize=1)
def _model() -> dict:
    return json.loads((Path(__file__).with_name("router_model.json")).read_text(encoding="utf-8"))


EAGER_P = 0.2  # without a model: answer with the likeliest fixed intent above this (recall 55% vs 45%, wrong 4.7% vs 2.6%)


def _learned_intent(q: str, eager: bool = False) -> str:
    """TF-IDF over words and word pairs + logistic regression, fitted on ~21k
    real-phrased questions labelled by the lookup they needed
    (experiments/router_classifier.py, which also exports router_model.json).
    Scored here in plain Python so the API needs no scikit-learn. It replaced
    keyword rules that sent 45% of free-form questions to a wrong answer
    ("how many races did Senna win?" got this weekend's forecast).

    `eager`: no local model will answer a `general` question, only a generic
    reply, so take the likeliest fixed intent when it is reasonably likely."""
    m = _model()
    toks = _TOKEN_RE.findall(q.lower())
    counts = Counter(g for g in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])] if g in m["vocabulary"])
    x = {m["vocabulary"][g]: (1 + math.log(c)) * m["idf"][m["vocabulary"][g]] for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
    scores = [b + sum(row[i] * v for i, v in x.items()) / norm for row, b in zip(m["coef"], m["intercept"])]
    best = max(range(len(scores)), key=scores.__getitem__)
    if not eager or m["classes"][best] != "general":
        return m["classes"][best]
    top = max(scores)
    probs = [math.exp(v - top) for v in scores]
    fixed = max((i for i, c in enumerate(m["classes"]) if c != "general"), key=probs.__getitem__)
    return m["classes"][fixed] if probs[fixed] / sum(probs) > EAGER_P else "general"


# "why" with "predicted"/"forecast": 124 of 129 such training questions wanted the breakdown,
# and the classifier alone was on a knife-edge for them ("why is he predicted there")
_WHY_PREDICTED_RE = re.compile(r"\bwhy\b.*\b(predict|forecast)\w*|\b(predict|forecast)\w*\b.*\bwhy\b")


def _classify_specific(q: str, eager: bool = False) -> str | None:
    if _ARTICLE_RE.search(q):  # "what does B1.6.2 say": an article number is unambiguous
        return "search_rules"
    if _WHY_PREDICTED_RE.search(q):
        return "explain_prediction"
    intent = _learned_intent(q, eager)
    return None if intent == "general" else intent


def classify_intent(messages: list[dict], eager: bool = False) -> str:
    """The intent for the latest message, falling back to the previous
    user turn's intent for an explicit continuation ("and what about...") --
    a real discourse signal, not just "nothing else matched"."""
    question = (messages[-1].get("content") or "").lower().strip()
    specific = _classify_specific(question, eager)
    if specific:
        return specific
    if question.startswith(CONTINUATION) and len(messages) > 1:
        prev = next((m["content"] for m in reversed(messages[:-1]) if m.get("role") == "user"), None)
        if prev:
            inherited = _classify_specific(prev.lower().strip(), eager)
            if inherited:
                return inherited
    return "general"


# --- per-intent handlers: look up real data, render plain English ---

def _handle_race_forecast(question: str, ctx: dict) -> dict:
    ql = question.lower()
    data = t.forecast(ctx.get("season"), ctx.get("round"))
    drivers = data["drivers"]
    quali_lens = any(w in ql for w in ("pole", "qualify", "quali", "q3", "q1", "grid"))
    if quali_lens and any(d.get("pole") is not None for d in drivers):
        ranked = sorted(drivers, key=lambda d: -(d.get("pole") or 0))[:8]
        rows = [{"Driver": d["driver"], "Team": d["team"], "Pole": _pctfmt(d["pole"]), "Q3": _pctfmt(d["q3"]),
                 "Lap": _lapfmt(d["predicted_quali_lap_s"])} for d in ranked]
        top = ranked[0]
        lead = f"**{top['driver']}** ({top['team']}) is favourite for pole, {_pctfmt(top['pole'])} to get there."
    else:
        ranked = drivers[:8]
        rows = [{"Pos": d["predicted_rank"], "Driver": d["driver"], "Team": d["team"], "Win": _pctfmt(d["win"]),
                 "Podium": _pctfmt(d["podium"]), "Points": _pctfmt(d["points"])} for d in ranked]
        top = drivers[0]
        lead = f"**{top['driver']}** ({top['team']}) is favourite, about {_pctfmt(top['win'])} to win."
    text = f"{lead}\n\n{_md_table(rows)}\n\n_{data['race']}, forecast {data['forecast_as_of'] or 'as of the current stage'}._"
    return {"tool": "race_forecast", "text": text, "sources": None}


def _handle_explain_prediction(question: str, ctx: dict) -> dict:
    season, round_ = ctx.get("season"), ctx.get("round")
    code = next(iter(t.drivers_in(question, _field(ctx))), None) or ctx.get("driver")
    if not code:
        raise ValueError('Tell me which driver -- e.g. "why is Norris predicted there?"')
    exp = t.explain_prediction(code, target=_target_in(question.lower()), season=season, round=round_)
    return {"tool": "explain_prediction", "llm": True, "llm_payload": exp, "sources": None}


def _handle_head_to_head(question: str, ctx: dict) -> dict:
    season, round_ = ctx.get("season"), ctx.get("round")
    codes = t.drivers_in(question, _field(ctx))
    if len(codes) < 2:
        raise ValueError('Name two drivers in this race -- e.g. "Norris vs Verstappen" or "NOR vs VER".')
    a, b = codes[0], codes[1]
    h = t.head_to_head(a, b, season, round_)
    text = (f"**{a}** finishes ahead of **{b}** in {_pctfmt(h['a_ahead_of_b'])} of 10,000 simulated races. "
            f"Expected finish: {a} P{h['expected_finish_a']:.1f}, {b} P{h['expected_finish_b']:.1f}.")
    return {"tool": "head_to_head", "text": text, "sources": None}


def _handle_constructor_standings(question: str, ctx: dict) -> dict:
    data = t.championship(kind="constructors")
    teams = data["constructors"]
    lead = f"**{teams[0]['name']}** leads the constructors' championship with {teams[0]['points']:.0f} points."
    rows = [{"Pos": c["position"], "Team": c["name"], "Points": c["points"], "Wins": c["wins"], "Title": _pctfmt(c["title_chance"])} for c in teams[:8]]
    return {"tool": "championship", "text": f"{lead}\n\n{_md_table(rows)}", "sources": None}


def _handle_driver_standings(question: str, ctx: dict) -> dict:
    data = t.championship(kind="drivers")
    drivers = data["drivers"]
    named = _named(drivers, question)
    if named:
        lead = (f"**{named['name']}** ({named['team']}) is P{named['position']} with {named['points']:.0f} points and "
                f"{named['wins']} win{'s' if named['wins'] != 1 else ''}. Title chance: {_pctfmt(named['title_chance'])}.")
    else:
        lead = f"**{drivers[0]['name']}** leads the drivers' championship with {drivers[0]['points']:.0f} points."
    rows = [{"Pos": d["position"], "Driver": d["driver"], "Team": d["team"], "Points": d["points"], "Title": _pctfmt(d["title_chance"])} for d in drivers[:8]]
    return {"tool": "championship", "text": f"{lead}\n\n{_md_table(rows)}", "sources": None}


def _handle_title_scenario(question: str, ctx: dict) -> dict:
    data = t.championship(kind="drivers")
    named = _named(data["drivers"], question)
    driver = named["driver"] if named else data["drivers"][0]["driver"]  # no one named: default to the leader
    s = t.title_scenario_for(driver)
    if not s["still_mathematically_in_contention"]:
        text = (f"**{s['driver_name']}** is mathematically out of the running against {s['rival_name']} "
                f"({s['rival_points']:.0f} points), with only {s['max_points_available']:.0f} points still available.")
    elif s["points_behind"] <= 0:
        text = (f"**{s['driver_name']}** currently leads {s['rival_name']} by {abs(s['points_behind']):.0f} points, "
                f"with {s['remaining_races']} races left and a {_pctfmt(s['title_chance'])} simulated title chance.")
    else:
        text = (f"**{s['driver_name']}** is {s['points_behind']:.0f} points behind {s['rival_name']}, with "
                f"{s['remaining_races']} races ({s['remaining_sprints']} sprints) left. Needs "
                f"{s['points_needed_if_rival_scores_zero']:.0f} points to draw level if {s['rival_name']} scores nothing more. "
                f"Simulated title chance: {_pctfmt(s['title_chance'])}.")
    return {"tool": "title_scenario_for", "text": text, "sources": None}


def _handle_race_strategy(question: str, ctx: dict) -> dict:
    ql = question.lower()
    sc_lap = int(m.group(1)) if "safety car" in ql and (m := _LAP_RE.search(ql)) else None
    s = t.race_strategy(sc_lap, ctx.get("season"), ctx.get("round"))
    best = s["strategies"][0]
    lead = (f"Fastest plan: **{best['name']}**, pitting on lap {', '.join(str(x) for x in best['pit_laps'])}. "
            f"Safety car chance this race: {_pctfmt(s['safety_car_probability'])}.")
    rows = [{"Plan": p["name"], "Pits": ", ".join(f"lap {x}" for x in p["pit_laps"]),
             "vs best": "fastest" if not p["time_vs_best_s"] else f"+{p['time_vs_best_s']:.1f}s",
             "Chance fastest": _pctfmt(p["chance_fastest"])} for p in s["strategies"][:4]]
    text = f"{lead}\n\n{_md_table(rows)}"
    if s.get("scenario"):
        sc = s["scenario"]
        gain = f"about {sc['gain_vs_sticking_to_plan_s']:.1f}s" if sc["gain_vs_sticking_to_plan_s"] > 0.05 else "no real gain"
        text += (f"\n\nWith a safety car on lap {sc['sc_lap']}: best becomes **{sc['best']}**, pitting lap "
                 f"{', '.join(str(x) for x in sc['pit_laps'])}{' (under the safety car)' if sc['pits_under_safety_car'] else ''}. "
                 f"{gain.capitalize()} over sticking to the original plan.")
    return {"tool": "race_strategy", "text": text, "sources": None}


def _handle_forecast_timeline(question: str, ctx: dict) -> dict:
    season, round_ = ctx.get("season"), ctx.get("round")
    code = next(iter(t.drivers_in(question, _field(ctx))), None) or ctx.get("driver")
    timeline = t.forecast_timeline(code, season, round_)
    if not timeline:
        raise ValueError("No timeline recorded for this race yet.")
    if code:
        rows = [{"When": s["when"], "Predicted": f"P{s['drivers'][0]['predicted_finish']:.0f}" if s["drivers"] and s["drivers"][0]["predicted_finish"] is not None else "—"} for s in timeline]
        text = f"How **{code}**'s predicted finish moved through the weekend:\n\n{_md_table(rows)}"
    else:
        last = sorted(timeline[-1]["drivers"], key=lambda d: d["predicted_finish"] if d["predicted_finish"] is not None else 99)[:8]
        rows = [{"Driver": d["driver"], "Now": f"P{d['predicted_finish']:.0f}" if d["predicted_finish"] is not None else "—"} for d in last]
        text = f"The forecast has updated {len(timeline)} time(s) this weekend ({', '.join(s['when'] for s in timeline)}).\n\n{_md_table(rows)}"
    return {"tool": "forecast_timeline", "text": text, "sources": None}


def _handle_search_rules(question: str, ctx: dict) -> dict:
    hits = t.search_rules(question, k=5)
    if not hits:
        return {"tool": "search_rules",
                "text": 'No matching regulation or steward decision. Try more specific terms, e.g. "unsafe release" or "grid penalty power unit".',
                "sources": []}
    lines, sources = [], []
    for h in hits[:3]:
        cite = h["title"] + (f", Article {h['article']}" if h["article"] else "")
        snippet = h["text"][:400].strip()
        lines.append(f"**{cite}**\n> {snippet}{'…' if len(h['text']) > 400 else ''}")
        sources.append({"filename": h["source"], "title": h["title"], "article": h["article"], "doc_type": h["doc_type"]})
    return {"tool": "search_rules", "text": "\n\n".join(lines), "sources": sources}


def _handle_circuit_guide(question: str, ctx: dict) -> dict:
    circuit = _circuit_in(question.lower())
    if not circuit:
        raise ValueError('Which circuit? e.g. "what\'s Monaco like?"')
    text = t.circuit_guide(circuit)
    if not text:
        raise ValueError(f"No write-up for {circuit} yet.")
    return {"tool": "circuit_guide", "text": text, "sources": None}


def _handle_past_race_prediction(question: str, ctx: dict) -> dict:
    ql = question.lower()
    rm, ym = _ROUND_RE.search(ql), _YEAR_RE.search(ql)
    season = int(ym.group(1)) if ym else ctx.get("season")
    round_ = int(rm.group(1)) if rm else ctx.get("round")
    if not season or not round_:
        raise ValueError('Tell me which race -- e.g. "what did you predict for round 10, 2025?"')
    p = t.past_race_prediction(season, round_)
    rows = [{"Driver": d["driver"], "Predicted": f"P{d['predicted_finish']:.0f}" if d["predicted_finish"] is not None else "—",
             "Actual": f"P{d['actual_finish']:.0f}" if d["actual_finish"] is not None else "—"} for d in p["drivers"][:10]]
    text = f"{season} round {round_}, {p['circuit']} -- {p.get('method', 'predicted before the race')}:\n\n{_md_table(rows)}"
    return {"tool": "past_race_prediction", "text": text, "sources": None}


def _handle_model_track_record(question: str, ctx: dict) -> dict:
    rec = t.model_track_record()
    rows = []
    for target, m in rec["held_out"].items():
        last_stage = list(m["error_by_stage"])[-1]
        acc = m["error_by_stage"][last_stage]
        rows.append({"Target": target.replace("_", " "), "Error": f"{acc['mae']:.2f}{m['unit'] if '%' in m['unit'] else ' ' + m['unit']}",
                     "Baseline": f"{acc['baseline_mae']:.2f}", "vs": m["baseline"]})
    text = ("Accuracy on races the models had never seen (not the races they trained on):\n\n" + _md_table(rows)
            + "\n\n_Lower error is better; \"baseline\" is the simple guess each model has to beat._")
    return {"tool": "model_track_record", "text": text, "sources": None}


def _handle_season_schedule(question: str, ctx: dict) -> dict:
    ql = question.lower()
    ym = _YEAR_RE.search(ql)
    sched = t.season_schedule(int(ym.group(1)) if ym else ctx.get("season"))
    rows = [{"Rnd": r["round"], "Event": r["event"], "Location": r["location"],
             "Sprint": "yes" if r["sprint_weekend"] else ""} for r in sched["rounds"]]
    next_round = sched.get("next_round")
    lead = f"Next round: **{next_round}**." if next_round else "No more rounds scheduled this season."
    return {"tool": "season_schedule", "text": f"{lead}\n\n{_md_table(rows)}", "sources": None}


def _handle_driver_ratings(question: str, ctx: dict) -> dict:
    r = t.driver_ratings()
    rows = r["drivers"]
    me = _named(rows, question)
    fmt = lambda x: f"{x:+.2f}%"  # noqa: E731
    if me:
        mate = next((d for d in rows if d["team"] == me["team"] and d["driver"] != me["driver"]), None)
        text = (f"**{me['name'] or me['driver']}** ranks **{me['rank']}** of {len(rows)} on qualifying pace with the car taken out: "
                f"{fmt(me['rating_pct'])} of a lap against an average driver"
                + (f", {fmt(me['vs_teammate_pct'])} against {mate['name'] or mate['driver']}." if mate else "."))
    else:
        table = _md_table([{"Rank": d["rank"], "Driver": d["driver"], "Team": d["team"], "Rating": fmt(d["rating_pct"]),
                            "vs teammate": fmt(d["vs_teammate_pct"])} for d in rows[:10]])
        text = f"**{rows[0]['name'] or rows[0]['driver']}** is the fastest qualifier once the car is taken out.\n\n{table}"
    return {"tool": "driver_ratings", "text": f"{text}\n\n_Negative is faster. {r['method'].capitalize()}._", "sources": None}


def _handle_general(question: str, ctx: dict) -> dict:
    """Facts handed to the local model for an unmatched question. The current
    race's own circuit guide is always included, not just when the question
    happens to name the circuit -- a real generation test showed the model
    guessing "street circuit" from nothing when it wasn't given this, for a
    question that was obviously about the current race weekend without
    naming the circuit (e.g. "what's going on this weekend")."""
    facts: dict = {"question": question}
    circuit = _circuit_in(question.lower())
    try:
        data = t.forecast(ctx.get("season"), ctx.get("round"))
        facts["current_forecast_top3"] = [{"driver": d["driver"], "team": d["team"], "win": d.get("win")} for d in data["drivers"][:3]]
        facts["race"] = data["race"]
        circuit = circuit or data["circuit"]
    except Exception:
        pass
    if circuit:
        facts["circuit_guide"] = t.circuit_guide(circuit)
    return {"tool": "general", "llm": True, "llm_payload": facts, "sources": None}


DISPATCH: dict[str, Callable[[str, dict], dict]] = {
    "race_forecast": _handle_race_forecast,
    "explain_prediction": _handle_explain_prediction,
    "head_to_head": _handle_head_to_head,
    "constructor_standings": _handle_constructor_standings,
    "driver_standings": _handle_driver_standings,
    "title_scenario": _handle_title_scenario,
    "race_strategy": _handle_race_strategy,
    "forecast_timeline": _handle_forecast_timeline,
    "search_rules": _handle_search_rules,
    "circuit_guide": _handle_circuit_guide,
    "past_race_prediction": _handle_past_race_prediction,
    "model_track_record": _handle_model_track_record,
    "season_schedule": _handle_season_schedule,
    "driver_ratings": _handle_driver_ratings,
    "general": _handle_general,
}


# a handler's own successful "tool" label can differ from the intent name
# (two intents share one underlying lookup) -- used so a FAILED lookup still
# reports the same label a successful one would have
_DISPLAY_TOOL = {"constructor_standings": "championship", "driver_standings": "championship", "title_scenario": "title_scenario_for"}


def dispatch(intent: str, question: str, context: dict | None = None) -> dict:
    """Run one intent's handler, turning a failed lookup into a clear
    message instead of a crash -- the `ok` flag tells chat.py whether to
    report the step as succeeded or failed."""
    handler = DISPATCH.get(intent, _handle_general)
    try:
        result = handler(question, context or {})
        result.setdefault("llm", False)
        result.setdefault("ok", True)
        return result
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else f"Couldn't answer that ({type(exc).__name__})."
        return {"tool": _DISPLAY_TOOL.get(intent, intent), "text": message, "sources": None, "llm": False, "ok": False}


# --- the LangGraph wrapper: classify -> dispatch, a real StateGraph ---

class RouteState(TypedDict):
    messages: list[dict]
    context: dict
    eager: bool
    intent: str | None
    result: dict | None


def _classify_node(state: RouteState) -> dict:
    return {"intent": classify_intent(state["messages"], state.get("eager", False))}


def _dispatch_node(state: RouteState) -> dict:
    question = state["messages"][-1]["content"]
    return {"result": dispatch(state["intent"], question, state.get("context"))}


@lru_cache(maxsize=1)
def _graph():
    g = StateGraph(RouteState)
    g.add_node("classify", _classify_node)
    g.add_node("dispatch", _dispatch_node)
    g.add_edge(START, "classify")
    g.add_edge("classify", "dispatch")
    g.add_edge("dispatch", END)
    return g.compile()


def route(messages: list[dict], context: dict | None = None, eager: bool = False) -> dict:
    """The one entry point chat.py calls: classify the latest message and
    run its handler. Returns {tool, ok, sources, and either text (ready to
    show) or llm=True + llm_payload (for chat.py to hand to the local model)}."""
    out = _graph().invoke({"messages": messages, "context": context or {}, "eager": eager, "intent": None, "result": None})
    return out["result"]
