"""Deterministic chat router.

A LangGraph StateGraph with two real nodes: `classify` matches the latest
question to one of a fixed set of intents by keyword (the same discipline
the project used before -- see LEARNING.md's chapter on why a weak local
model can't be trusted to pick its own tool), and `dispatch` calls the one
matching function in tools.py and renders a plain-English answer from its
real, already-correct result. Neither node calls an LLM, and every number in
every answer traces to a real lookup -- there is nothing here for a model to
hallucinate, because no model is involved.

Two intents hand their result to the local Llama adapter afterwards, outside
this module (see chat.py): `explain_prediction`, to turn a SHAP breakdown
into prose (the model's own fine-tuned use case), and `general`, a
best-effort answer when nothing else matched, with an honest caveat. Every
other intent works even where the local model isn't installed at all.
"""
import re
from functools import lru_cache
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


def _circuit_in(question_lower: str) -> str | None:
    for name in _known_circuits():
        if name.lower() in question_lower:
            return name
    return None


def _target_in(question_lower: str) -> str:
    if any(w in question_lower for w in ("qualify", "quali", "pole", "grid slot")):
        return "qualifying"
    if any(w in question_lower for w in ("places gained", "place gained", "grid to finish", "overtake", "positions gained")):
        return "quali_delta"
    if any(w in question_lower for w in ("gap to the winner", "gap to winner", "race time", "behind the winner", "time gap")):
        return "race_time"
    return "finish_position"


def _by_name_or_code(drivers: list[dict], question: str, code_key: str = "driver", name_key: str = "name") -> dict | None:
    """Find a driver entry by an uppercase code in the question, else a surname match."""
    codes = {d[code_key] for d in drivers}
    for code in _codes_in(question):
        if code in codes:
            return next(d for d in drivers if d[code_key] == code)
    ql = question.lower()
    for d in drivers:
        surname = d.get(name_key, "").split()[-1].lower() if d.get(name_key) else ""
        if surname and surname in ql:
            return d
    return None


# --- intent classification: pure keyword matching, no network, no LLM ---

def _classify_specific(q: str) -> str | None:
    if any(p in q for p in ("need to do to win", "clinch", "mathematically", "still in contention", "win the title", "win the championship")):
        return "title_scenario"
    if any(p in q for p in ("constructor", "team standings", "team's points", "team points", "constructors championship",
                             "constructors' championship", "team is leading", "teams are leading")):
        return "constructor_standings"
    if any(p in q for p in ("championship standing", "driver standings", "drivers' championship", "points total",
                             "how many points", "leading the championship", "championship lead", "who's leading",
                             "who is leading", "championship position")):
        return "driver_standings"
    if any(p in q for p in ("strategy", "pit stop", "pit window", "tyre", "tire", "stint", "undercut", "overcut")):
        return "race_strategy"
    if _ARTICLE_RE.search(q) or any(p in q for p in ("regulation", " rule", "rules say", "penalty", "steward",
                                                       "article", "disqualif", "infringement", "unsafe release", "legal")):
        return "search_rules"
    if any(p in q for p in ("how accurate", "track record", "how good is the model", "how reliable", "trust this",
                             "margin of error", "is this any good")):
        return "model_track_record"
    if any(p in q for p in ("predicted for", "predicted before", "what did you predict", "called that race", "got that race right")):
        return "past_race_prediction"
    if any(p in q for p in ("timeline", "changed through the weekend", "moved through the weekend", "since practice",
                             "since fp1", "since fp2", "since fp3")):
        return "forecast_timeline"
    if any(p in q for p in ("schedule", "calendar", "next race", "upcoming race", "when is the", "when's the")):
        return "season_schedule"
    if any(p in q for p in (" vs ", " versus ", "head to head", "head-to-head", "beat ", "ahead of", " against ")):
        return "head_to_head"
    if any(p in q for p in ("why", "explain", "reason", "what's pushing", "what is pushing", "what's driving",
                             "what is driving", "break down", "breakdown")):
        return "explain_prediction"
    if any(p in q for p in ("what's", "what is", "tell me about", "circuit guide", "track guide", "layout", "like at")) and _circuit_in(q):
        return "circuit_guide"
    if any(p in q for p in ("history", "results at", "how has", "how did", "past results", "previous results")):
        return "driver_history"
    if any(p in q for p in ("win", "favourite", "favorite", "podium", "who wins", "pole", "qualify", "quali",
                             "grid", "forecast", "predict", "finish", "chance", "odds")):
        return "race_forecast"
    return None


def classify_intent(messages: list[dict]) -> str:
    """The intent for the latest message, falling back to the previous
    user turn's intent for an explicit continuation ("and what about...") --
    a real discourse signal, not just "nothing else matched"."""
    question = (messages[-1].get("content") or "").lower().strip()
    specific = _classify_specific(question)
    if specific:
        return specific
    if question.startswith(CONTINUATION) and len(messages) > 1:
        prev = next((m["content"] for m in reversed(messages[:-1]) if m.get("role") == "user"), None)
        if prev:
            inherited = _classify_specific(prev.lower().strip())
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
    data = t.forecast(season, round_)
    pool = [d["driver"] for d in data["drivers"]]
    found = _codes_in(question)
    code = next((c for c in found if c in pool), None) or ctx.get("driver")
    if not code:
        raise ValueError('Tell me which driver -- e.g. "why is NOR predicted there?"')
    exp = t.explain_prediction(code, target=_target_in(question.lower()), season=season, round=round_)
    return {"tool": "explain_prediction", "llm": True, "llm_payload": exp, "sources": None}


def _handle_head_to_head(question: str, ctx: dict) -> dict:
    season, round_ = ctx.get("season"), ctx.get("round")
    codes = _codes_in(question)
    if len(codes) < 2:
        raise ValueError('Name two drivers by their 3-letter codes -- e.g. "NOR vs VER".')
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
    named = _by_name_or_code(drivers, question)
    if named:
        lead = (f"**{named['name']}** ({named['team']}) is P{named['position']} with {named['points']:.0f} points and "
                f"{named['wins']} win{'s' if named['wins'] != 1 else ''}. Title chance: {_pctfmt(named['title_chance'])}.")
    else:
        lead = f"**{drivers[0]['name']}** leads the drivers' championship with {drivers[0]['points']:.0f} points."
    rows = [{"Pos": d["position"], "Driver": d["driver"], "Team": d["team"], "Points": d["points"], "Title": _pctfmt(d["title_chance"])} for d in drivers[:8]]
    return {"tool": "championship", "text": f"{lead}\n\n{_md_table(rows)}", "sources": None}


def _handle_title_scenario(question: str, ctx: dict) -> dict:
    data = t.championship(kind="drivers")
    named = _by_name_or_code(data["drivers"], question)
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
    data = t.forecast(season, round_)
    pool = [d["driver"] for d in data["drivers"]]
    code = next((c for c in _codes_in(question) if c in pool), None) or ctx.get("driver")
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


def _handle_driver_history(question: str, ctx: dict) -> dict:
    codes = _codes_in(question)
    code = codes[0] if codes else ctx.get("driver")
    if not code:
        raise ValueError("Tell me which driver (3-letter code) to look up.")
    circuit = _circuit_in(question.lower())
    results = t.driver_history(code, circuit)
    if not results:
        raise ValueError(f"No results found for {code}{' at ' + circuit if circuit else ''}.")
    def _pos(v):
        return None if v is None else int(v)

    rows = [{"Race": f"{r['season']} R{r['round']} {r['location']}", "Grid": _pos(r.get("grid_position")),
             "Finish": _pos(r.get("finish_position")), "Status": r.get("status")} for r in results]
    return {"tool": "driver_history", "text": f"**{code}**'s last {len(results)} result(s){' at ' + circuit if circuit else ''}:\n\n{_md_table(rows)}", "sources": None}


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
    "driver_history": _handle_driver_history,
    "past_race_prediction": _handle_past_race_prediction,
    "model_track_record": _handle_model_track_record,
    "season_schedule": _handle_season_schedule,
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
    intent: str | None
    result: dict | None


def _classify_node(state: RouteState) -> dict:
    return {"intent": classify_intent(state["messages"])}


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


def route(messages: list[dict], context: dict | None = None) -> dict:
    """The one entry point chat.py calls: classify the latest message and
    run its handler. Returns {tool, ok, sources, and either text (ready to
    show) or llm=True + llm_payload (for chat.py to hand to the local model)}."""
    out = _graph().invoke({"messages": messages, "context": context or {}, "intent": None, "result": None})
    return out["result"]
