"""The chat model's toolbelt: one registry that is the single source of truth
for the tool schemas the model sees, the training data generator, the agent
loop and the tests. `call_tool` validates the arguments the model wrote and
returns an error dict (never an exception) the model can read and recover from.

Descriptions are deliberately short and say WHEN to use each tool: that text is
what a 7B model picks tools from, and every token of it is paid on every turn.
"""
import inspect
import json
from typing import Any, Callable

from src.agent import chat_tools, history, tools

_STR, _INT = {"type": "string"}, {"type": "integer"}


def _spec(fn: Callable, description: str, props: dict[str, dict], required: list[str] | None = None) -> dict:
    return {"fn": fn, "schema": {"type": "function", "function": {
        "name": fn.__name__, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}}


_TARGET = {"type": "string", "enum": ["finish_position", "qualifying", "quali_delta", "race_time"],
           "description": "which prediction: finishing position, qualifying gap, places gained, gap to winner"}
_SEASON = {"season": {**_INT, "description": "year; omit for the current race weekend"}, "round": {**_INT, "description": "round number; omit for the current race weekend"}}

REGISTRY: dict[str, dict] = {s["schema"]["function"]["name"]: s for s in [
    # --- history, 1950 onwards (computed from the results table) ---
    _spec(history.history_results, "Results of past races: one race's full classification (season + round, or season + grand_prix), a season's races and winners (season only), or every winner of a Grand Prix (grand_prix only).",
          {"season": _INT, "round": _INT, "grand_prix": {**_STR, "description": "e.g. 'Monza', 'Italian Grand Prix', 'Monaco'"}}),
    _spec(history.driver_career, "A driver's whole career: starts, wins, poles, podiums, titles, teams and a season-by-season table. Use for any 'how many... did <driver>' question.",
          {"driver": {**_STR, "description": "name, surname or 3-letter code, e.g. 'Hamilton', 'Ayrton Senna', 'VER'"}}, ["driver"]),
    _spec(history.team_history, "A constructor's record: seasons, wins, poles, podiums, constructors' titles and its top winners.",
          {"team": {**_STR, "description": "e.g. 'Ferrari', 'McLaren', 'Red Bull'"}}, ["team"]),
    _spec(history.season_summary, "One season: champions, final standings, race winners and wins per driver.", {"year": _INT}, ["year"]),
    _spec(history.records, "Computed leaderboards ('who has the most...', 'how many wins at Monza'). ALWAYS use this for superlatives and counts, never answer them from memory.",
          {"metric": {"type": "string", "enum": list(history.METRICS)}, "by": {"type": "string", "enum": list(history.GROUPS), "description": "what to rank (default driver)"},
           "n": {**_INT, "description": "how many leaders (default 10)"}, "driver": _STR, "team": _STR, "circuit": {**_STR, "description": "circuit or Grand Prix name"},
           "from_year": _INT, "to_year": _INT}, ["metric"]),
    _spec(tools.search_knowledge, "Search written sources for explanations and background: Wikipedia, the 2026 FIA rules and steward rulings (with article numbers), and circuit notes. Not for numbers.",
          {"query": {**_STR, "description": "specific keywords"}, "k": {**_INT, "description": "hits (default 5)"},
           "only": {"type": "string", "enum": ["rules", "wikipedia", "circuit_notes"], "description": "search one source only"}}, ["query"]),
    # --- this season's forecast, standings and strategy (our models) ---
    _spec(chat_tools.forecast, "The forecast for a race weekend: predicted top 10 with win, podium, points and pole chances, and race facts. Pass driver for one driver's full forecast.",
          {**_SEASON, "driver": {**_STR, "description": "3-letter code, for one driver's full forecast"}}),
    _spec(tools.explain_prediction, "WHY the model predicts what it does for one driver: the inputs that moved the prediction, with their effect.",
          {"driver": {**_STR, "description": "3-letter code, e.g. 'ANT'"}, "target": _TARGET, **_SEASON}, ["driver"]),
    _spec(tools.head_to_head, "How often driver A finishes ahead of driver B in simulated races, and their expected finishes.",
          {"driver_a": _STR, "driver_b": _STR, **_SEASON}, ["driver_a", "driver_b"]),
    _spec(tools.race_strategy, "Best tyre strategies and pit windows for a race, optionally with a safety car on a given lap.", {"safety_car_lap": _INT, **_SEASON}),
    _spec(tools.forecast_timeline, "How the forecast moved through the weekend (after each session).", {"driver": _STR, **_SEASON}),
    _spec(tools.championship, "The current championship standings with simulated title chances.",
          {"kind": {"type": "string", "enum": ["drivers", "constructors"]}}),
    _spec(tools.title_scenario_for, "What a driver needs to win this season's title.", {"driver": _STR}, ["driver"]),
    _spec(tools.season_schedule, "A season's calendar: rounds, venues, dates, sprint weekends, which race is next.", {"season": _INT}),
    _spec(tools.past_race_prediction, "What the model predicted for a finished race (before it) next to the real result.", {"season": _INT, "round": _INT}, ["season", "round"]),
    _spec(chat_tools.model_track_record, "How accurate the prediction models have been on races they never saw, against a simple guess.", {}),
]}  # search_rules, circuit_guide and driver_history are folded into search_knowledge / the history tools (fewer, clearer choices)


def schemas() -> list[dict]:
    """The tool list in the OpenAI-style format Qwen's chat template takes."""
    return [s["schema"] for s in REGISTRY.values()]


def _check(fn: Callable, args: dict) -> str | None:
    sig = inspect.signature(fn)
    unknown = set(args) - set(sig.parameters)
    if unknown:
        return f"unknown argument(s) {sorted(unknown)}; allowed: {list(sig.parameters)}"
    missing = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty and n not in args]
    return f"missing required argument(s) {missing}" if missing else None


MAX_RESULT_CHARS = 3500  # ~1k tokens: a small model's context window is shared by instructions, tools, every result and the answer


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _longest_list(obj, path: str = "") -> tuple[str, list | None]:
    """The list (anywhere in the result) that takes the most characters."""
    best: tuple[int, str, list | None] = (0, "", None)
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else ()
    if isinstance(obj, list) and len(obj) > 1:
        best = (len(_dumps(obj)), path or "result", obj)
    for key, value in items:
        if isinstance(value, (dict, list)):
            sub_path, sub = _longest_list(value, f"{path}.{key}" if path else str(key))
            if sub is not None and len(_dumps(sub)) > best[0]:
                best = (len(_dumps(sub)), sub_path, sub)
    return best[1], best[2]


def format_result(res: dict) -> str:
    """A tool outcome as the compact JSON string the model reads. Identical in
    training, evaluation and the running app. Too long: the longest list is
    halved (repeatedly) and a note says what was cut, so the JSON stays valid and
    the model knows it is seeing part of the answer."""
    obj = json.loads(_dumps(res.get("result") if res.get("ok") else {k: v for k, v in res.items() if k != "ok"}))
    notes: dict[str, str] = {}
    text = _dumps(obj)
    while len(text) > MAX_RESULT_CHARS:
        path, lst = _longest_list(obj)
        if lst is None:
            return text[:MAX_RESULT_CHARS] + '..."[cut]"'
        total = int(notes[path].split(" of ")[1].split()[0]) if path in notes else len(lst)
        del lst[max(1, len(lst) // 2):]
        notes[path] = f"showing {len(lst)} of {total} (the rest cut to fit)"
        text = _dumps({**obj, "_truncated": notes} if isinstance(obj, dict) else {"result": obj, "_truncated": notes})
    return text


def call_tool(name: str, args: dict[str, Any] | str | None) -> dict:
    """Run one tool call. Always returns {"ok": bool, "result"|"error": ...}."""
    if name not in REGISTRY:
        return {"ok": False, "error": f"unknown tool '{name}'; available: {sorted(REGISTRY)}"}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "arguments must be a JSON object"}
    args = {k: v for k, v in (args or {}).items() if v is not None}
    fn = REGISTRY[name]["fn"]
    if problem := _check(fn, args):
        return {"ok": False, "error": problem}
    try:
        result = fn(**args)
    except Exception as exc:  # a tool failing is information for the model, not a crash
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    if isinstance(result, dict) and "error" in result:
        return {"ok": False, "error": result["error"], **{k: v for k, v in result.items() if k in ("candidates",)}}
    return {"ok": True, "result": result}
