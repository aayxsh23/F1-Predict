"""The chat model's toolbelt (src/agent/toolspec.py): every schema matches the
function it names, and call_tool turns every mistake a model can make into a
readable error instead of an exception. No model involved. Run with pytest."""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import toolspec


def test_every_schema_matches_its_function():
    for name, spec in toolspec.REGISTRY.items():
        fn, schema = spec["fn"], spec["schema"]["function"]
        params = inspect.signature(fn).parameters
        assert schema["name"] == fn.__name__ == name
        assert schema["description"] and len(schema["description"]) < 330, f"{name}: description should be short"
        props, required = schema["parameters"]["properties"], schema["parameters"]["required"]
        assert set(props) == set(params), f"{name}: schema args {set(props)} != function args {set(params)}"
        assert set(required) == {p for p, v in params.items() if v.default is inspect.Parameter.empty}, f"{name}: required args"
        for p in props.values():
            assert p.get("type") in {"string", "integer"} or "enum" in p


def test_the_schemas_stay_small_enough_to_pay_for_on_every_turn():
    assert 14 <= len(toolspec.schemas()) <= 17  # fewer, clearer tools for a small model
    assert len(json.dumps(toolspec.schemas())) < 12_000  # ~3k tokens


def test_mistakes_come_back_as_errors_not_exceptions():
    assert toolspec.call_tool("no_such_tool", {})["ok"] is False
    assert "unknown argument" in toolspec.call_tool("forecast", {"bad": 1})["error"]
    assert "missing required" in toolspec.call_tool("driver_career", {})["error"]
    assert toolspec.call_tool("forecast", "{not json")["ok"] is False
    assert toolspec.call_tool("forecast", {"season": "x" * 5, "round": 1})["ok"] is False  # tool raised: reported, not thrown


def test_null_arguments_are_dropped_and_json_strings_accepted():
    r = toolspec.call_tool("season_schedule", '{"season": null}')
    assert r["ok"] and r["result"]["rounds"]


def test_tool_errors_keep_their_candidates_so_the_model_can_retry():
    from src.agent.history import HISTORY_DIR

    if not (HISTORY_DIR / "races.json").exists():
        return
    r = toolspec.call_tool("driver_career", {"driver": "Schumacher"})
    assert r["ok"] is False and r["candidates"]


def test_knowledge_search_finds_the_named_page_first():
    from src.rag.corpus import WIKI_INDEX_PATH

    if not WIKI_INDEX_PATH.exists():
        return  # built by `python -m src.data.wiki`
    for query, title in (("who is Max Verstappen", "Max Verstappen"), ("Monaco Grand Prix history", "Monaco Grand Prix"), ("what is DRS", "Drag reduction system")):
        hits = toolspec.call_tool("search_knowledge", {"query": query, "k": 3})["result"]["hits"]
        assert hits[0]["title"] == title and hits[0]["source"].startswith("https://en.wikipedia.org/wiki/")
        assert hits[0]["license"] == "CC BY-SA 4.0"  # attribution travels with the text
    assert toolspec.call_tool("search_knowledge", {"query": "blorptastic zzz"})["result"]["hits"] == []


def test_long_results_are_trimmed_to_valid_json_with_a_note():
    big = {"ok": True, "result": {"season": 2026, "rows": [{"driver": "X" * 40, "points": i} for i in range(300)]}}
    text = toolspec.format_result(big)
    parsed = json.loads(text)  # still valid JSON, never cut mid-string
    assert len(text) <= toolspec.MAX_RESULT_CHARS + 200 and parsed["season"] == 2026
    assert "rows" in parsed["_truncated"] and parsed["rows"][0]["points"] == 0


def test_a_replayed_weekend_never_sees_later_results():
    from datetime import date

    from src.agent import context
    from src.agent.history import HISTORY_DIR

    if not (HISTORY_DIR / "races.json").exists():
        return
    with context.as_of(2026, 16, date(2026, 9, 30)):  # the Wednesday before Kuala Lumpur
        races = toolspec.call_tool("history_results", {"season": 2026})["result"]["races"]
        assert max(r["round"] for r in races) == 15
        assert toolspec.call_tool("season_schedule", {})["result"]["next_round"] == 16
        assert toolspec.call_tool("forecast", {})["result"]["round"] == 16
        assert "hasn't been raced" in toolspec.call_tool("past_race_prediction", {"season": 2026, "round": 16})["error"]


def test_the_system_prompt_carries_the_date():
    from datetime import date

    from src.agent.prompts import system_prompt

    assert "Today is Wednesday 30 September 2026." in system_prompt(date(2026, 9, 30))
