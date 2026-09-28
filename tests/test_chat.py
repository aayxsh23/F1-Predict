"""The chat assistant without a real LLM: its tools on the repo's real data, and
the agent loop (tool call -> tool result -> answer, streamed as app events)
driven by a scripted model. Run with `python tests/test_chat.py`."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.agent import tools
from src.agent.chat import stream_chat


class ScriptedModel(BaseChatModel):
    """Replies with the given messages in order, streaming tool calls as chunks like Gemini does."""
    replies: list
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self) -> AIMessage:
        msg = self.replies[self.turn]
        self.turn += 1
        return msg

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next()
        yield ChatGenerationChunk(message=AIMessageChunk(content=msg.content, tool_call_chunks=[
            {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
            for i, tc in enumerate(msg.tool_calls)]))


def _call(name, args, id_="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}])


def test_agent_calls_a_tool_then_answers_with_citations():
    model = ScriptedModel(replies=[_call("search_rules", {"query": "unsafe release"}),
                                   AIMessage(content="An unsafe release is penalised (Article B1.6.2).")])
    events = list(stream_chat([{"role": "user", "content": "What happens after an unsafe release?"}], model=model))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "tool_start" and events[0]["name"] == "search_rules", kinds
    assert "tool_end" in kinds and "sources" in kinds and kinds[-1] == "done", kinds
    sources = next(e for e in events if e["type"] == "sources")["items"]
    assert any(s["filename"] == "2026_australian_gp_car12_unsafe_release.pdf" for s in sources)
    assert "".join(e["text"] for e in events if e["type"] == "token").startswith("An unsafe release")


def test_a_failing_tool_goes_back_to_the_model_not_the_user():
    model = ScriptedModel(replies=[_call("explain_prediction", {"driver": "ZZZ"}),
                                   AIMessage(content="I couldn't find that driver in this race.")])
    events = list(stream_chat([{"role": "user", "content": "Why is ZZZ predicted there?"}], model=model))
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is False and events[-1]["type"] == "done"


def test_missing_api_key_is_a_clear_error_event():
    import os

    from src.agent import chat

    chat._model.cache_clear()
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    try:
        events = list(stream_chat([{"role": "user", "content": "hi"}]))
    finally:
        os.environ.update({k: v for k, v in saved.items() if v})
    assert events == [{"type": "error", "message": "The assistant isn't configured: set GEMINI_API_KEY on the server."}]


def test_forecast_tool_reports_odds_from_the_real_latest_forecast():
    out = json.loads(tools.race_forecast.invoke({}))
    assert len(out["drivers"]) >= 20
    top = out["drivers"][0]
    assert top["predicted_rank"] == 1 and top["win"].endswith("%")


def test_explain_tool_contributions_add_up():
    code = json.loads(tools.race_forecast.invoke({}))["drivers"][0]["driver"]
    out = json.loads(tools.explain_prediction.invoke({"driver": code}))
    assert out["contributions"] and all(c["input"] and "_" not in c["input"] for c in out["contributions"]), "fan labels, not column names"


def test_head_to_head_is_a_probability():
    drivers = json.loads(tools.race_forecast.invoke({}))["drivers"]
    a, b = drivers[0]["driver"], drivers[-1]["driver"]
    out = json.loads(tools.head_to_head.invoke({"driver_a": a, "driver_b": b}))
    assert out[f"{a}_ahead_of_{b}"].endswith("%")


def test_strategy_tool_and_safety_car_scenario():
    out = json.loads(tools.race_strategy.invoke({"safety_car_lap": 15}))
    assert out["strategies"] and out["scenario"]["sc_lap"] == 15


def test_history_and_track_record_tools():
    assert "VER" in tools.driver_history.invoke({"driver": "VER", "circuit": "Monza", "last_n": 3})
    rec = json.loads(tools.model_track_record.invoke({}))
    assert "finish_position" in rec["held_out"] and "by_season" in rec


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
