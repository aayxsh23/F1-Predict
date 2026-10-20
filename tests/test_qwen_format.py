"""The chat model's prompt format: rendering matches Qwen3.5's official chat
template exactly (needs `transformers` and network access to fetch the
template once; skipped otherwise), and replies parse back into typed calls."""
import pytest

from src.agent import toolspec
from src.rag.qwen_format import parse_reply, render, segments

CONVERSATION = [
    {"role": "system", "content": "You are Pit Radio. Today is Monday 19 October 2026."},
    {"role": "user", "content": "How many wins did Senna get at Monaco?"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "records", "arguments": {"metric": "wins", "driver": "Ayrton Senna", "circuit": "Monaco", "n": 3}}}]},
    {"role": "tool", "name": "records", "content": '{"leaders":[{"name":"Ayrton Senna","wins":6}]}'},
    {"role": "assistant", "content": "Six, a record (from the race results since 1950)."},
    {"role": "user", "content": "and Prost?"},
    {"role": "assistant", "content": "Let me check.", "tool_calls": [{"type": "function", "function": {"name": "driver_career", "arguments": {"driver": "Prost"}}},
                                                                     {"type": "function", "function": {"name": "records", "arguments": {"metric": "wins", "circuit": "Monaco"}}}]},
    {"role": "tool", "name": "driver_career", "content": '{"wins":51}'},
    {"role": "tool", "name": "records", "content": '{"leaders":[]}'},
    {"role": "assistant", "content": "Prost won there four times."},
]


def test_matches_official_template():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception as exc:  # offline
        pytest.skip(f"template unavailable: {exc}")
    tools = toolspec.schemas()
    assert render(CONVERSATION, tools, add_generation_prompt=False) == tok.apply_chat_template(CONVERSATION, tools=tools, tokenize=False)
    assert render(CONVERSATION[:6], tools) == tok.apply_chat_template(CONVERSATION[:6], tools=tools, tokenize=False,
                                                                       add_generation_prompt=True, enable_thinking=False)


def test_only_the_current_turn_is_trainable():
    trained = "".join(t for t, learn in segments(CONVERSATION, toolspec.schemas()) if learn)
    assert "Prost won there four times." in trained and "<function=driver_career>" in trained
    assert "Six, a record" not in trained and "<think>" not in trained and "<tool_response>" not in trained


def test_parse_reply_types_arguments():
    reply = ("Checking.\n\n<tool_call>\n<function=history_results>\n<parameter=season>\n1988\n</parameter>\n"
             "<parameter=grand_prix>\nMonaco\n</parameter>\n</function>\n</tool_call><|im_end|>")
    text, calls = parse_reply(reply, toolspec.schemas())
    assert text == "Checking."
    assert calls == [{"name": "history_results", "arguments": {"season": 1988, "grand_prix": "Monaco"}}]
    assert parse_reply("Just an answer.<|im_end|>", toolspec.schemas()) == ("Just an answer.", [])
