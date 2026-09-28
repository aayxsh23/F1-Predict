"""The chat assistant: a LangGraph tool-calling agent (langchain's
create_agent) on Google Gemini, with the tools in tools.py.

Stateless on purpose: the app sends the recent conversation with every
message, so any server instance (a cold serverless function included) can
answer, and nothing about a user is stored.

`stream_chat` yields the events the app renders:
  {"type": "token", "text"}                  answer text as it's generated
  {"type": "tool_start", "id", "name", "args"}  a lookup started
  {"type": "tool_end", "id", "name", "ok"}   ...and finished
  {"type": "sources", "items"}               rules/rulings a lookup cited
  {"type": "done"} or {"type": "error", "message"}
"""
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from src.agent.tools import TOOLS

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_HISTORY = 16  # messages kept from the conversation the app sends
MAX_CHARS = 4000  # per message

SYSTEM_PROMPT = """You are the race analyst inside Pit Wall, an F1 prediction app. You explain a \
machine-learning forecast to fans: clearly enough for a casual viewer, precisely enough for an engineer.

Today is {today} (UTC). {context}

How to answer
- Lead with the answer in one or two plain sentences. Then the evidence: short bullets, or a compact \
Markdown table when comparing drivers (at most 8 rows). Add a caveat only when it changes how much to \
trust the answer.
- Every number you state (chances, positions, lap times, gaps, points, laps, dates) must come from a \
tool result in this conversation. Never estimate or recall one from memory. If no tool covers it, say \
the app doesn't have that data.
- Use tools before answering anything about forecasts, standings, strategy, rules, results or accuracy. \
Call several at once when a question needs them (say, the forecast and the explanation).
- Speak like a fan: "grid slot", "recent form", "tyre wear", "pit window". Mention SHAP, XGBoost or \
calibration only when asked how the model works, and then be exact.
- A chance is not a promise: write "about 1 in 3 (34%)". Say when a forecast is from before qualifying \
and so less certain.
- Cite rules as (Article B1.6.2, Sporting Regulations) and steward decisions by race and car number.
- Drivers appear as three-letter codes (VER, NOR); use full names when a tool gives them.
- For "how accurate is this", use model_track_record and give the held-out error next to the baseline.
- Stay on Formula 1 and this app. No betting advice. No headings in short answers."""

EXPLAIN_PROMPT = """You explain one prediction from an F1 forecasting model to a fan, in 3 to 5 \
sentences of plain English. Use only the facts given: the prediction, the inputs that moved it (with \
their effect), and the circuit and rules context. Say which inputs mattered most and in which \
direction, connected to what they mean on track. Do not invent numbers, incidents or rule articles. \
No lists, no headings."""


class ChatNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _model():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ChatNotConfigured("The assistant isn't configured: set GEMINI_API_KEY on the server.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODEL, api_key=key, temperature=0.3, max_retries=2)


@wrap_tool_call
def _tool_errors_to_model(request, handler):
    """A failed lookup (unknown driver, no forecast for that race) goes back to
    the model as a message it can recover from, instead of ending the chat."""
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(content=f"Lookup failed: {exc}", tool_call_id=request.tool_call["id"],
                           name=request.tool_call["name"], status="error")


def _context_line(context: dict | None) -> str:
    c = context or {}
    if not c.get("season") or not c.get("round"):
        return "The user hasn't picked a race; use the current race weekend."
    parts = [f"The user is looking at season {c['season']} round {c['round']}"
             + (f" ({c['race_name']})" if c.get("race_name") else "")]
    if c.get("driver"):
        parts.append(f"with {c['driver']} selected")
    return " ".join(parts) + ". Use these when they say \"this race\" or \"he\"."


def build_agent(model=None, context: dict | None = None):
    prompt = SYSTEM_PROMPT.format(today=datetime.now(timezone.utc).strftime("%A %d %B %Y"), context=_context_line(context))
    return create_agent(model or _model(), tools=TOOLS, system_prompt=prompt, middleware=[_tool_errors_to_model])


def to_messages(history: list[dict]) -> list:
    msgs = []
    for m in history[-MAX_HISTORY:]:
        text = str(m.get("content", ""))[:MAX_CHARS]
        msgs.append(HumanMessage(text) if m.get("role") == "user" else AIMessage(text))
    return msgs


def stream_chat(history: list[dict], context: dict | None = None, model=None) -> Iterator[dict]:
    try:
        agent = build_agent(model, context)
        for mode, chunk in agent.stream({"messages": to_messages(history)}, stream_mode=["messages", "updates"]):
            if mode == "messages":
                msg, _meta = chunk
                if isinstance(msg, AIMessageChunk) and (text := str(msg.text)):
                    yield {"type": "token", "text": text}
                continue
            for update in chunk.values():
                for m in (update or {}).get("messages", []):
                    if isinstance(m, AIMessage):
                        for tc in m.tool_calls:
                            yield {"type": "tool_start", "id": tc["id"], "name": tc["name"], "args": tc["args"]}
                    elif isinstance(m, ToolMessage):
                        yield {"type": "tool_end", "id": m.tool_call_id, "name": m.name, "ok": m.status != "error"}
                        if m.artifact:
                            yield {"type": "sources", "items": m.artifact}
        yield {"type": "done"}
    except ChatNotConfigured as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:  # the stream has already started; report instead of dropping the connection
        yield {"type": "error", "message": f"The assistant hit a problem: {type(exc).__name__}"}


def explain_text(prediction_facts: str, model=None) -> str:
    """One short paragraph explaining a prediction (the "Why" tab)."""
    reply = (model or _model()).invoke([SystemMessage(EXPLAIN_PROMPT), HumanMessage(prediction_facts)])
    return str(reply.text).strip()
