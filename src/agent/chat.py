"""The chat assistant. Two tiers:

1. The keyword router (src/agent/router.py) answers most questions
   deterministically: no model, nothing to invent.
2. What it can't answer (`general`) and the written "why" behind a prediction
   (`explain_prediction`) go to the local fine-tuned model (src/rag/llm.py,
   Ollama) when this machine runs it: the model picks lookups from the
   toolbelt (src/agent/toolspec.py), Python runs them, the model writes the
   answer from the results. Every answer then passes the grounding check: a
   number or name no lookup returned replaces it with the plain data answer.
   Without the model (the serverless API) both intents degrade to
   deterministic answers, never an error.

Stateless on purpose: the app sends the recent conversation with every
message, so any server instance can answer, and nothing about a user is
stored.

`stream_chat` yields the events the app renders:
  {"type": "token", "text"}                  answer text
  {"type": "tool_start", "id", "name", "args"}  a lookup started
  {"type": "tool_end", "id", "name", "ok"}   ...and finished
  {"type": "sources", "items"}               rules/rulings/articles a lookup cited
  {"type": "done"} or {"type": "error", "message"}
"""
import logging
import re
from collections.abc import Iterator
from functools import lru_cache

from src.agent import context as clock
from src.agent import toolspec
from src.agent.grounding import ungrounded  # noqa: F401  (also re-exported for tests)
from src.agent.prompts import system_prompt
from src.agent.router import route

MAX_LOOKUP_ROUNDS = 4  # then one more turn with NUDGE; the eval's unfinished answers all hit this limit refining searches
NUDGE = "Answer now with what you have. Do not call any more tools."  # the words the teacher got (training/episodes.py)
HISTORY_MESSAGES = 6
GENERAL_FALLBACK = (
    "I can answer questions about forecasts, qualifying, strategy, standings, the title fight, driver ratings, past "
    'results, model accuracy, the calendar, and FIA rules directly. Try one of those, or ask "why is '
    '<driver> predicted there?" for a prediction breakdown.'
)
log = logging.getLogger("f1.chat")
_TOPIC_WORDS = re.compile(r"[a-z]+")


def _explain_fallback_text(exp: dict) -> str:
    """explain_prediction without the model: the same real numbers as a list."""
    lines = [f"**{exp['driver']}** ({exp['team']}) -- predicted {exp['prediction_of']}: "
             f"{exp['predicted']:.2f} (an average driver here: {exp['average_prediction']:.2f})."]
    for c in exp["contributions"][:5]:
        lines.append(f"- {c['input']}: {c['value']} (effect {c['effect']:+.2f})")
    return "\n".join(lines)


def _general_fallback(facts: dict) -> str:
    top = facts.get("current_forecast_top3") or []
    if "race" not in facts or not top:
        return GENERAL_FALLBACK
    front = ", ".join(f"**{d['driver']}** ({d['team']})" for d in top)
    return f"{facts['race']}: the forecast has {front} at the front.\n\n{GENERAL_FALLBACK}"


@lru_cache(maxsize=1)
def _vocab() -> frozenset[str]:
    text = " ".join(f"{s['function']['name']} {s['function']['description']}" for s in toolspec.schemas())
    return frozenset(w for w in _TOPIC_WORDS.findall(text.replace("_", " ").lower()) if len(w) > 3)


def _count_unanswered(question: str) -> None:
    """Which F1 topics fall through to the catch-all, to pick the next router
    intents. Logs only words from the app's own tool descriptions (never the
    question itself), so nothing a user typed is stored."""
    words = sorted({w for w in _TOPIC_WORDS.findall(question.lower()) if w in _vocab()})
    log.info("chat_unmatched topics=%s", ",".join(words) or "-")


def _sources(name: str, res: dict) -> list[dict]:
    if name != "search_knowledge" or not res.get("ok"):
        return []
    return [{"filename": h["source"], "title": h["title"], "article": h.get("article"), "doc_type": h["doc_type"]}
            for h in res["result"]["hits"]]


def agent_answer(history: list[dict], first_calls: list[dict] | None = None) -> Iterator[dict]:
    """The model's lookup loop for the latest question. `first_calls` are
    lookups already known to be right (the router resolved them), played as
    the model's first turn, as in the training data's templated questions.
    Yields tool/source events, then exactly one {"type": "answer", "text",
    "grounded"}."""
    from src.rag import llm
    from src.rag.qwen_format import parse_reply, render

    tools = toolspec.schemas()
    system = system_prompt(clock.today())
    msgs = [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m.get("content") or ""} for m in history[-HISTORY_MESSAGES:] if m.get("role") in ("user", "assistant")]
    evidence = [system] + [m["content"] for m in msgs[1:]]
    n = 0

    def run(calls: list[dict], text: str = "") -> Iterator[dict]:
        nonlocal n
        msgs.append({"role": "assistant", "content": text, "tool_calls": [{"type": "function", "function": c} for c in calls]})
        for c in calls:
            n += 1
            yield {"type": "tool_start", "id": f"a{n}", "name": c["name"], "args": c["arguments"]}
            res = toolspec.call_tool(c["name"], c["arguments"])
            body = toolspec.format_result(res)
            evidence.append(body)
            msgs.append({"role": "tool", "name": c["name"], "content": body})
            yield {"type": "tool_end", "id": f"a{n}", "name": c["name"], "ok": res["ok"]}
            if items := _sources(c["name"], res):
                yield {"type": "sources", "items": items}

    if first_calls:
        yield from run(first_calls)
    for round_no in range(MAX_LOOKUP_ROUNDS + 1):
        last = round_no == MAX_LOOKUP_ROUNDS
        # out of lookups: told to answer, as the teacher was when the training data was written
        text, calls = parse_reply(llm.generate(render(msgs + [{"role": "user", "content": NUDGE}] if last else msgs, tools)), tools)
        if not calls or last:
            bad = ungrounded(text, "\n".join(evidence)) if text else ["(empty)"]
            if bad:
                log.info("chat_ungrounded %s", bad[:4])
            yield {"type": "answer", "text": text, "grounded": not bad}
            return
        yield from run(calls, text)


def stream_chat(history: list[dict], context: dict | None = None) -> Iterator[dict]:
    from src.rag import llm

    available = llm.is_available()
    try:
        result = route(history, context, eager=not available)
    except Exception as exc:
        yield {"type": "error", "message": f"The assistant hit a problem: {type(exc).__name__}"}
        return

    tool = result["tool"]
    if not result.get("llm"):
        yield {"type": "tool_start", "id": "r1", "name": tool, "args": {}}
        yield {"type": "tool_end", "id": "r1", "name": tool, "ok": result.get("ok", True)}
        if result.get("sources"):
            yield {"type": "sources", "items": result["sources"]}
        if result.get("text"):
            yield {"type": "token", "text": result["text"]}
        yield {"type": "done"}
        return

    payload = result["llm_payload"]
    fallback = _explain_fallback_text(payload) if tool == "explain_prediction" else _general_fallback(payload)
    if tool == "general":
        _count_unanswered(history[-1].get("content") or "")
    try:
        if not available:
            yield {"type": "tool_start", "id": "r1", "name": tool, "args": {}}
            yield {"type": "tool_end", "id": "r1", "name": tool, "ok": True}
            yield {"type": "token", "text": fallback}
            yield {"type": "done"}
            return
        first = None
        if tool == "explain_prediction":
            first = [{"name": "explain_prediction", "arguments": {"driver": payload["driver"], "target": payload["target"],
                                                                 "season": payload["season"], "round": payload["round"]}}]
        for event in agent_answer(history, first):
            if event["type"] == "answer":
                yield {"type": "token", "text": event["text"] if event["grounded"] else fallback}
            else:
                yield event
        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "message": f"The local model hit a problem: {type(exc).__name__}"}


class ExplainerUnavailable(RuntimeError):
    pass


def explain_text(season: int, round: int, driver: str, target: str) -> tuple[str, list[str]]:
    """One short paragraph on why the model predicts what it does for a
    driver (POST /explain), written by the local model from the stored SHAP
    breakdown, and the documents it searched; raises ExplainerUnavailable
    where the model isn't running."""
    from src.rag import llm

    if not llm.is_available():
        raise ExplainerUnavailable("Written explanations need the local model (Ollama with f1-analyst) running on this server.")
    question = [{"role": "user", "content": f"Why is {driver} predicted where he is ({target.replace('_', ' ')})?"}]
    first = [{"name": "explain_prediction", "arguments": {"driver": driver, "target": target, "season": season, "round": round}}]
    events = list(agent_answer(question, first))
    answer = events[-1]
    if not answer["grounded"]:
        raise ExplainerUnavailable("The written explanation didn't pass the fact check; the breakdown above is exact.")
    return answer["text"], [s["filename"] for e in events if e["type"] == "sources" for s in e["items"]]
