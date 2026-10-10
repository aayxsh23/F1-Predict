"""The chat assistant: dispatches every question to src/agent/router.py,
which answers almost everything deterministically (no model involved, so no
hallucination risk) and hands exactly two intents to the local, fine-tuned
Llama adapter to turn into prose: explain_prediction and general. See
router.py's own docstring for why, and LEARNING.md for the fuller story of
why this replaced an LLM that chose its own tools.

Stateless on purpose: the app sends the recent conversation with every
message, so any server instance can answer, and nothing about a user is
stored.

`stream_chat` yields the events the app renders:
  {"type": "token", "text"}                  answer text as it's generated
  {"type": "tool_start", "id", "name", "args"}  a lookup started
  {"type": "tool_end", "id", "name", "ok"}   ...and finished
  {"type": "sources", "items"}               rules/rulings a lookup cited
  {"type": "done"} or {"type": "error", "message"}
"""
import re
from collections.abc import Iterator

from src.agent.grounding import ungrounded  # noqa: F401  (also re-exported for tests)
from src.agent.router import route
from src.rag.shap_query import EXPLAIN_SYSTEM_PROMPT, build_retrieval_query, build_user_prompt, format_context, retrieve_context

GENERAL_SYSTEM_PROMPT = (
    "You are the F1 Predict app's assistant. Answer the user's question using ONLY the facts given "
    "below. Do not invent drivers, numbers, or facts that aren't given to you. If the facts don't "
    "cover what's asked, say so plainly and suggest asking about a forecast, strategy, standings, a "
    "circuit, or the rules instead. Keep it to 2-4 sentences."
)
GENERAL_FALLBACK = (
    "I can answer questions about forecasts, qualifying, strategy, standings, the title fight, past "
    'results, model accuracy, the calendar, and FIA rules directly. Try one of those, or ask "why is '
    '<driver> predicted there?" for a prediction breakdown.'
)


class ExplainerUnavailable(RuntimeError):
    pass


def _as_shap_features(contributions: list[dict]) -> list[dict]:
    return [{"feature": c["feature"], "shap_value": c["effect"], "phrase": c["input"]} for c in contributions]


def _explain_prompt(exp: dict) -> tuple[str, list[dict]]:
    """The exact prompt shape the LoRA adapter was fine-tuned on (shared with
    rag/explain.py and build_finetune_dataset.py via shap_query.py), built
    from an already-computed SHAP breakdown -- no pandas/XGBoost needed here."""
    features = _as_shap_features(exp["contributions"])
    query = build_retrieval_query(exp["circuit"], exp["predicted"], exp["target"], features)
    hits = retrieve_context(exp["circuit"], query)
    return build_user_prompt(exp["predicted"], exp["target"], exp["circuit"], features, format_context(hits)), hits


def _explain_fallback_text(exp: dict) -> str:
    """What explain_prediction shows when the local model isn't installed:
    the same real numbers, as a plain list instead of a written paragraph."""
    lines = [f"**{exp['driver']}** ({exp['team']}) -- predicted {exp['prediction_of']}: "
             f"{exp['predicted']:.2f} (an average driver here: {exp['average_prediction']:.2f})."]
    for c in exp["contributions"][:5]:
        lines.append(f"- {c['input']}: {c['value']} (effect {c['effect']:+.2f})")
    return "\n".join(lines)


def _general_prompt(facts: dict) -> str:
    lines = [f"Question: {facts['question']}"]
    if "race" in facts:
        lines.append(f"\nCurrent race: {facts['race']}")
        top = facts.get("current_forecast_top3") or []
        if top:
            lines.append("Top of the forecast: " + ", ".join(f"{d['driver']} ({d['team']})" for d in top))
    if "circuit_guide" in facts:
        lines.append(f"\nCircuit notes:\n{facts['circuit_guide'][:1500]}")
    return "\n".join(lines)


def _general_fallback(facts: dict) -> str:
    top = facts.get("current_forecast_top3") or []
    if "race" not in facts or not top:
        return GENERAL_FALLBACK
    front = ", ".join(f"**{d['driver']}** ({d['team']})" for d in top)
    return f"{facts['race']}: the forecast has {front} at the front.\n\n{GENERAL_FALLBACK}"


def _llm_source_items(hits: list[dict]) -> list[dict]:
    return [{"filename": h["source"], "title": h["title"], "article": h["article"], "doc_type": h["doc_type"]} for h in hits]


def stream_chat(history: list[dict], context: dict | None = None) -> Iterator[dict]:
    try:
        result = route(history, context)
    except Exception as exc:
        yield {"type": "error", "message": f"The assistant hit a problem: {type(exc).__name__}"}
        return

    tool, call_id = result["tool"], "r1"
    yield {"type": "tool_start", "id": call_id, "name": tool, "args": {}}
    yield {"type": "tool_end", "id": call_id, "name": tool, "ok": result.get("ok", True)}

    if not result.get("llm"):
        if result.get("sources"):
            yield {"type": "sources", "items": result["sources"]}
        if result.get("text"):
            yield {"type": "token", "text": result["text"]}
        yield {"type": "done"}
        return

    try:
        from src.rag.llm import generate_stream, is_available

        if tool == "explain_prediction":
            system_prompt, (user_prompt, hits) = EXPLAIN_SYSTEM_PROMPT, _explain_prompt(result["llm_payload"])
            if hits:
                yield {"type": "sources", "items": _llm_source_items(hits)}
        else:
            system_prompt, user_prompt = GENERAL_SYSTEM_PROMPT, _general_prompt(result["llm_payload"])

        if not is_available():
            fallback = _explain_fallback_text(result["llm_payload"]) if tool == "explain_prediction" else _general_fallback(result["llm_payload"])
            yield {"type": "token", "text": fallback}
            yield {"type": "done"}
            return

        if tool == "general":
            # short (2-4 sentences), so check it whole before showing it; the
            # explain path rewords its numbers (-4.1479 -> "about 4 places") and
            # streams, its grounding is pinned by the prompt shape instead
            answer = "".join(generate_stream(system_prompt, user_prompt))
            invented = ungrounded(answer, system_prompt + "\n" + user_prompt)
            yield {"type": "token", "text": _general_fallback(result["llm_payload"]) if invented else answer}
            yield {"type": "done"}
            return

        for chunk in generate_stream(system_prompt, user_prompt):
            if chunk:
                yield {"type": "token", "text": chunk}
        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "message": f"The local model hit a problem: {type(exc).__name__}"}


def explain_text(prediction_facts: str) -> str:
    """One short paragraph explaining a prediction (the POST /explain route)."""
    from src.rag.llm import generate, is_available

    if not is_available():
        raise ExplainerUnavailable(
            "The written-explanation feature needs the local model installed on this server "
            "(pip install -r requirements-llm.txt)."
        )
    return generate(EXPLAIN_SYSTEM_PROMPT, prediction_facts).strip()
