"""One training conversation = one question, played out for real: the teacher
model (gpt-oss-120b, see bedrock.py) chooses tools, OUR tools run against the real data, the
teacher writes the answer from the real results. The model we later fine-tune
never sees an invented tool result.

Output messages use the OpenAI-style roles Qwen's chat template expects:
system / user / assistant(tool_calls) / tool / assistant.
"""
import json
import re
import uuid
from contextlib import nullcontext
from datetime import date

from src.agent import context, toolspec
from src.agent.prompts import system_prompt
from training.bedrock import Bedrock
from training.quality import NO_LOOKUP_NEEDED

MAX_ROUNDS = 4  # lookups; on the last round the teacher is told to answer
NUDGE = {"text": "Answer now with what you have. Do not call any more tools."}
TEACHER_ADDENDUM = """

You are writing the ideal assistant turns for training data, so follow these exactly:
- Use tools for anything factual: at most 3 lookups. If something is still missing, answer with what you have and say plainly what you don't have.
- Write ONLY facts, numbers and names that appear in the tool results. Do not add background from your own knowledge, even if you are sure of it. If the results are thin, give a shorter answer. A driver code (like ANT) may only be expanded to a full name if a result gives that name.
- No arithmetic of your own (no sums, rates or percentages you work out); converting a probability such as 0.19 to 19% is fine.
- Mention a caveat when a result carries a note that affects the answer.
- Plain, natural, at most about 110 words (a small table is fine). Never say "tool", "search results", "the data shows", "based on" or mention function names; name the source instead ("from the race results since 1950", "Wikipedia: Monaco Grand Prix").
- If you can't answer (news, rumours, the future, not F1), say so in one or two friendly sentences and say what you can help with. Good: "That's one for a cooking site: I only cover Formula 1. Happy to dig into results, records, rules or this weekend's forecast." Good: "I don't have news or rumours, only results, records, rules and forecasts. I can tell you where he stands in the data, though." Bad: "My tools only cover..." or "the search results don't show..."."""


# gpt-oss likes typographic characters (non-breaking hyphens, curly quotes): the model we train should write plain ones
_PLAIN = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
                        " ": " ", " ": " ", " ": " ", "​": ""})


def plain(text: str) -> str:
    return text.translate(_PLAIN)


def _clean_use(use: dict) -> dict:
    """gpt-oss occasionally garbles a call ("functions.records", a stray token, a
    string instead of an object). Bedrock then rejects the whole conversation when
    it is sent back, so repair it here: a name that still isn't a real tool
    becomes "unknown_tool", which our runner answers with an error the model can
    recover from."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "", use.get("name", "").split(".")[-1].split("<")[0])
    use["name"] = name if name in toolspec.REGISTRY else "unknown_tool"
    args = use.get("input")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    use["input"] = {k: v for k, v in args.items() if k} if isinstance(args, dict) else {}  # gpt-oss sends {"": {}} for no-argument tools
    return use


def bedrock_tools() -> list[dict]:
    return [{"toolSpec": {"name": s["function"]["name"], "description": s["function"]["description"],
                          "inputSchema": {"json": s["function"]["parameters"]}}} for s in toolspec.schemas()]


def _replay(item: dict):
    """The weekend the conversation happens in: a replayed snapshot, or now."""
    snap = item.get("snapshot")
    return context.as_of(snap["season"], snap["round"], date.fromisoformat(snap["today"]), snap.get("variant", "")) if snap else nullcontext()


def _play(bedrock: Bedrock, item: dict, messages: list[dict], train: list[dict], calls: list[dict], force_first_lookup: bool) -> str | None:
    """Run the lookup loop from the current point of a conversation; returns the
    final answer (None if the writer never stopped looking things up)."""
    tools = bedrock_tools()
    teacher_system = system_prompt(context.today()) + TEACHER_ADDENDUM

    def run_calls(uses: list[dict], text: str) -> None:
        train.append({"role": "assistant", "content": text, "tool_calls": [
            {"type": "function", "function": {"name": u["name"], "arguments": u["input"]}} for u in uses]})
        results = []
        for u in uses:
            res = toolspec.call_tool(u["name"], u["input"])
            body = toolspec.format_result(res)
            calls.append({"name": u["name"], "arguments": u["input"], "ok": res["ok"], "result": body})
            train.append({"role": "tool", "name": u["name"], "content": body})
            results.append({"toolResult": {"toolUseId": u["toolUseId"], "content": [{"text": body}], **({} if res["ok"] else {"status": "error"})}})
        messages.append({"role": "user", "content": results})

    if item.get("gold"):  # the right first lookup is known: start from it
        # 9 letters/digits: the strictest format any provider requires (Mistral rejects anything else)
        uses = [{"toolUseId": uuid.uuid4().hex[:9], "name": g["name"], "input": g["arguments"]} for g in item["gold"]]
        messages.append({"role": "assistant", "content": [{"toolUse": u} for u in uses]})
        run_calls(uses, "")

    for round_no in range(MAX_ROUNDS + 1):
        if round_no == MAX_ROUNDS:  # tool config must stay attached while the history holds tool calls
            messages[-1]["content"] = messages[-1]["content"] + [NUDGE]
        # a question that needs data and starts with no known lookup: the writer must look something up first
        force = round_no == 0 and not item.get("gold") and item["category"] not in NO_LOOKUP_NEEDED and force_first_lookup
        resp = bedrock.converse(bedrock.writer, teacher_system, messages, tools=tools, max_tokens=700, temperature=0.3, force_tool=force)
        content = resp["output"]["message"]["content"]
        text = plain("".join(b.get("text", "") for b in content if "text" in b).strip())
        uses = [_clean_use(b["toolUse"]) for b in content if "toolUse" in b]
        if not uses:
            if round_no == MAX_ROUNDS:  # the student never sees the nudge: drop it from the kept history
                messages[-1]["content"] = messages[-1]["content"][:-1]
            return text or None
        if round_no == MAX_ROUNDS:
            return None  # still wanted more lookups after the nudge: unfinished, dropped
        messages.append({"role": "assistant", "content": [b for b in content if "text" in b or "toolUse" in b]})
        run_calls(uses, text)
    return None


def run_episode(bedrock: Bedrock, item: dict, force_first_lookup: bool = False) -> dict:
    """Play one question. Returns the episode with `messages` (training format),
    `calls` (name, arguments, ok), `final`, `ok` (False if it never finished) and
    `bedrock_messages` (the raw conversation, so a follow-up can continue it)."""
    with _replay(item):
        messages = [{"role": "user", "content": [{"text": item["question"]}]}]
        train = [{"role": "system", "content": system_prompt(context.today())}, {"role": "user", "content": item["question"]}]
        calls: list[dict] = []
        final = _play(bedrock, item, messages, train, calls, force_first_lookup)
    if final:
        train.append({"role": "assistant", "content": final})
    return {**{k: v for k, v in item.items() if k != "gold"}, "gold_tools": [g["name"] for g in item.get("gold", [])],
            "calls": calls, "final": final, "ok": bool(final), "messages": train, "bedrock_messages": messages}


def continue_episode(bedrock: Bedrock, ep: dict, followup: str, new_id: str) -> dict:
    """A second user turn on a finished conversation ("and what about Prost?").
    The model sees the whole first exchange, lookups included, and may answer
    from them or look more up."""
    item = {"id": new_id, "category": "followup", "question": followup, **({"snapshot": ep["snapshot"]} if ep.get("snapshot") else {})}
    with _replay(item):
        messages = ep["bedrock_messages"] + [{"role": "assistant", "content": [{"text": ep["final"]}]},
                                             {"role": "user", "content": [{"text": followup}]}]
        train = ep["messages"] + [{"role": "user", "content": followup}]
        calls = list(ep["calls"])
        final = _play(bedrock, item, messages, train, calls, force_first_lookup=False)
    if final:
        train.append({"role": "assistant", "content": final})
    return {**item, "gold_tools": [], "parent": ep["id"], "context": {"question": ep["question"], "answer": ep["final"]},
            "calls": calls, "new_calls": len(calls) - len(ep["calls"]), "final": final, "ok": bool(final), "messages": train,
            "bedrock_messages": messages}
