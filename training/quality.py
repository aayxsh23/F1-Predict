"""Quality gates for a generated conversation, cheapest first:
  1. free checks in code (finished, sane length, no talk about tools, every
     number and name in the answer comes from the question or the tool results)
  2. a judge model (Qwen3-Next 80B, a different family from the writer) that
     scores correctness, grounding, helpfulness and style.
Only conversations that pass both are kept for training."""
import json
import re

from src.agent.grounding import ungrounded
from src.agent.prompts import AGENT_SYSTEM_PROMPT
from training.bedrock import Bedrock

# questions whose answer must come from a lookup; the rest (not F1, news, opinions) may be answered without one
NO_LOOKUP_NEEDED = {"out_of_scope", "unanswerable_news", "opinion"}
TOOL_TALK = re.compile(r"\b(tool|tools|function|json|api|search results?|driver_career|history_results|search_knowledge)\b", re.I)

JUDGE_PROMPT = """You are grading one assistant answer for an F1 chat assistant. The assistant may only state facts that appear in the tool results below.

{earlier}QUESTION:
{question}

TOOL CALLS AND RESULTS (what the assistant was allowed to use):
{evidence}

ASSISTANT ANSWER:
{answer}

Give each of four criteria an INTEGER from 1 (bad) to 5 (excellent), never true/false, and reply with ONLY JSON like this example:
{{"correct": 5, "grounded": 4, "helpful": 5, "style": 3, "problem": "one short sentence, empty string if none"}}
- correct: the answer is right given the results and the question
- grounded: every fact and number comes from the results, nothing added
- helpful: it answers what was asked, or sensibly says it cannot
- style: short, natural, caveats and sources in plain words, no talk of tools

Guidance: a refusal of a non-F1 question, or an honest "I don't have that" for news/rumours/the future, is correct and helpful. Asking which driver when a surname is ambiguous is correct. Stating facts with no tool results is NOT grounded."""


def evidence_text(ep: dict, limit: int = 3500) -> str:
    if not ep["calls"]:
        return "(no tools were called)"
    return "\n".join(f"- {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) -> {c['result'][:limit // max(1, len(ep['calls']))]}" for c in ep["calls"])


def code_checks(ep: dict) -> str | None:
    """None if the conversation passes, else the reason it failed."""
    if not ep["ok"] or not ep["final"]:
        return "no final answer"
    final = ep["final"]
    if not 20 <= len(final) <= 1400:
        return "answer length"
    if not ep["calls"] and ep.get("category") not in NO_LOOKUP_NEEDED:
        return "answered without a lookup"  # includes lazy refusals of questions the data could answer
    if TOOL_TALK.search(final):
        return "talks about tools"
    earlier = ep.get("context") or {}
    # the system prompt as the writer saw it (date filled in), not the template
    system = ep["messages"][0]["content"] if ep.get("messages") and ep["messages"][0].get("role") == "system" else AGENT_SYSTEM_PROMPT
    given = "\n".join([earlier.get("question", ""), earlier.get("answer", ""), ep["question"], *(c["result"] for c in ep["calls"]), system])
    bad = ungrounded(final, given)
    return f"ungrounded: {bad[:4]}" if bad else None


def judge(bedrock: Bedrock, ep: dict) -> dict:
    earlier = ep.get("context")
    prompt = JUDGE_PROMPT.format(question=ep["question"], evidence=evidence_text(ep), answer=ep["final"],
                                 earlier=f"EARLIER IN THIS CONVERSATION:\nUser: {earlier['question']}\nAssistant: {earlier['answer']}\n\n" if earlier else "")
    resp = bedrock.converse(bedrock.judge, "You are a strict, concise grader. Reply with JSON only.", [{"role": "user", "content": [{"text": prompt}]}],
                            max_tokens=1800, temperature=0.0)  # room for a reasoning model to think first; only its text block is read
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    m = re.search(r"\{.*\}", text, re.S)
    try:
        s = json.loads(m.group())
        scores = {k: int(s[k]) if not isinstance(s[k], bool) else 0 for k in ("correct", "grounded", "helpful", "style")}  # a bool means the judge ignored the scale
        if not all(1 <= v <= 5 for v in scores.values()):
            raise ValueError("score out of range")
        return {**scores, "problem": str(s.get("problem", ""))}
    except (AttributeError, KeyError, ValueError, json.JSONDecodeError):
        return {"correct": 0, "grounded": 0, "helpful": 0, "style": 0, "problem": "judge reply unreadable"}


def passes(scores: dict) -> bool:
    return min(scores["correct"], scores["grounded"]) >= 4 and min(scores["helpful"], scores["style"]) >= 3
