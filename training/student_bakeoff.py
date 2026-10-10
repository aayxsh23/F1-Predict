"""Compare candidate STUDENT models (the ones we could fine-tune and run on the
laptop), untuned, exactly as the app would run them: our system prompt, our
16 tools, Ollama, the same compressed (Q4) files the laptop would use. The
Ollama server can be local or an EC2 GPU box reached through an SSH tunnel;
answers are identical, only speed differs.

    python -m training.student_bakeoff --models qwen3:14b-q4_K_M qwen2.5:7b --ollama http://localhost:11435

Questions: a fixed sample of the held-out eval set (all traps + a stratified
rest). Scored with the same code gates as the training data and ONE judge
(Kimi K2.5) for every model, plus: did it pick the expected first tool
(templated questions), did it print a tool call as text instead of calling it,
did it ever finish.
"""
import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agent import context, toolspec
from src.agent.prompts import system_prompt
from training import evalset, quality
from training.bedrock import NEUTRAL, Bedrock
from training.episodes import plain

OUT = Path(__file__).resolve().parent / "data" / "student_bakeoff"
MAX_ROUNDS = 4
NUM_CTX = 12288
THINKING_FAMILIES = ("qwen3", "qwen3.5")  # hybrid-thinking models: run them in non-thinking mode, as the app would
NUDGE = "Answer now with what you have. Do not call any more tools."


def sample(n: int, seed: int = 7) -> list[dict]:
    items = evalset.load()
    traps = [x for x in items if x["category"].startswith("trap_")]
    rest = [x for x in items if not x["category"].startswith("trap_")]
    by_cat = defaultdict(list)
    for x in rest:
        by_cat[x["category"]].append(x)
    rng, picked = random.Random(seed), []
    while len(picked) < n - len(traps) and any(by_cat.values()):  # round-robin over categories: every kind is represented
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < n - len(traps):
                picked.append(by_cat[cat].pop(rng.randrange(len(by_cat[cat]))))
    return traps + picked


def _chat(url: str, model: str, messages: list[dict], tools: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "tools": tools, "stream": False, "options": {"num_ctx": NUM_CTX, "temperature": 0.2}}
    if model.split(":")[0] in THINKING_FAMILIES:
        body["think"] = False
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(30):  # the SSH tunnel can drop for a few seconds (laptop IP change); wait and retry rather than lose the answer
        try:
            return json.load(urllib.request.urlopen(req, timeout=600))
        except (ConnectionError, urllib.error.URLError) as exc:
            if attempt == 29:
                raise
            time.sleep(10)


def _looks_like_tool_text(text: str) -> bool:
    t = text.lower()
    return "<tool_call>" in t or ('"name"' in t and ('"arguments"' in t or '"parameters"' in t)) or "[tool_calls]" in t


def play(url: str, model: str, item: dict) -> dict:
    tools = toolspec.schemas()
    today = context.today()
    messages = [{"role": "system", "content": system_prompt(today)}, {"role": "user", "content": item["question"]}]
    calls, final, gen_tokens, gen_s, t0 = [], None, 0, 0.0, time.time()
    try:
        for round_no in range(MAX_ROUNDS + 1):
            if round_no == MAX_ROUNDS:
                messages.append({"role": "user", "content": NUDGE})
            r = _chat(url, model, messages, tools)
            gen_tokens += r.get("eval_count", 0)
            gen_s += r.get("eval_duration", 0) / 1e9
            msg = r["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = plain((msg.get("content") or "").strip())
                break
            if round_no == MAX_ROUNDS:
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
                res = toolspec.call_tool(name, args)
                body = toolspec.format_result(res)
                calls.append({"name": name, "arguments": args, "ok": res["ok"], "result": body})
                messages.append({"role": "tool", "content": body, "tool_name": name})
        error = None
    except Exception as exc:  # a model crashing or timing out is a result, not a reason to stop the bake-off
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
    ep = {**{k: v for k, v in item.items() if k != "gold"}, "model": model, "calls": calls, "final": final, "ok": bool(final),
          "messages": messages, "wall_s": round(time.time() - t0, 1), "gen_tokens": gen_tokens, "gen_s": round(gen_s, 2), "error": error,
          "tool_as_text": bool(final and _looks_like_tool_text(final))}
    if item.get("gold"):
        ep["expected_tool"] = item["gold"][0]["name"]
        ep["picked_expected_tool"] = bool(calls) and calls[0]["name"] == ep["expected_tool"]
    return ep


def summarise(eps: list[dict]) -> dict:
    tpl = [e for e in eps if "expected_tool" in e]
    traps = [e for e in eps if e["category"].startswith("trap_")]
    judged = [e for e in eps if e.get("scores")]
    kept = [e for e in judged if quality.passes(e["scores"])]
    mean = lambda k: round(statistics.mean(e["scores"][k] for e in judged), 2) if judged else None  # noqa: E731
    return {
        "questions": len(eps), "finished": sum(e["ok"] for e in eps), "errors": sum(bool(e["error"]) for e in eps),
        "tool_call_printed_as_text": sum(e["tool_as_text"] for e in eps),
        "picked_expected_tool": f"{sum(e['picked_expected_tool'] for e in tpl)}/{len(tpl)}",
        "avg_lookups": round(statistics.mean(len(e["calls"]) for e in eps), 2),
        "code_checks_passed": len(judged), "judge_passed": len(kept), "judge_pass_rate": round(len(kept) / max(1, len(eps)), 3),
        "traps_passed": f"{sum(1 for e in traps if e in kept)}/{len(traps)}",
        "mean_scores": {k: mean(k) for k in ("correct", "grounded", "helpful", "style")},
        "code_rejects": dict(Counter((e.get("reject") or "pass").split(":")[0] for e in eps)),
        "median_wall_s_on_this_server": statistics.median(e["wall_s"] for e in eps),
        "gen_tokens_per_s_on_this_server": round(sum(e["gen_tokens"] for e in eps) / max(1e-9, sum(e["gen_s"] for e in eps)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--judge-cap", type=float, default=4.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = sample(args.n)
    bedrock = Bedrock(args.judge_cap, judge=NEUTRAL)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for model in args.models:
        path = OUT / f"{model.replace(':', '_').replace('/', '_')}.jsonl"
        if model in summary:
            print(f"{model}: already done, skipped", flush=True)
            continue
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            eps = list(pool.map(lambda it: play(args.ollama, model, it), items))
        for e in eps:
            e["reject"] = quality.code_checks(e) if not e["error"] else f"error: {e['error']}"
        with ThreadPoolExecutor(8) as pool:
            for e, s in zip([e for e in eps if not e["reject"]], pool.map(lambda e: quality.judge(bedrock, e), [e for e in eps if not e["reject"]])):
                e["scores"] = s
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in eps) + "\n", encoding="utf-8")
        summary[model] = {**summarise(eps), "minutes": round((time.time() - t0) / 60, 1)}
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"{model}: {json.dumps(summary[model])}", flush=True)
    print(f"judge cost: ${bedrock.ledger.spent:.2f}")


if __name__ == "__main__":
    main()
