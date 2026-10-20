"""The evaluation gate for the fine-tuned chat model (phase 9, step 5): the
held-out questions, played through the app's own lookup loop
(src.agent.chat.agent_answer: raw Ollama, the shared prompt format, the
grounding check), so what is measured is what ships. Scored like the
student bake-off: the same code gates as the training data, one judge
(Kimi K2.5) for every model, the expected first lookup on templated
questions, and the traps.

    python -m training.eval_chat --models qwen3.5:9b-q4_K_M f1-analyst [--n 200]

Ships only if f1-analyst beats the untuned base on right-tool and grounding
(phase9-open-chat-plan.md section 8). Results: training/data/eval_chat/.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from src.agent import toolspec
from src.agent.chat import agent_answer
from src.rag import llm
from training import quality
from training.bedrock import NEUTRAL, Bedrock
from training.student_bakeoff import sample, summarise

OUT = Path(__file__).resolve().parent / "data" / "eval_chat"


def play(model: str, item: dict) -> dict:
    calls, real = [], toolspec.call_tool

    def recording(name, args):
        res = real(name, args)
        calls.append({"name": name, "arguments": args, "ok": res["ok"], "result": toolspec.format_result(res)})
        return res

    t0, error, answer = time.time(), None, None
    try:
        with patch.object(llm, "MODEL", model), patch("src.agent.chat.toolspec.call_tool", side_effect=recording):
            answer = next(e for e in agent_answer([{"role": "user", "content": item["question"]}]) if e["type"] == "answer")
    except Exception as exc:  # a model failing is a result, not a reason to stop
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
    final = answer["text"] if answer else None
    ep = {**{k: v for k, v in item.items() if k != "gold"}, "model": model, "calls": calls, "final": final, "ok": bool(final),
          "messages": [], "wall_s": round(time.time() - t0, 1), "gen_tokens": 0, "gen_s": 0.0, "error": error,
          "tool_as_text": bool(final and "<tool_call>" in final), "grounded_in_app": bool(answer and answer["grounded"])}
    if item.get("gold"):
        ep["expected_tool"] = item["gold"][0]["name"]
        ep["picked_expected_tool"] = bool(calls) and calls[0]["name"] == ep["expected_tool"]
    return ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--judge-cap", type=float, default=3.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = sample(args.n)
    bedrock = Bedrock(args.judge_cap, judge=NEUTRAL)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for model in args.models:
        path = OUT / f"{model.replace(':', '_')}.jsonl"
        done = {json.loads(line)["id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else {}
        t0 = time.time()
        for i, item in enumerate(items, 1):  # one at a time: the laptop runs one model call at once anyway
            if item["id"] in done:
                continue
            ep = play(model, item)
            ep["reject"] = quality.code_checks(ep) if not ep["error"] else f"error: {ep['error']}"
            if not ep["reject"]:
                ep["scores"] = quality.judge(bedrock, ep)
            done[item["id"]] = ep
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {model}: {i}/{len(items)}", flush=True)
        changed = 0
        for ep in done.values():  # the code checks may have changed since an answer was scored: re-apply them
            reject = quality.code_checks(ep) if not ep["error"] else ep["reject"]
            if reject != ep["reject"]:
                ep["reject"], changed = reject, changed + 1
                if reject:
                    ep.pop("scores", None)
                elif not ep.get("scores"):
                    ep["scores"] = quality.judge(bedrock, ep)
        if changed:
            path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in done.values()), encoding="utf-8")
            print(f"  {model}: {changed} saved answers re-scored", flush=True)
        eps = [done[x["id"]] for x in items]
        summary[model] = {**summarise(eps), "grounded_in_app": sum(e["grounded_in_app"] for e in eps),
                          "minutes_this_run": round((time.time() - t0) / 60, 1) or summary.get(model, {}).get("minutes_this_run")}
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"{model}: {json.dumps(summary[model])}", flush=True)
    print(f"judge cost: ${bedrock.ledger.spent:.2f}")


if __name__ == "__main__":
    main()
