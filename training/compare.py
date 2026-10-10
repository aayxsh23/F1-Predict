"""Compare two writers on the same questions: the free code gates, then a
NEUTRAL judge (a third model family) grading every conversation that passed
the code gates, so neither writer is graded by its own family.

    python -m training.compare cmp_qwen cmp_deepseek
"""
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training import quality
from training.bedrock import NEUTRAL, Bedrock

DATA = Path(__file__).resolve().parent / "data"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]


def main() -> None:
    names = sys.argv[1:]
    bedrock = Bedrock(2.0, judge=NEUTRAL)
    rows = {}
    for name in names:
        eps = load(name)
        stats = json.loads((DATA / name / "stats.json").read_text(encoding="utf-8"))
        passed = [e for e in eps if quality.code_checks(e) is None]
        with ThreadPoolExecutor(8) as pool:
            neutral = list(pool.map(lambda e: quality.judge(bedrock, e), passed))
        for e, s in zip(passed, neutral):
            e["neutral"] = s
        ok_neutral = [e for e in passed if quality.passes(e["neutral"])]
        rows[name] = {
            "writer": stats["models"]["writer"], "n": len(eps),
            "code_gate_pass": f"{len(passed)}/{len(eps)}", "neutral_judge_pass": f"{len(ok_neutral)}/{len(eps)}",
            "pipeline_kept (own judge)": f"{stats['kept']}/{len(eps)}",
            "code_rejects": dict(Counter((quality.code_checks(e) or "pass").split(":")[0] for e in eps)),
            "avg_lookups": round(statistics.mean(len(e["calls"]) for e in eps), 2),
            "avg_words": round(statistics.mean(len((e["final"] or "").split()) for e in passed), 1) if passed else None,
            "neutral_mean": {k: round(statistics.mean(e["neutral"][k] for e in passed), 2) for k in ("correct", "grounded", "helpful", "style")} if passed else None,
            "writer_cost_usd": stats["cost"]["spent_usd"],
            "cost_per_neutral_kept_usd": round(stats["cost"]["spent_usd"] / max(1, len(ok_neutral)), 4),
        }
        (DATA / name / f"neutral_scores_{NEUTRAL.split('.')[0]}.json").write_text(json.dumps({e["id"]: e["neutral"] for e in passed}, indent=1), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"\n{'run':22s} {'code-pass':>9s} {'kept':>5s} {'lookups':>7s} {'correct':>7s} {'grounded':>8s} {'helpful':>7s} {'style':>5s} {'$/100':>6s} {'$/kept':>7s}")
    for k, v in rows.items():
        m = v["neutral_mean"] or {}
        print(f"{k:22s} {v['code_gate_pass']:>9s} {v['neutral_judge_pass'].split('/')[0]:>5s} {v['avg_lookups']:>7} {m.get('correct', '-'):>7} {m.get('grounded', '-'):>8} "
              f"{m.get('helpful', '-'):>7} {m.get('style', '-'):>5} {v['writer_cost_usd']:>6} {v['cost_per_neutral_kept_usd']:>7}")
    print(f"neutral judge ({NEUTRAL}) cost: ${bedrock.ledger.spent:.3f}")


if __name__ == "__main__":
    main()
