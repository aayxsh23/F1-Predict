"""Generate a small batch of training conversations end to end, on-demand (not
batch mode), so the result can be read before the full run is paid for.

    python -m training.trial --n 300 --cap 5

Writes training/data/trial/: questions.jsonl, episodes.jsonl (everything),
kept.jsonl (passed every gate: the training-format conversations), review.md
(a readable sample for a human), stats.json (funnel, cost, per-category yield).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training import questions, quality
from training.bedrock import JUDGE, QUESTIONER, WRITER, Bedrock, CapExceeded
from training.episodes import run_episode

OUT = Path(__file__).resolve().parent / "data" / "trial"


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def process(bedrock: Bedrock, item: dict, force_first_lookup: bool = False) -> dict:
    try:
        ep = run_episode(bedrock, item, force_first_lookup)
    except CapExceeded:
        raise
    except Exception as exc:  # one bad conversation must not stop the batch
        return {**{k: v for k, v in item.items() if k != "gold"}, "ok": False, "final": None, "calls": [], "messages": [],
                "reject": f"error: {type(exc).__name__}: {str(exc)[:120]}"}
    ep["reject"] = quality.code_checks(ep)
    if not ep["reject"]:
        ep["scores"] = quality.judge(bedrock, ep)
        if not quality.passes(ep["scores"]):
            ep["reject"] = f"judge: {ep['scores']['problem'] or ep['scores']}"
    return ep


def review_md(eps: list[dict], per_category: int = 2) -> str:
    by_cat = defaultdict(list)
    for e in eps:
        by_cat[e["category"]].append(e)
    lines = ["# Trial conversations (a sample, to read)\n"]
    for cat in sorted(by_cat):
        for e in by_cat[cat][:per_category]:
            status = "KEPT" if not e["reject"] else f"DROPPED: {e['reject']}"
            lines.append(f"## [{cat}] {e['question']}\n**{status}**  scores: {e.get('scores')}\n")
            for c in e["calls"]:
                lines.append(f"- call `{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})` {'ok' if c['ok'] else 'ERROR'}: `{c['result'][:160]}`")
            lines.append(f"\n> {e['final']}\n")
    return "\n".join(lines)


def finish(bedrock: Bedrock, bank: list[dict], episodes: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes.sort(key=lambda e: e["id"])

    kept = [e for e in episodes if not e["reject"]]
    _write(OUT / "episodes.jsonl", episodes)
    _write(OUT / "kept.jsonl", [{"id": e["id"], "category": e["category"], "messages": e["messages"]} for e in kept])
    (OUT / "review.md").write_text(review_md(episodes), encoding="utf-8")
    yield_by_cat = {c: f"{sum(1 for e in episodes if e['category'] == c and not e['reject'])}/{sum(1 for e in episodes if e['category'] == c)}"
                    for c in sorted({e["category"] for e in episodes})}
    stats = {"questions": len(bank), "episodes": len(episodes), "kept": len(kept), "reject_reasons": Counter((e["reject"] or "kept").split(":")[0] for e in episodes),
             "yield_by_category": yield_by_cat, "cost": bedrock.ledger.summary(), "models": {"writer": bedrock.writer, "judge": bedrock.judge}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=dict))



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cap", type=float, default=5.0, help="hard spending cap in USD")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--writer", default=WRITER)
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--questions", help="reuse an existing questions.jsonl instead of generating a bank")
    ap.add_argument("--limit", type=int, help="with --questions: a random sample of this many")
    ap.add_argument("--out", help="output directory name under training/data/ (default: trial)")
    ap.add_argument("--force-first-lookup", action="store_true", help="make the writer look something up first on questions that need data")
    ap.add_argument("--recheck", action="store_true", help="re-run the gates on the saved episodes instead of generating new ones")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(__file__).resolve().parent / "data" / args.out
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    bedrock = Bedrock(args.cap, writer=args.writer, judge=args.judge)

    if args.recheck:
        episodes = [json.loads(line) for line in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        bank = episodes
        for ep in episodes:
            if ep.get("ok") and (ep["reject"] or "").split(":")[0] in ("ungrounded", "talks about tools"):  # only what the gate change could flip
                ep["reject"] = quality.code_checks(ep)
                if not ep["reject"]:
                    ep["scores"] = quality.judge(bedrock, ep)
                    if not quality.passes(ep["scores"]):
                        ep["reject"] = f"judge: {ep['scores']['problem'] or ep['scores']}"
        finish(bedrock, bank, episodes)
        return
    if args.questions:
        bank = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines()]
        for item in bank:  # the saved bank keeps `gold`; episodes are rebuilt from scratch
            item.pop("calls", None)
        if args.limit:
            bank = random.Random(args.seed).sample(bank, args.limit)
    else:
        n_tpl = round(args.n * 0.4)
        bank = questions.templated(n_tpl, rng) + questions.free_form(args.n - n_tpl, bedrock.converse, rng, QUESTIONER)
        rng.shuffle(bank)
    _write(OUT / "questions.jsonl", bank)
    print(f"{len(bank)} questions. cost so far ${bedrock.ledger.spent:.3f}", flush=True)

    episodes = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, bedrock, item, args.force_first_lookup) for item in bank]
        try:
            for i, f in enumerate(as_completed(futures), 1):
                episodes.append(f.result())
                if i % 25 == 0:
                    print(f"  {i}/{len(bank)}  spent ${bedrock.ledger.spent:.3f}", flush=True)
        except CapExceeded as exc:
            print(f"STOPPED: {exc}", flush=True)
            for f in futures:
                f.cancel()
    finish(bedrock, bank, episodes)


if __name__ == "__main__":
    main()
