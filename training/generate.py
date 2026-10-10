"""Generate the fine-tuning conversations. Same command for the pilot and the
full run; everything is saved as it happens, so a stopped run resumes:

    python -m training.generate --name pilot --n 1000 --cap 6
    python -m training.generate --name full --n 42000 --cap 90

Steps: build the question bank (templated + free-form, de-duplicated, never
overlapping the held-out eval set), give each question a replayed race weekend
(src/agent/context.py) so current-season answers aren't all one forecast, play
each conversation for real, gate it (code checks + judge), then add follow-up
turns to a share of the kept ones. Writes training/data/<name>/:
  questions.jsonl  episodes.jsonl (everything)  kept.jsonl (training format)
  review.md (samples to read)  stats.json (yield per category, cost)
"""
import argparse
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from src.models.refresh_job import OUT_DIR as PREDICTIONS_DIR
from training import evalset, questions, quality
from training.bedrock import QUESTIONER, Bedrock, CapExceeded
from training.episodes import continue_episode
from training.trial import process, review_md

DATA = Path(__file__).resolve().parent / "data"
TEMPLATED_SHARE = 0.32   # of single-turn questions (templated pass ~86%, free-form ~44%: this keeps the kept mix balanced)
FOLLOWUP_SHARE = 0.08    # of all conversations: second turns on kept ones
NO_FOLLOWUP = {"out_of_scope", "unanswerable_news"}
# answers that depend on the replayed weekend: de-duplicated per weekend, not against the eval set
CURRENT = {"tpl_forecast", "tpl_forecast_driver", "tpl_explain", "tpl_h2h", "tpl_standings", "tpl_title_maths", "tpl_strategy", "tpl_schedule",
           "tpl_timeline", "tpl_last_race_model", "tpl_track_record", "current_season"}
FOLLOWUP_PROMPT = """Below are {n} short conversations between a user and a Formula 1 assistant. For each one, write ONE natural follow-up question the same user might ask next, building on the answer (for example "and what about Prost?", "who was second?", "why was that?", "how does that compare to Hamilton?"). Keep them short and casual, like real chat. Output ONLY a JSON array of {n} strings, in the same order.

{conversations}"""


def training_snapshots() -> list[dict]:
    """The race weekends conversations are replayed in: every built snapshot
    (`training.snapshots`), before practice ("pre", the Wednesday) and after
    qualifying ("q", the Saturday evening). The live weekend is NOT one of them:
    the eval set runs on live data, so training on it would leak the answers."""
    snaps = []
    for path in sorted((PREDICTIONS_DIR / "snapshots").glob("*_*_*.json")):
        season, rnd, variant = path.stem.split("_")
        p = json.loads(path.read_text(encoding="utf-8"))
        race_day = date.fromisoformat(p["race_start_utc"][:10])
        today = race_day - timedelta(days=4 if variant == "pre" else 1)
        snaps.append({"season": int(season), "round": int(rnd), "variant": variant, "today": today.isoformat(), "weight": 1,
                      "label": f"{p['season']} {p['location']} ({p['session_label']})"})
    if len(snaps) < 3:
        raise SystemExit("too few weekend snapshots: run `python -m training.snapshots --rounds 2-16` first")
    return snaps


def _snap_fields(snap: dict) -> dict:
    return {k: snap[k] for k in ("season", "round", "today", "variant")}


def _pick_snapshot(rng: random.Random, snaps: list[dict], question: str) -> dict:
    """A question that names 2026 must not be asked from inside 2025."""
    pool = [s for s in snaps if s["season"] == 2026] if "2026" in question else snaps
    return rng.choices(pool, weights=[s["weight"] for s in pool])[0]


def _dedupe_bank(items: list[dict], eval_items: list[dict]) -> list[dict]:
    current = [x for x in items if x["category"] in CURRENT]
    other = [x for x in items if x["category"] not in CURRENT]
    # "who wins this weekend?" is a different conversation on each weekend, so the weekend is part of its identity
    weekend = lambda x: "w{season}{round}{variant}".format(**{"variant": "", **x.get("snapshot", {"season": 0, "round": 0})})  # noqa: E731
    tagged = [{**x, "question": f"{x['question']} {weekend(x)}", "_orig": x["question"]} for x in current]
    kept_current = [{k: v for k, v in x.items() if k != "_orig"} | {"question": x["_orig"]} for x in questions.dedupe(tagged)]
    return kept_current + questions.dedupe(other, against=eval_items)


def balance_from(run: str | None) -> dict[str, float]:
    """Ask for more of the free-form categories a previous run kept less often
    (inverse pass rate, relative to the average, clamped to 1-2.5x), so the kept
    data matches the intended mix instead of over-representing easy categories."""
    if not run:
        return {}
    stats = json.loads((DATA / run / "stats.json").read_text(encoding="utf-8"))["yield_by_category"]
    rates = {c: int(v.split("/")[0]) / max(1, int(v.split("/")[1])) for c, v in stats.items() if c in questions.FREE_FORM}
    mean = sum(rates.values()) / max(1, len(rates))
    return {c: min(2.5, max(1.0, mean / max(r, 0.05))) for c, r in rates.items()}


def build_bank(n_single: int, rng: random.Random, bedrock: Bedrock, snaps: list[dict], boost: dict[str, float] | None = None) -> list[dict]:
    eval_items = evalset.load()
    n_tpl = round(n_single * TEMPLATED_SHARE)
    tpl = _dedupe_bank(questions.templated(int(n_tpl * 1.25), rng, snaps, prefix="t"), eval_items)
    rng.shuffle(tpl)  # shuffle BEFORE trimming: the lists come out grouped by category, and trimming the tail dropped whole categories
    tpl = tpl[:n_tpl]
    free = questions.free_form(int((n_single - n_tpl) * 1.15), bedrock.converse, rng, QUESTIONER, prefix="f", boost=boost, workers=16)
    for x in free:
        x["snapshot"] = _snap_fields(_pick_snapshot(rng, snaps, x["question"]))
    free = _dedupe_bank(free, eval_items)
    rng.shuffle(free)
    free = free[: n_single - n_tpl]
    bank = tpl + free
    rng.shuffle(bank)
    return bank


def _followup_questions(bedrock: Bedrock, eps: list[dict]) -> list[str | None]:
    convo = "\n".join(f"{i + 1}. User: {e['question']}\n   Assistant: {e['final'][:450]}" for i, e in enumerate(eps))
    resp = bedrock.converse(QUESTIONER, "You write realistic follow-up questions. Reply with JSON only.",
                            [{"role": "user", "content": [{"text": FOLLOWUP_PROMPT.format(n=len(eps), conversations=convo)}]}], max_tokens=1200, temperature=0.9)
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    m = re.search(r"\[.*\]", text, re.S)
    try:
        qs = json.loads(m.group()) if m else []
    except json.JSONDecodeError:
        qs = []
    qs = [q.strip() if isinstance(q, str) and 3 < len(q.strip()) < 250 else None for q in qs]
    return (qs + [None] * len(eps))[: len(eps)]


def _process_followup(bedrock: Bedrock, parent: dict, question: str) -> dict:
    try:
        ep = continue_episode(bedrock, parent, question, f"u{parent['id']}")
    except CapExceeded:
        raise
    except Exception as exc:
        return {"id": f"u{parent['id']}", "category": "followup", "question": question, "ok": False, "final": None, "calls": [], "messages": [],
                "reject": f"error: {type(exc).__name__}: {str(exc)[:120]}"}
    ep["reject"] = quality.code_checks(ep)
    if not ep["reject"]:
        ep["scores"] = quality.judge(bedrock, ep)
        if not quality.passes(ep["scores"]):
            ep["reject"] = f"judge: {ep['scores']['problem'] or ep['scores']}"
    return ep


class Store:
    """Append-only episodes file, safe across threads and restarts."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.done = {json.loads(line)["id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else {}

    def add(self, ep: dict) -> None:
        if ep.get("reject"):  # only kept conversations can get a follow-up, so only they need the raw transcript
            ep = {k: v for k, v in ep.items() if k != "bedrock_messages"}
        with self.lock:
            self.done[ep["id"]] = ep
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def _run(pool_items, fn, workers: int, label: str, bedrock: Bedrock, store: Store) -> bool:
    """Run `fn(item)` for every item in parallel, storing results; False if the spending cap stopped it."""
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, item) for item in pool_items]
        try:
            for i, f in enumerate(as_completed(futures), 1):
                store.add(f.result())
                if i % 50 == 0 or i == len(futures):
                    print(f"  {label} {i}/{len(futures)}  spent ${bedrock.ledger.spent:.2f}", flush=True)
        except CapExceeded as exc:
            print(f"STOPPED: {exc}", flush=True)
            for f in futures:
                f.cancel()
            return False
    return True


def _recheck(bedrock: Bedrock, store: Store) -> None:
    """After a checker fix: re-run the code checks on conversations they rejected, and judge the ones that now pass."""
    flips = [e for e in store.done.values() if e.get("ok") and (e.get("reject") or "").split(":")[0] in ("ungrounded", "talks about tools", "answered without a lookup")]
    passed = [e for e in flips if quality.code_checks(e) is None]

    def judge(e: dict) -> dict:
        e = dict(e, reject=None, scores=quality.judge(bedrock, e))
        if not quality.passes(e["scores"]):
            e["reject"] = f"judge: {e['scores']['problem'] or e['scores']}"
        return e

    with ThreadPoolExecutor(8) as pool:
        for e in pool.map(judge, passed):
            store.add(e)
    print(f"recheck: {len(passed)} of {len(flips)} code-rejected conversations now pass the checks; judged", flush=True)


def _stay_awake() -> None:
    """Ask Windows not to sleep while this process runs (released automatically when it exits)."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except (AttributeError, OSError):
        pass  # not Windows


def reassign_weekends(out: Path, rng: random.Random, snaps: list[dict], done: dict) -> None:
    """Spread the not-yet-played weekend-dependent questions over the current
    snapshot list (used when more weekends were built mid-run). Generic ones get
    a new weekend; templated ones name drivers from their weekend's grid, so they
    are regenerated on the new weekends. Everything else is left alone."""
    qpath = out / "questions.jsonl"
    bank = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    (out / "questions_before_reassign.jsonl").write_text(qpath.read_text(encoding="utf-8"), encoding="utf-8")
    unplayed = [x for x in bank if x["category"] in CURRENT and x["id"] not in done]
    free = [x for x in unplayed if not x["category"].startswith("tpl_")]
    tpl_ids = {x["id"] for x in unplayed if x["category"].startswith("tpl_")}
    for x in free:
        x["snapshot"] = _snap_fields(_pick_snapshot(rng, snaps, x["question"]))
    kinds = {c.removeprefix("tpl_") for c in CURRENT if c.startswith("tpl_")}
    fresh = questions.templated(len(tpl_ids) * 2 + 50, rng, snaps, prefix="tw", only=kinds)
    fresh = _dedupe_bank(fresh, [])
    rng.shuffle(fresh)
    fresh = fresh[: len(tpl_ids)]
    bank = [x for x in bank if x["id"] not in tpl_ids] + fresh
    rng.shuffle(bank)
    qpath.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in bank) + "\n", encoding="utf-8")
    per = Counter(f"{x['snapshot']['season']} R{x['snapshot']['round']} {x['snapshot'].get('variant', '')}" for x in bank
                  if x["category"] in CURRENT and x["id"] not in done)
    print(f"reassigned {len(free)} generic and regenerated {len(fresh)} templated weekend questions over {len(snaps)} weekends "
          f"({min(per.values())}-{max(per.values())} each)", flush=True)


def main() -> None:
    _stay_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, required=True, help="conversations to attempt (single-turn + follow-ups)")
    ap.add_argument("--cap", type=float, required=True, help="hard spending cap in USD for this run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--balance-from", help="a previous run whose per-category pass rates set how many of each free-form category to ask for")
    ap.add_argument("--reassign-weekends", action="store_true", help="spread the unplayed weekend questions over all built snapshots")
    ap.add_argument("--retry-errors", action="store_true", help="re-play conversations that crashed")
    ap.add_argument("--recheck", action="store_true", help="re-apply the code checks to saved rejections (judging any that now pass)")
    args = ap.parse_args()
    out = DATA / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    bedrock = Bedrock(args.cap)
    snaps = training_snapshots()
    n_follow = round(args.n * FOLLOWUP_SHARE)

    qpath = out / "questions.jsonl"
    if qpath.exists():
        bank = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    else:
        boost = balance_from(args.balance_from)
        if boost:
            print("free-form boost from", args.balance_from, json.dumps({k: round(v, 2) for k, v in boost.items()}), flush=True)
        bank = build_bank(args.n - n_follow, rng, bedrock, snaps, boost)
        qpath.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in bank) + "\n", encoding="utf-8")
    print(f"{len(bank)} questions over {len(snaps)} weekends: {', '.join(s['label'] for s in snaps)}  (spent ${bedrock.ledger.spent:.2f})", flush=True)

    store = Store(out / "episodes.jsonl")
    if args.reassign_weekends:
        reassign_weekends(out, rng, snaps, store.done)
        bank = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    if args.recheck:
        _recheck(bedrock, store)
    crashed = lambda i: args.retry_errors and (store.done[i].get("reject") or "").startswith("error")  # noqa: E731
    todo = [x for x in bank if x["id"] not in store.done or crashed(x["id"])]
    finished = _run(todo, lambda item: process(bedrock, item, True), args.workers, "conversations", bedrock, store)

    if finished:
        have = {e["id"] for e in store.done.values() if e["category"] == "followup"}
        parents = [e for e in store.done.values() if not e.get("reject") and e["category"] not in NO_FOLLOWUP and e["category"] != "followup"
                   and f"u{e['id']}" not in have and e.get("bedrock_messages")]
        rng.shuffle(parents)
        parents = parents[: max(0, n_follow - len(have))]
        pairs = []
        for i in range(0, len(parents), 8):
            batch = parents[i: i + 8]
            pairs += [(p, q) for p, q in zip(batch, _followup_questions(bedrock, batch)) if q]
        _run(pairs, lambda pq: _process_followup(bedrock, *pq), args.workers, "follow-ups", bedrock, store)

    finish(out, bank, list(store.done.values()), bedrock, snaps)


def _total_spend(out: Path, bedrock: Bedrock) -> float:
    """Spending across every (resumed) invocation of this run, kept in spend.json."""
    path = out / "spend.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append(bedrock.ledger.summary())
    path.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return round(sum(r["spent_usd"] for r in runs), 3)


def finish(out: Path, bank: list[dict], episodes: list[dict], bedrock: Bedrock, snaps: list[dict]) -> None:
    episodes.sort(key=lambda e: e["id"])
    total = _total_spend(out, bedrock)
    kept = [e for e in episodes if not e.get("reject")]
    (out / "kept.jsonl").write_text("\n".join(json.dumps({"id": e["id"], "category": e["category"], "snapshot": e.get("snapshot"), "messages": e["messages"]},
                                                         ensure_ascii=False) for e in kept) + "\n", encoding="utf-8")
    (out / "review.md").write_text(review_md(episodes, per_category=3), encoding="utf-8")
    by_cat = defaultdict(lambda: [0, 0])
    for e in episodes:
        by_cat[e["category"]][0] += 1
        by_cat[e["category"]][1] += int(not e.get("reject"))
    single = [e for e in episodes if e["category"] != "followup"]
    stats = {
        "questions": len(bank), "conversations": len(episodes), "kept": len(kept), "keep_rate": round(len(kept) / max(1, len(episodes)), 3),
        "followups": sum(1 for e in episodes if e["category"] == "followup"), "followups_kept": sum(1 for e in kept if e["category"] == "followup"),
        "templated_kept": f"{sum(1 for e in kept if e['category'].startswith('tpl_'))}/{sum(1 for e in single if e['category'].startswith('tpl_'))}",
        "freeform_kept": f"{sum(1 for e in kept if e['category'] not in ('followup',) and not e['category'].startswith('tpl_'))}/{sum(1 for e in single if not e['category'].startswith('tpl_'))}",
        "reject_reasons": dict(Counter((e.get("reject") or "kept").split(":")[0] for e in episodes)),
        "kept_weekend_versions": len({(e["snapshot"]["season"], e["snapshot"]["round"], e["snapshot"].get("variant", "")) for e in kept
                                      if e["category"] in CURRENT and e.get("snapshot")}),
        "avg_lookups_kept": round(sum(len(e["calls"]) for e in kept) / max(1, len(kept)), 2),
        "yield_by_category": {c: f"{k}/{n}" for c, (n, k) in sorted(by_cat.items())},
        "cost_this_invocation": bedrock.ledger.summary(), "spent_total_usd": total, "cost_per_kept_usd": round(total / max(1, len(kept)), 5),
        "weekends": [s["label"] for s in snaps],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
