"""Does the chat's question classifier hold up on unusual wording, and do
reworded training questions fix it? (2026-10-21)

The classifier (experiments/router_classifier.py) learned from generated
questions, which are tidier than what people type. Here a writer model
(DeepSeek V3.2 on Bedrock) rewrites a sample of training questions three
ways each: casual or misspelled, indirect, second-language or slang. A
rewording keeps its source question's label. Questions are split by SOURCE,
so no rewording of a test question is trained on. Both classifiers are
scored on the reworded test questions:

    python -m experiments.router_paraphrases            # generate (cached, ~$0.5), compare
    python -m experiments.router_paraphrases --export   # ship: refit on everything, write router_model.json

Rewordings are cached in training/data/router_paraphrases.jsonl.
"""
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.router_classifier import MODEL_PATH, load, pipeline, score
from training.bedrock import QUESTIONER, Bedrock

CACHE = Path(__file__).resolve().parents[1] / "training" / "data" / "router_paraphrases.jsonl"
PER_LABEL = 160   # source questions per fixed intent (fewer if a label has fewer); "general" gets as many as all of them
PROMPT = """Rewrite each question below three ways, the way real people type into an F1 app. Keep the exact meaning and any names, years or numbers:
1. casual: lowercase, short, maybe a typo or abbreviation
2. indirect: a roundabout way of asking the same thing
3. second-language or slang: simple non-native English, or fan slang

Questions:
{questions}

Reply with ONLY a JSON array with one array of three strings per question, in order."""


def _split(text: str) -> bool:
    """True for the test fifth, by the source question."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 5 == 0


def sources(rows: list[dict]) -> list[dict]:
    rng = random.Random(7)
    by = defaultdict(list)
    for r in rows:
        by[r["y"]].append(r)
    fixed = [y for y in by if y != "general"]
    picked = []
    for y in fixed:
        rng.shuffle(by[y])
        picked += by[y][:PER_LABEL]
    rng.shuffle(by["general"])
    return picked + by["general"][: PER_LABEL * len(fixed) // 2]


def generate(rows: list[dict], cap: float = 1.0) -> list[dict]:
    done = [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines()] if CACHE.exists() else []
    seen = {d["source"] for d in done}
    todo = [r for r in sources(rows) if r["q"] not in seen]
    if not todo:
        return done
    bedrock = Bedrock(cap)

    def ask(batch: list[dict]) -> list[dict]:
        listing = "\n".join(f"{i + 1}. {r['q']}" for i, r in enumerate(batch))
        resp = bedrock.converse(QUESTIONER, "You rewrite questions. Reply with JSON only.",
                                [{"role": "user", "content": [{"text": PROMPT.format(questions=listing)}]}], max_tokens=2500, temperature=0.9)
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        m = re.search(r"\[.*\]", text, re.S)
        try:
            out = json.loads(m.group()) if m else []
        except json.JSONDecodeError:
            return []
        return [{"source": r["q"], "y": r["y"], "q": p} for r, ps in zip(batch, out) if isinstance(ps, list)
                for p in ps if isinstance(p, str) and 3 < len(p) < 300]

    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    with ThreadPoolExecutor(8) as pool, CACHE.open("a", encoding="utf-8") as f:
        for got in pool.map(ask, batches):
            for g in got:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            done += got
    print(f"rewordings: {len(done)} (spent ${bedrock.ledger.spent:.2f})", flush=True)
    return done


def main() -> None:
    rows = load()
    para = generate(rows)
    train = [r for r in rows if not _split(r["q"])]
    test_para = [p for p in para if _split(p["source"])]
    test_orig = [r for r in rows if _split(r["q"]) and not r["templated"]]
    before = pipeline().fit([r["q"] for r in train], [r["y"] for r in train])
    extra = [p for p in para if not _split(p["source"])]
    after = pipeline().fit([r["q"] for r in train] + [p["q"] for p in extra], [r["y"] for r in train] + [p["y"] for p in extra])
    out = {}
    for name, clf in (("without rewordings", before), ("with rewordings", after)):
        out[name] = {"reworded test questions": score(list(clf.predict([p["q"] for p in test_para])), [p["y"] for p in test_para]),
                     "original free-form test questions": score(list(clf.predict([r["q"] for r in test_orig])), [r["y"] for r in test_orig])}
    print(json.dumps(out, indent=1))


def export() -> None:
    """Refit on every question and every rewording; write what router.py scores."""
    rows = load()
    para = generate(rows)
    clf = pipeline().fit([r["q"] for r in rows] + [p["q"] for p in para], [r["y"] for r in rows] + [p["y"] for p in para])
    vec, lr = clf[0], clf[-1]
    MODEL_PATH.write_text(json.dumps({
        "classes": list(lr.classes_), "vocabulary": {t: int(i) for t, i in sorted(vec.vocabulary_.items(), key=lambda x: x[1])},
        "idf": [round(float(x), 4) for x in vec.idf_], "coef": [[round(float(x), 4) for x in row] for row in lr.coef_],
        "intercept": [round(float(x), 4) for x in lr.intercept_], "trained_on": len(rows) + len(para)}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {MODEL_PATH} ({len(rows)} questions + {len(para)} rewordings)")


if __name__ == "__main__":
    export() if "--export" in sys.argv else main()
