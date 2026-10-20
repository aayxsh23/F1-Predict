"""Chat router: keyword rules, or a small learned classifier (TF-IDF + logistic
regression)? Phase 10 item 5.2.

Labels come from the training conversations (training/data/<run>/kept.jsonl):
the lookup the teacher model chose first says what the question was, mapped
to the router intent that answers it, or `general` when no deterministic
intent can (history, records, background, refusals: the local model's job).
Questions are split by conversation id; the classifier learns on 80%, both
are scored on the other 20%, templated and free-form questions separately
(templated ones are formulaic and flatter any learner).

What matters most is the wrong deterministic answer: the router confidently
answering with the wrong intent (a forecast table for a history question).
Falling through to `general` is safe: the local model, or the plain fallback,
handles it.

    python -m experiments.router_classifier            # the comparison
    python -m experiments.router_classifier --export   # refit on every question, write src/agent/router_model.json

Outcome (2026-10-19, 21,668 questions): on free-form questions the keyword
rules answered 44.9% with a wrong deterministic intent (history questions
containing "win" got this weekend's forecast); the classifier 2.6%, at 91.6%
accuracy vs 52.1%; vetoing the rules with the classifier was worse than the
classifier alone. Shipped: the classifier routes (src/agent/router.py), plain
Python over the exported weights.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import re
import subprocess

from src.agent.router import _ARTICLE_RE, _circuit_in

DATA = Path(__file__).resolve().parents[1] / "training" / "data"
MAX_FEATURES = 3000  # 3k/6k/10k/20k features scored 0.916/0.911/0.914/0.912 on free-form questions: smallest wins
VETO = (0.5, 0.7, 0.9)
MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "agent" / "router_model.json"
SIMPLE = {"forecast": "race_forecast", "explain_prediction": "explain_prediction", "head_to_head": "head_to_head",
          "title_scenario_for": "title_scenario", "race_strategy": "race_strategy", "forecast_timeline": "forecast_timeline",
          "season_schedule": "season_schedule", "past_race_prediction": "past_race_prediction",
          "model_track_record": "model_track_record", "driver_ratings": "driver_ratings"}


BASELINE = "5ece5d3"  # the last commit with the keyword router


def keyword_rules():
    """The keyword router as shipped before this experiment, loaded from git so the comparison stays re-runnable."""
    src = subprocess.run(["git", "show", f"{BASELINE}:src/agent/router.py"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    block = src[src.index("CIRCUIT_TOPIC_WORDS = "):src.index("def classify_intent(")]
    ns = {"re": re, "_ARTICLE_RE": _ARTICLE_RE, "_circuit_in": _circuit_in}
    exec(block, ns)  # noqa: S102  (our own code, from our own history)
    return lambda q: ns["_classify_specific"](q.lower().strip()) or "general"


def label(first_call: dict | None) -> str:
    if not first_call:
        return "general"
    name, args = first_call["function"]["name"], first_call["function"].get("arguments") or {}
    if name == "championship":
        return "constructor_standings" if args.get("kind") == "constructors" else "driver_standings"
    if name == "search_knowledge":
        return {"rules": "search_rules", "circuit_notes": "circuit_guide"}.get(args.get("only"), "general")
    return SIMPLE.get(name, "general")


def load() -> list[dict]:
    rows = []
    for run in ("full", "topup", "topup2"):
        path = DATA / run / "kept.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["category"] == "followup":
                continue
            msgs = r["messages"]
            first = next((m["tool_calls"][0] for m in msgs if m["role"] == "assistant" and m.get("tool_calls")), None)
            rows.append({"id": r["id"], "q": msgs[1]["content"], "y": label(first), "templated": r["category"].startswith("tpl_")})
    return rows


def score(pred: list[str], gold: list[str]) -> dict:
    n = len(gold)
    wrong_det = sum(p != g and p != "general" for p, g in zip(pred, gold))
    det = [(p, g) for p, g in zip(pred, gold) if g != "general"]
    return {"n": n, "accuracy": round(sum(p == g for p, g in zip(pred, gold)) / n, 3),
            "wrong_deterministic_answer": round(wrong_det / n, 3),
            "deterministic_recall": round(sum(p == g for p, g in det) / max(1, len(det)), 3)}


def pipeline():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=MAX_FEATURES, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=4.0))


def export() -> None:
    """Fit on every question and write what router.py needs to score a question without scikit-learn."""
    rows = load()
    clf = pipeline().fit([r["q"] for r in rows], [r["y"] for r in rows])
    vec, lr = clf[0], clf[-1]
    MODEL_PATH.write_text(json.dumps({
        "classes": list(lr.classes_), "vocabulary": {t: int(i) for t, i in sorted(vec.vocabulary_.items(), key=lambda x: x[1])},
        "idf": [round(float(x), 4) for x in vec.idf_], "coef": [[round(float(x), 4) for x in row] for row in lr.coef_],
        "intercept": [round(float(x), 4) for x in lr.intercept_], "trained_on": len(rows)}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB, {len(rows)} questions)")


def main() -> None:
    rows = load()
    test = [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 5 == 0]
    train = [r for r in rows if r not in test] if len(rows) < 2000 else [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 5]
    print(f"{len(rows)} questions; labels: {dict(Counter(r['y'] for r in rows).most_common())}")
    clf = pipeline()
    clf.fit([r["q"] for r in train], [r["y"] for r in train])
    general = list(clf.classes_).index("general")
    old = keyword_rules()
    out = {}
    for part, sel in (("free-form", lambda r: not r["templated"]), ("templated", lambda r: r["templated"])):
        rs = [r for r in test if sel(r)]
        gold = [r["y"] for r in rs]
        rules = [old(r["q"]) for r in rs]
        rules = ["general" if p in ("driver_history",) else p for p in rules]  # recent results: no clean equivalent in the labels
        learned = list(clf.predict([r["q"] for r in rs]))
        p_general = clf.predict_proba([r["q"] for r in rs])[:, general]
        out[part] = {"keyword rules": score(rules, gold), "tf-idf + logistic": score(learned, gold)}
        for t in VETO:  # the rules decide, but the classifier vetoes a deterministic answer it thinks is a general question
            vetoed = ["general" if p != "general" and pg > t else p for p, pg in zip(rules, p_general)]
            out[part][f"rules, vetoed when P(general) > {t}"] = score(vetoed, gold)
        mistakes = Counter((g, p) for p, g in zip(rules, gold) if p != g and p != "general")
        out[part]["rules' most common wrong answers (truth, answered)"] = [f"{g} -> {p}: {k}" for (g, p), k in mistakes.most_common(6)]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys

    export() if "--export" in sys.argv else main()
