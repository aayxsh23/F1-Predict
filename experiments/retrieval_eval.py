"""Does the knowledge search find the right article? Phase 10 item 5.3.

Search is BM25 over two indexes (src/rag/corpus.py): Wikipedia, and the FIA
rules + steward rulings + circuit notes. A labelled set, built without
judging anything by hand: questions that name a driver, team or circuit, whose
right article is that entity's own Wikipedia page (data/history/people.json
records it). Two query styles:

  held-out   the raw question from the 600-question evaluation set
  model      the query the teacher model actually wrote for a question
             (training conversations): what the fine-tuned model imitates

Score: hit@5 (the right page among the 5 hits search_knowledge returns,
which is what the model reads) and MRR. Questions naming several entities
count a hit for any of them.

Rules (added 2026-10-21): training conversations where the teacher searched
the rules and its answer, which passed every check, cites an article that
search returned. That article is the label; the query is the user's own
question, searched over the rules only (as `only="rules"` does).

    python -m experiments.retrieval_eval
"""
import json
import re
from pathlib import Path

from src.agent import tools
from src.agent.history import _fold

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = json.loads((ROOT / "data" / "history" / "people.json").read_text(encoding="utf-8"))
ENTITY_CATEGORIES = {"bio_background", "team_story", "circuit_story", "comparisons", "history_stats"}


def _names() -> list[tuple[str, str]]:
    """(name, Wikipedia URL): full driver names, unique surnames of drivers, team and circuit names.
    Matched as written, capitals included, so "a driver" never means Paddy Driver."""
    drivers = PEOPLE["drivers"].values()
    surnames: dict[str, list[str]] = {}
    for d in drivers:
        surnames.setdefault(_fold(d["family"]).title(), []).append(d["url"])
    names = [(_fold(d["name"]).title(), d["url"]) for d in drivers if d.get("url")]
    names += [(s, urls[0]) for s, urls in surnames.items() if len(urls) == 1 and len(s) > 3]
    names += [(_fold(t["name"]).title(), t["url"]) for t in PEOPLE["constructors"].values() if len(t["name"]) > 3]
    names += [(_fold(c[k]).title(), c["url"]) for c in PEOPLE["circuits"].values() for k in ("name", "locality") if len(c[k]) > 3]
    return names


def _fold_keep_case(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


NAMES = _names()


def expected(question: str) -> set[str]:
    q = " ".join(w if w[:1].isupper() else w.lower() for w in _fold_keep_case(question).split())
    return {url.replace("http://", "https://") for name, url in NAMES if re.search(rf"\b{re.escape(name)}\b", q)}


def rank(query: str, want: set[str]) -> int | None:
    hits = tools.search_knowledge(query, k=5)["hits"]
    return next((i for i, h in enumerate(hits, 1) if h["source"].replace("http://", "https://") in want), None)


def report(pairs: list[tuple[str, set[str]]]) -> dict:
    ranks = [rank(q, want) for q, want in pairs]
    return {"n": len(pairs), "hit@1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
            "hit@5": round(sum(r is not None for r in ranks) / len(ranks), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3)}


_ARTICLE = re.compile(r"\b[Aa]rticles?\s+([A-Z]?\d+(?:\.\d+)+)")


def rules_pairs(teacher_query: bool = False) -> list[tuple[str, set[str]]]:
    """(query, cited articles): the user's question, or the teacher's first rules query."""
    pairs = []
    for run in ("full", "topup", "topup2", "topup3"):
        path = ROOT / "training" / "data" / run / "kept.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            msgs = r["messages"]
            queries = [c["function"]["arguments"].get("query") for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []
                       if c["function"]["name"] == "search_knowledge" and c["function"]["arguments"].get("only") == "rules"]
            if not queries or r["category"] == "followup":
                continue
            returned = {a for m in msgs if m["role"] == "tool" for a in re.findall(r'"article":"([^"]+)"', m["content"])}
            cited = set(_ARTICLE.findall(msgs[-1]["content"])) & returned
            if cited:
                pairs.append((queries[0] if teacher_query else msgs[1]["content"], cited))
    return pairs


def rules_rank(query: str, want: set[str]) -> int | None:
    hits = tools.search_knowledge(query, k=5, only="rules")["hits"]
    return next((i for i, h in enumerate(hits, 1) if h.get("article") in want), None)


def main() -> None:
    indexed = {c["source"].replace("http://", "https://") for c in json.loads((ROOT / "data" / "corpus" / "wiki_index.json").read_text(encoding="utf-8"))["chunks"]}
    held_out = []
    for line in (ROOT / "training" / "eval_questions.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        want = expected(x["question"]) & indexed
        if x["category"] in ENTITY_CATEGORIES and want:
            held_out.append((x["question"], want))
    model = []
    for run in ("full", "topup", "topup2", "topup3"):
        path = ROOT / "training" / "data" / run / "kept.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["category"] not in ENTITY_CATEGORIES:
                continue
            want = expected(r["messages"][1]["content"]) & indexed
            calls = [c["function"] for m in r["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]
            query = next((c["arguments"].get("query") for c in calls if c["name"] == "search_knowledge" and c["arguments"].get("only") != "rules"), None)
            if want and query:
                model.append((query, want))
    out = {"held-out questions as queries": report(held_out), "the teacher's own queries": report(model[:1500])}
    for label, teacher in (("rules: the user's question", False), ("rules: the teacher's query", True)):
        pairs = rules_pairs(teacher)
        ranks = [rules_rank(q, want) for q, want in pairs]
        out[label] = {"n": len(pairs), "hit@1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
                      "hit@5": round(sum(r is not None for r in ranks) / len(ranks), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
