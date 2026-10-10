"""The grounding corpus -- FIA regulations, steward decisions, circuit write-ups
(data/corpus/) -- as one prebuilt JSON index plus keyword search over it.

`python -m src.rag.corpus` extracts the PDFs once and writes
data/corpus/index.json; the API only ever reads that file, so serving needs no
PDF parser, vector database or embedding model. Regulations are chunked on
their own article numbers ("B1.8.5 ...") so every search hit can be cited by
article, and circuit write-ups carry their circuit so a Baku question never
retrieves Le Castellet.

Search is BM25: exact terms ("unsafe release", "B1.8.5", "grid penalty")
matter more than loose meaning in rulebooks, and the chat model rewrites
vague questions into keywords before calling it."""
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
INDEX_PATH = CORPUS_DIR / "index.json"
WIKI_INDEX_PATH = CORPUS_DIR / "wiki_index.json"  # built by src/data/wiki.py; big, gitignored, never loaded by the API
DOC_TYPES = {"regulations": "regulation", "steward_decisions": "steward_decision", "race_summaries": "circuit_summary"}
BROWSABLE = {"regulation", "steward_decision"}  # circuit write-ups feed the chat, not the Rules tab

_ARTICLE = re.compile(r"(?m)^([A-Z]?\d{1,2}(?:\.\d{1,2}){1,3})\s+(?=\S)")
_STEWARD = re.compile(r"^(\d{4})_([a-z]+)_gp_car(\d+)_(.+)\.pdf$")
_ACRONYMS = {"Fia": "FIA", "Pu": "PU", "Gp": "GP", "Drs": "DRS", "Vsc": "VSC", "Sc": "SC"}
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())
CHUNK_CHARS = 1500


def _normalize(text: str) -> str:
    # NFKC folds the PDFs' ligatures ("ﬁ") and non-breaking spaces into plain text
    return unicodedata.normalize("NFKC", text)


def _prettify(stem: str) -> str:
    return " ".join(_ACRONYMS.get(w, w) for w in stem.replace("_", " ").replace("-", " ").title().split())


def _slug(s: str) -> str:
    return re.sub(r"[^a-z]+", "_", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).strip("_")


def _circuit_for(stem: str) -> str | None:
    """race_summaries/<name>.txt -> circuit location, by slug match against the circuit table."""
    from src.features.circuit_reference import load
    for loc in load()["location"].unique():
        if _slug(loc).split("_")[0] in stem.split("_"):
            return loc
    return None


def _split(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-packed pieces of at most ~size characters."""
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=\.)\n", text):
        if cur and len(cur) + len(para) > size:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _article_chunks(text: str) -> list[tuple[str | None, str]]:
    """(article number, text) pieces, cut at every article heading."""
    heads = list(_ARTICLE.finditer(text))
    if not heads:
        return [(None, t) for t in _split(text)]
    out = [(None, t) for t in _split(text[: heads[0].start()])]
    for m, nxt in zip(heads, heads[1:] + [None]):
        body = text[m.start(): nxt.start() if nxt else len(text)]
        out += [(m.group(1), t) for t in _split(body)]
    return out


def build_index() -> dict:
    from pypdf import PdfReader  # build-time only

    documents, chunks = [], []
    for subdir, doc_type in DOC_TYPES.items():
        for path in sorted((CORPUS_DIR / subdir).glob("*")):
            if path.suffix == ".pdf":
                text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
            elif path.suffix == ".txt":
                text = path.read_text(encoding="utf-8")
            else:
                continue
            text = _normalize(text)
            doc = {
                "filename": path.name, "doc_type": doc_type, "title": _prettify(path.stem),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            if doc_type == "steward_decision" and (m := _STEWARD.match(path.name)):
                _season, gp, car, infringement = m.groups()
                doc.update(grand_prix=gp.title(), car_number=int(car),
                           title=f"{gp.title()} GP — Car {car} ({infringement.replace('_', ' ')})")
            circuit = _circuit_for(path.stem) if doc_type == "circuit_summary" else None
            pieces = _article_chunks(text) if doc_type == "regulation" else [(None, t) for t in _split(text)]
            for article, body in pieces:
                chunks.append({"source": path.name, "doc_type": doc_type, "title": doc["title"],
                               "article": article, "circuit": circuit, "text": body})
            documents.append({**doc, "text": text})
    index = {"documents": documents, "chunks": chunks}
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


@lru_cache(maxsize=2)
def load_index(path: Path = INDEX_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if t not in _STOP]


@lru_cache(maxsize=2)
def _bm25(path: Path = INDEX_PATH):
    chunks = load_index(path)["chunks"]
    tfs = [Counter(_tokens(c["title"] + " " + (c["article"] or "") + " " + c["text"])) for c in chunks]
    df = Counter(t for tf in tfs for t in tf)
    lengths = [sum(tf.values()) for tf in tfs]
    avg = sum(lengths) / max(1, len(lengths))
    idf = {t: math.log(1 + (len(tfs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    titles = [frozenset(_tokens(c["title"])) for c in chunks]
    return chunks, tfs, lengths, avg, idf, titles


def search(query: str, k: int = 5, doc_types: set[str] | None = None, circuit: str | None = None,
           index_path: Path = INDEX_PATH) -> list[dict]:
    """Top-k chunks by BM25 (k1=1.5, b=0.75). `circuit` keeps circuit
    write-ups to that circuit only (rules and rulings always pass).
    `index_path` picks the index (the Wikipedia one lives in its own file)."""
    chunks, tfs, lengths, avg, idf, titles = _bm25(index_path)
    terms = set(_tokens(query))
    scored = []
    for c, tf, n, title in zip(chunks, tfs, lengths, titles):
        if doc_types and c["doc_type"] not in doc_types:
            continue
        if c["doc_type"] == "circuit_summary" and circuit and c["circuit"] != circuit:
            continue
        s = sum(idf[t] * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * n / avg)) for t in terms if t in tf)
        if s > 0:
            # every word of the page's title is in the query: the user named this page ("Max Verstappen" over "Jos Verstappen")
            scored.append((s * 2 if title and title <= terms else s, c))
    scored.sort(key=lambda x: -x[0])
    return [{**c, "score": round(s, 3)} for s, c in scored[:k]]


def circuit_summary(circuit: str) -> str:
    """The whole write-up for one circuit, or "" if there isn't one."""
    return "\n\n".join(c["text"] for c in load_index()["chunks"] if c["circuit"] == circuit)


# --- the Rules tab's browse/read/search ---

def list_documents() -> list[dict]:
    return [{k: v for k, v in d.items() if k != "text"} for d in load_index()["documents"] if d["doc_type"] in BROWSABLE]


def get_document_text(filename: str) -> str:
    for d in load_index()["documents"]:
        if d["filename"] == filename and d["doc_type"] in BROWSABLE:
            return d["text"]
    raise FileNotFoundError(filename)


def search_documents(query: str, k: int = 5) -> list[dict]:
    return [
        {"filename": h["source"], "doc_type": h["doc_type"], "article": h["article"], "snippet": h["text"][:400]}
        for h in search(query, k=k, doc_types=BROWSABLE)
    ]


if __name__ == "__main__":
    idx = build_index()
    print(f"indexed {len(idx['documents'])} documents -> {len(idx['chunks'])} chunks -> {INDEX_PATH}")
