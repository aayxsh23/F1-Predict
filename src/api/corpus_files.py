"""Browse/read/search data/corpus/{regulations,steward_decisions}/ -- the
static, hand-curated FIA documents that src/rag/ used to only ever consume
silently as backend RAG context. This is a genuinely separate concern from
cache.py's get_json(): these files ship with the repo and only change on
redeploy, so there's no TTL/remote-fallback story to build -- just read
data/corpus/ off disk directly.

Chunk metadata in the Chroma store is too sparse for a listing UI (only
`source`/`doc_type`, see ingest_corpus.py) -- list_documents() reads the
filesystem directly instead. search_documents() is the one place that does
use the vector store, since full-text semantic search is exactly what it's
already built for.
"""
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.rag import ingest_corpus

# race_summaries are real corpus content but not part of the browsable
# Regulations feature -- they're circuit write-ups, not rules/decisions.
_EXCLUDED_DOC_TYPES = {"circuit_summary"}

# Informal filename convention for steward decisions, e.g.
# "2026_australian_gp_car12_unsafe_release.pdf" -- a string convention, not
# an enforced schema, so a non-matching filename just falls back to a plain
# prettified title instead of erroring.
_STEWARD_FILENAME_RE = re.compile(r"^(\d{4})_([a-z]+)_gp_car(\d+)_(.+)\.pdf$")


def _prettify(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").title()


def _document_meta(path: Path, doc_type: str) -> dict:
    stat = path.stat()
    meta = {
        "filename": path.name,
        "doc_type": doc_type,
        "title": _prettify(path.stem),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if doc_type == "steward_decision":
        match = _STEWARD_FILENAME_RE.match(path.name)
        if match:
            _season, gp, car, infringement = match.groups()
            meta["grand_prix"] = gp.title()
            meta["car_number"] = int(car)
            meta["title"] = f"{gp.title()} GP — Car {car} ({infringement.replace('_', ' ')})"
    return meta


def list_documents() -> list[dict]:
    """All browsable corpus documents, read straight off disk."""
    docs = []
    for subdir, doc_type in ingest_corpus.DOC_TYPES.items():
        if doc_type in _EXCLUDED_DOC_TYPES:
            continue
        folder = ingest_corpus.CORPUS_DIR / subdir
        for path in sorted(folder.glob("*")):
            if path.suffix not in (".pdf", ".txt"):
                continue
            docs.append(_document_meta(path, doc_type))
    return docs


@lru_cache(maxsize=None)
def get_document_text(filename: str) -> str:
    """Full extracted text of one document. filename is untrusted HTTP
    input -- reject path separators/traversal before touching disk."""
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise FileNotFoundError(filename)
    for subdir, doc_type in ingest_corpus.DOC_TYPES.items():
        if doc_type in _EXCLUDED_DOC_TYPES:
            continue
        path = ingest_corpus.CORPUS_DIR / subdir / filename
        if path.is_file():
            return ingest_corpus.extract_text(path)
    raise FileNotFoundError(filename)


def search_documents(query: str, k: int = 5) -> list[dict]:
    """Semantic search over regulations/steward-decision text only (the
    corpus also holds race_summaries in the same store -- over-fetch then
    filter in Python rather than depend on Chroma's metadata-filter operator
    syntax, since the whole corpus is small enough that this costs nothing)."""
    store = ingest_corpus.load_vector_store()
    candidates = store.similarity_search(query, k=max(k * 4, 20))
    hits = [d for d in candidates if d.metadata.get("doc_type") not in _EXCLUDED_DOC_TYPES]
    return [
        {
            "filename": d.metadata["source"],
            "doc_type": d.metadata["doc_type"],
            "snippet": d.page_content[:400],
        }
        for d in hits[:k]
    ]
