"""F1 narrative from Wikipedia, for the chat's `search_knowledge` tool: every
driver, constructor, circuit, championship season and Grand Prix that Jolpica
knows (it supplies the Wikipedia URLs), plus a short list of rules explainers.

  data/raw/wikipedia/pages.jsonl   one page per line, resumable (gitignored)
  data/corpus/wiki_index.json      BM25 chunks over those pages (gitignored)

Both are regenerable: `python -m src.data.wiki` (about ten minutes). Wikipedia
text is CC BY-SA 4.0, so every chunk carries its page title and URL and the
chat shows them as sources. Needs the Jolpica people lists in the cache
(`python -m src.data.history` fetches them first).
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests

from src.data import history
from src.rag.corpus import WIKI_INDEX_PATH, _split

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "wikipedia" / "pages.jsonl"
API = "https://en.wikipedia.org/w/api.php"
GAP_S = 0.15
MAX_CHARS = 60_000
MIN_CHARS = 200
LICENSE = "CC BY-SA 4.0"
# cut a page at the first of these headings: they are links and citations, not content
_TAIL = re.compile(r"(?m)^==+\s*(References|External links|Notes|Sources|Further reading|Bibliography|See also|Citations|Footnotes)\s*==+\s*$")

RULES_PAGES = [
    "Drag reduction system", "Formula One tyres", "Racing flags", "Formula One regulations", "Safety car", "Virtual safety car",
    "Pit stop", "Parc fermé", "Formula One car", "Formula One engines", "Formula One sprint", "Formula One",
    "List of Formula One World Championship points scoring systems", "Formula One racing", "Grand Prix motor racing",
    "Formula One qualifying", "Ground effect in cars", "Pole position", "Fastest lap", "Slipstream", "Undercut (motorsport)",
    "FIA Formula One World Championship", "Formula One Group", "Fédération Internationale de l'Automobile",
    "Track limits", "Active suspension", "Turbocharger", "Formula One Constructors' Championship", "World Drivers' Championship",
]
# Wikipedia's own titles differ from Jolpica's/ours: try these in order until a real article comes back
SEASON_FORMS = ("{y} Formula One World Championship", "{y} Formula One season")
DISAMBIGUATION_SUFFIXES = {"driver": (" (racing driver)", " (Formula One driver)", " (driver)"),
                           "constructor": (" (Formula One)", " (constructor)", " (racing team)", " (Formula One team)", " (auto racing)"),
                           "grand_prix": (" (Formula One)",)}


def _title(url: str) -> str:
    return unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ")


def targets() -> list[tuple[str, str]]:
    """(kind, title) for everything worth fetching."""
    out: list[tuple[str, str]] = []
    for path, table, key, kind in (("drivers.json", "DriverTable", "Drivers", "driver"),
                                    ("constructors.json", "ConstructorTable", "Constructors", "constructor"),
                                    ("circuits.json", "CircuitTable", "Circuits", "circuit")):
        out += [(kind, _title(x["url"])) for x in history.pages(path, table, key) if x.get("url")]
    out += [("season", SEASON_FORMS[0].format(y=y)) for y in range(history.FIRST_SEASON, history.current_season() + 1)]
    races_file = history.OUT_DIR / "races.json"  # Grand Prix names come from the built history, so this makes no Jolpica calls
    if races_file.exists():
        out += [("grand_prix", n) for n in sorted({r["name"] for r in json.loads(races_file.read_text(encoding="utf-8"))})]
    out += [("rules", t) for t in RULES_PAGES]
    seen, unique = set(), []
    for kind, title in out:
        if title not in seen:
            seen.add(title)
            unique.append((kind, title))
    return unique


def fetch_page(title: str) -> dict | None:
    resp = requests.get(API, headers=history.HEADERS, timeout=30, params={
        "action": "query", "format": "json", "prop": "extracts|info", "inprop": "url", "explaintext": 1, "redirects": 1,
        "titles": title, "formatversion": 2})
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = pages[0] if pages else {}
    if page.get("missing") or not page.get("extract"):
        return None
    text = page["extract"]
    if m := _TAIL.search(text):
        text = text[:m.start()]
    text = text.strip()[:MAX_CHARS]
    if len(text) < MIN_CHARS or "may refer to" in text[:300]:
        return None
    return {"title": page["title"], "url": page.get("fullurl"), "text": text}


def fetch_with_fallbacks(kind: str, title: str) -> dict | None:
    alternatives = [title]
    if kind == "season":
        alternatives = [f.format(y=title[:4]) for f in SEASON_FORMS]
    else:
        alternatives += [title + suffix for suffix in DISAMBIGUATION_SUFFIXES.get(kind, ())]
    for alt in alternatives:
        if page := fetch_page(alt):
            return page
        time.sleep(GAP_S)
    return None


def fetch_all() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RAW_PATH.exists():
        done = {p["asked"] for p in map(json.loads, RAW_PATH.read_text(encoding="utf-8").splitlines()) if not p.get("missing")}
    todo = [(k, t) for k, t in targets() if t not in done]
    print(f"{len(todo)} pages to fetch ({len(done)} already done)", flush=True)
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for i, (kind, title) in enumerate(todo, 1):
            for attempt in range(5):
                try:
                    page = fetch_with_fallbacks(kind, title)
                    break
                except requests.RequestException as exc:
                    print(f"  {title}: {exc}; retry {attempt + 1}", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                continue
            f.write(json.dumps({"asked": title, "kind": kind, **(page or {"title": title, "missing": True})}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
            time.sleep(GAP_S)


def build_index() -> dict:
    chunks, documents = [], []
    latest = {p["asked"]: p for p in map(json.loads, RAW_PATH.read_text(encoding="utf-8").splitlines())}  # a retry overrides an earlier miss
    for p in latest.values():
        if p.get("missing"):
            continue
        documents.append({"title": p["title"], "kind": p["kind"], "url": p["url"], "chars": len(p["text"])})
        for text in _split(p["text"]):
            chunks.append({"source": p["url"], "doc_type": "wikipedia", "kind": p["kind"], "title": p["title"], "article": None,
                           "circuit": None, "license": LICENSE, "text": text})
    index = {"documents": documents, "chunks": chunks}
    WIKI_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"indexed {len(documents)} pages -> {len(chunks)} chunks -> {WIKI_INDEX_PATH} ({WIKI_INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    return index


if __name__ == "__main__":
    fetch_all()
    build_index()
