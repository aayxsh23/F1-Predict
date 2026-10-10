"""Two facts only the FIA's own documents carry, read from fia.com's PDFs:

  starting grid    the official grid with penalties applied ("Final Starting
                   Grid", else "Provisional Starting Grid"), published a few
                   hours before the race. Qualifying order is only a proxy:
                   since 2024 a fifth of drivers started elsewhere.
  car upgrades     each team's "Car Presentation Submissions" (from the 2024
                   Saudi Arabian GP): every updated component and why. A
                   component whose primary reason is "Performance - ..." is
                   counted; circuit-specific trims (wing levels, cooling) are not.

`python -m src.data.fia --seasons 2024 2025 2026` writes
data/raw/upgrades/<season>.parquet (season, round, team, performance_upgrades).
PDFs are cached in data/raw/fia_cache/ (gitignored).
"""
import argparse
import io
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from src.models.catalog import TEAM_LINEAGE

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "fia_cache"
UPGRADES_DIR = ROOT / "data" / "raw" / "upgrades"
BASE = "https://www.fia.com"
CHAMPIONSHIP = "/documents/championships/fia-formula-one-world-championship-14"
HEADERS = {"User-Agent": "Mozilla/5.0 (F1 forecast research; polite, cached)"}
PAUSE_S = 1.0  # between uncached requests to fia.com

log = logging.getLogger(__name__)

# team header in a car-presentation PDF -> the team name FastF1 uses (then
# TEAM_LINEAGE). Order matters: "RACING BULLS" before "RED BULL".
_TEAM_PATTERNS = [
    (r"RACING BULLS|VISA CASH APP|\bRB F1\b|\bRB FORMULA|ALPHATAURI", "Racing Bulls"),
    (r"RED BULL", "Red Bull Racing"),
    (r"FERRARI", "Ferrari"), (r"MERCEDES", "Mercedes"), (r"MCLAREN", "McLaren"),
    (r"ASTON MARTIN", "Aston Martin"), (r"ALPINE", "Alpine"), (r"HAAS", "Haas F1 Team"),
    (r"WILLIAMS", "Williams"), (r"SAUBER|AUDI|ALFA ROMEO", "Audi"), (r"CADILLAC", "Cadillac"),
]


def _get(url: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("fia.com", 1)[-1])[-180:]
    if path.exists():
        return path.read_bytes()
    time.sleep(PAUSE_S)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if not url.endswith(".pdf"):  # event pages change during a weekend; don't cache them
        return r.content
    path.write_bytes(r.content)
    return r.content


@lru_cache(maxsize=None)
def _season_path(season: int) -> str:
    html = requests.get(BASE + CHAMPIONSHIP, headers=HEADERS, timeout=30).text
    ids = [int(i) for i in re.findall(rf"season-{season}-(\d+)", html)]
    if not ids:
        raise LookupError(f"no FIA documents page for {season}")
    return f"{CHAMPIONSHIP}/season/season-{season}-{max(ids)}"


def event_documents(season: int, event_name: str) -> dict[str, str]:
    """{document title (lower case): absolute PDF url} for one Grand Prix."""
    html = _get(f"{BASE}{_season_path(season)}/event/{quote(event_name)}").decode("utf-8", "ignore")
    docs = {}
    for href in re.findall(r'href="([^"]*decision-document/[^"]+\.pdf)"', html):
        title = re.sub(r"[_\s]+", " ", href.rsplit("/", 1)[-1][:-4]).replace("%20", " ").lower()
        docs.setdefault(title, BASE + quote(href, safe="/%:"))
    return docs


def _pdf_text(url: str) -> str:
    from pypdf import PdfReader

    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(_get(url))).pages)


def _find(docs: dict[str, str], *needles: str) -> str | None:
    for needle in needles:
        hits = [u for t, u in docs.items() if needle in t and "sprint" not in t]
        if hits:
            return hits[0]
    return None


def parse_grid(text: str) -> dict[int, int]:
    """{car number: grid slot} from a starting-grid PDF. Lines read
    "<slot> <car number> <First> <LAST>"; pit-lane starters, listed without a
    slot, are left out (the caller puts them at the back)."""
    grid = {}
    for slot, car in re.findall(r"^(\d{1,2})[ \t]+(\d{1,2})[ \t]+[A-Z][\w'\-]+(?:[ \t]+[\w'\-]+)*[ \t]+[A-Z]{2,}", text, flags=re.M):
        grid.setdefault(int(car), int(slot))
    return grid


def starting_grid(season: int, event_name: str) -> dict[int, int] | None:
    """The official race grid if the FIA has published it, else None."""
    try:
        docs = event_documents(season, event_name)
        url = _find(docs, "final starting grid", "provisional starting grid")
        grid = parse_grid(_pdf_text(url)) if url else {}
    except Exception as exc:  # the forecast must not fail because fia.com did
        log.warning("starting grid unavailable: %s", exc)
        return None
    return grid if len(grid) >= 10 else None


def parse_upgrades(text: str) -> dict[str, int]:
    """{team (FastF1 name): performance components updated} from a car
    presentation PDF."""
    counts: dict[str, int] = {}
    team = None
    for line in text.splitlines():
        s = line.strip().upper()
        if 0 < len(s) < 60 and not s.startswith(("CAR PRESENTATION", "UPDATED", "PRIMARY")):
            for pat, name in _TEAM_PATTERNS:
                if re.search(pat, s):
                    team = name
                    counts.setdefault(team, 0)
                    break
        if team and re.search(r"PERFORMANCE\s*([-–]|$)", s):
            counts[team] += 1
    return counts


def upgrades(season: int, event_name: str) -> dict[str, int] | None:
    """Performance upgrades per team at one Grand Prix, None if not published."""
    try:
        url = _find(event_documents(season, event_name), "car presentation submissions")
        return parse_upgrades(_pdf_text(url)) if url else None
    except Exception as exc:
        log.warning("car presentation unavailable for %s %s: %s", season, event_name, exc)
        return None


def build_upgrades(seasons: list[int]) -> None:
    from src.data.fastf1_client import event_schedule

    UPGRADES_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize(None)
    for season in seasons:
        rows = []
        for _, e in event_schedule(season).iterrows():
            if e["Session1DateUtc"] > now:
                break
            got = upgrades(season, e["EventName"])
            log.info("%s R%s %s: %s", season, e["RoundNumber"], e["EventName"], got)
            for team, n in (got or {}).items():
                rows.append({"season": season, "round": int(e["RoundNumber"]), "team": TEAM_LINEAGE.get(team, team),
                             "performance_upgrades": n})
        if rows:
            pd.DataFrame(rows).to_parquet(UPGRADES_DIR / f"{season}.parquet", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    build_upgrades(parser.parse_args().seasons)
