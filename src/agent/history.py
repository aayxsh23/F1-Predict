"""History tools: every F1 World Championship race since 1950, from the files
src/data/history.py builds (data/history/). Plain Python, no pandas: like the
rest of src/agent/tools.py these are deterministic lookups the chat model
calls, never something it recalls.

Every function returns a plain dict. Aggregates (wins, poles, records, titles)
are COMPUTED here from the results table; a model must never state one from
memory. Each result carries `source` and, where the data has gaps, `note`.

Definitions, so answers can state them:
  race        a result row for a driver, excluding drivers who did not start
  win/podium  finishing position 1 / 1-3
  pole        grid slot 1, the convention the record books use (a driver who
              qualified first but took a grid penalty does not get the pole);
              qualifying P1 only if nobody is listed on grid 1. Totals can differ
              from official counts by a pole or so in early seasons
  points      race points only (sprint points are not included)
  fastest lap fastest-lap rank 1, recorded from 2004 only
  title       first place in a completed season's final standings
"""
import html
import json
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from src.agent import context

HISTORY_DIR = Path(__file__).resolve().parents[2] / "data" / "history"
SOURCE = "Jolpica-F1 (Ergast) race results, 1950 onwards"
NOT_STARTED = {"Did not start", "Did not qualify", "Did not prequalify", "Withdrew", "Practiced only", "Excluded", "Not classified"}
FASTEST_LAP_FROM = 2004
METRICS = ("wins", "podiums", "poles", "points", "races", "fastest_laps", "titles")
GROUPS = ("driver", "team", "circuit", "season")


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


@lru_cache(maxsize=1)
def _data() -> dict:
    if not (HISTORY_DIR / "races.json").exists():
        raise FileNotFoundError("data/history/ is missing: run `python -m src.data.history`")
    races = json.loads((HISTORY_DIR / "races.json").read_text(encoding="utf-8"))
    people = json.loads((HISTORY_DIR / "people.json").read_text(encoding="utf-8"))
    standings = json.loads((HISTORY_DIR / "standings.json").read_text(encoding="utf-8"))
    for kind in ("constructors", "circuits"):  # Jolpica leaves HTML entities in names ("Lotus-Pratt &amp; Whitney")
        for entry in people[kind].values():
            entry["name"] = html.unescape(entry["name"])
    # Jolpica splits many 1950s-70s teams by engine ("brabham-repco", "lotus-ford"); record books count one team
    constructors = people["constructors"]
    family = {i: (i.split("-")[0] if i.split("-")[0] in constructors else "team_lotus" if i.startswith("lotus-") else i) for i in constructors}
    for st in standings.values():
        for c in st["constructors"]:
            c["team"] = family.get(c["team"], c["team"])
        for dr in st["drivers"]:
            dr["teams"] = [family.get(t, t) for t in dr["teams"]]
    rows = []  # one flat row per driver per race, with pole/win flags resolved
    for race in races:
        # everyone listed on grid slot 1 (a shared 1950s car lists two drivers); qualifying P1 only if nobody is
        pole_drivers = {r["driver"] for r in race["results"] if r["grid"] == 1} or {q["driver"] for q in race["qualifying"] if q["pos"] == 1}
        for r in race["results"]:
            if r["status"] in NOT_STARTED:
                continue
            pole = r["driver"] in pole_drivers
            rows.append({"season": race["season"], "round": race["round"], "date": race["date"] or "", "race": race["name"], "circuit": race["circuit"],
                         "driver": r["driver"], "team": family.get(r["team"], r["team"]), "grid": r["grid"], "pos": r["pos"], "points": r["points"] or 0.0,
                         "status": r["status"], "win": r["pos"] == 1, "podium": r["pos"] is not None and r["pos"] <= 3 and r["pos_text"] not in ("R", "D", "W", "F", "N", "E"),
                         "pole": bool(pole), "fl": r["fl_rank"] == 1})
    # a driver who took over a teammate's car (1950s) appears twice in one race: count the race once
    merged: dict[tuple, dict] = {}
    for row in rows:
        key = (row["driver"], row["season"], row["round"])
        if key not in merged:
            merged[key] = row
            continue
        first = merged[key]
        for flag in ("win", "podium", "pole", "fl"):
            first[flag] = first[flag] or row[flag]
        first["points"] += row["points"]
        if row["pos"] is not None and (first["pos"] is None or row["pos"] < first["pos"]):
            first.update(pos=row["pos"], team=row["team"], status=row["status"])
    rows = list(merged.values())
    titles = {"driver": defaultdict(list), "team": defaultdict(list)}
    for season, st in standings.items():
        complete = bool(st["rounds"] and st["rounds"] >= st["races_in_season"])
        if not complete:
            continue
        if champ := next((d for d in st["drivers"] if d["pos"] == 1), None):
            titles["driver"][champ["driver"]].append(int(season))
        if champ := next((c for c in st["constructors"] if c["pos"] == 1), None):
            titles["team"][champ["team"]].append(int(season))
    return {"races": races, "people": people, "standings": standings, "rows": rows, "titles": titles,
            "latest_season": max(r["season"] for r in races)}


def _rows() -> list[dict]:
    """Result rows before today (a replayed weekend, src/agent/context.py, sees no later race)."""
    cut = context.today().isoformat()
    return [r for r in _data()["rows"] if r["date"] < cut]


def _races() -> list[dict]:
    cut = context.today().isoformat()
    return [r for r in _data()["races"] if (r["date"] or "") < cut]


def _driver_name(driver_id: str) -> str:
    p = _data()["people"]["drivers"].get(driver_id)
    return p["name"] if p else driver_id


def _team_name(team_id: str) -> str:
    p = _data()["people"]["constructors"].get(team_id)
    return p["name"] if p else team_id


def _circuit_name(circuit_id: str) -> str:
    p = _data()["people"]["circuits"].get(circuit_id)
    return p["name"] if p else circuit_id


# --- name resolution ---

def _career_counts() -> Counter:
    return Counter(r["driver"] for r in _rows())


def find_driver(query: str) -> tuple[str | None, list[dict]]:
    """(driver_id, candidates). A unique match gives (id, []); several give
    (None, candidates ranked by races) so the caller can ask instead of guess."""
    q = _fold(query)
    people = _data()["people"]["drivers"]
    if q.upper() in {p["code"] for p in people.values() if p.get("code")} and q.isalpha() and len(q) == 3:
        hits = [i for i, p in people.items() if (p.get("code") or "").lower() == q]
        counts = _career_counts()
        return (max(hits, key=lambda i: counts[i]), []) if hits else (None, [])
    exact = [i for i, p in people.items() if _fold(p["name"]) == q or i == q.replace(" ", "_")]
    if len(exact) == 1:
        return exact[0], []
    hits = exact or [i for i, p in people.items() if _fold(p["family"]) == q or q in _fold(p["name"]).split()]
    hits = hits or [i for i, p in people.items() if q and q in _fold(p["name"])]
    counts = _career_counts()
    hits = sorted(set(hits), key=lambda i: -counts[i])
    if len(hits) == 1:
        return hits[0], []
    cands = [{"driver_id": i, "name": people[i]["name"], "born": people[i]["born"], "races": counts[i]} for i in hits[:6]]
    return None, cands


def find_team(query: str) -> tuple[str | None, list[dict]]:
    q = _fold(query)
    teams = _data()["people"]["constructors"]
    spans = defaultdict(list)
    for r in _rows():
        spans[r["team"]].append(r["season"])
    exact = [i for i, t in teams.items() if _fold(t["name"]) == q or i == q.replace(" ", "_")]
    hits = sorted({i for i, t in teams.items() if q in _fold(t["name"]) and spans[i]} | {i for i in exact if spans[i]}, key=lambda i: -len(spans[i]))
    # an exact name match wins only if no other team sharing the name has more entries ("Lotus" is Team Lotus, not the 2010-11 "Lotus")
    if len(hits) == 1 or (exact and hits and hits[0] in exact):
        return (hits[0] if hits else None), []
    cands = [{"team_id": i, "name": teams[i]["name"], "seasons": f"{min(spans[i])}-{max(spans[i])}", "race_entries": len(spans[i])} for i in hits[:6]]
    return None, cands


def find_circuit(query: str) -> str | None:
    q = _fold(query)
    for i, c in _data()["people"]["circuits"].items():
        if q in (i, _fold(c["name"]), _fold(c["locality"])) or (len(q) > 3 and q in _fold(c["name"])):
            return i
    races = _races()
    for r in reversed(races):  # a Grand Prix name: the newest circuit that hosted it
        if q and q in _fold(r["name"]):
            return r["circuit"]
    return None


def _also_matches(tid: str, query: str) -> list[dict]:
    """Other constructors whose name contains the query ("Lotus" also matches Lotus F1, 2011-15), so a
    unique-looking answer can still say another team shares the name."""
    q = _fold(query)
    spans = defaultdict(list)
    for r in _rows():
        spans[r["team"]].append(r["season"])
    return [{"team": t["name"], "team_id": i, "seasons": f"{min(spans[i])}-{max(spans[i])}"}
            for i, t in _data()["people"]["constructors"].items() if i != tid and spans[i] and q in _fold(t["name"])][:5]


def _ambiguous(kind: str, query: str, cands: list[dict]) -> dict:
    if not cands:
        return {"error": f"no {kind} found matching '{query}'", "source": SOURCE}
    return {"error": f"'{query}' matches several {kind}s; call again with a full name or the id", "candidates": cands, "source": SOURCE}


# --- shared helpers ---

def _summary(rows: list[dict]) -> dict:
    return {"races": len(rows), "wins": sum(r["win"] for r in rows), "podiums": sum(r["podium"] for r in rows),
            "poles": sum(r["pole"] for r in rows), "points": round(sum(r["points"] for r in rows), 1),
            "fastest_laps": sum(r["fl"] for r in rows)}


def _seasons(rows: list[dict]) -> str:
    ys = sorted({r["season"] for r in rows})
    return f"{ys[0]}-{ys[-1]}" if ys else ""


NOTES = {"points": "race points only; sprint points not included", "fastest_laps": f"fastest laps are recorded from {FASTEST_LAP_FROM} only",
         "poles": "pole = grid slot 1 (the record-book convention); early-season totals can differ from official counts by a pole or so"}


# --- the tools ---

def history_results(season: int | None = None, round: int | None = None, grand_prix: str | None = None) -> dict:
    """One race's full classification (season + round, or season + grand_prix
    name), a season's list of races and winners (season only), or every winner
    of one Grand Prix across the years (grand_prix only)."""
    d = _data()
    races = _races()
    if grand_prix and not round:
        circuit = find_circuit(grand_prix)
        name_hits = [r for r in races if _fold(grand_prix) in _fold(r["name"])]
        pool = name_hits or [r for r in races if circuit and r["circuit"] == circuit]
        if season:
            pool = [r for r in pool if r["season"] == season]
        if season and len(pool) == 1:
            round = pool[0]["round"]
        elif pool:
            pool = sorted(pool, key=lambda r: (r["season"], r["round"]), reverse=True)  # newest first, for the same reason
            return {"grand_prix": grand_prix, "winners": [
                {"season": r["season"], "round": r["round"], "race": r["name"], "circuit": _circuit_name(r["circuit"]),
                 "winner": _driver_name(r["results"][0]["driver"]), "team": _team_name(r["results"][0]["team"])} for r in pool if r["results"]],
                "source": SOURCE}
        else:
            return {"error": f"no race found for '{grand_prix}'", "source": SOURCE}
    if season and round:
        race = next((r for r in races if r["season"] == season and r["round"] == round), None)
        if not race:
            return {"error": f"no result for season {season} round {round} (data runs to {d['latest_season']})", "source": SOURCE}
        pole = next((q for q in race["qualifying"] if q["pos"] == 1), None)
        fl = next((r for r in race["results"] if r["fl_rank"] == 1), None)
        return {"season": season, "round": round, "race": race["name"], "date": race["date"], "circuit": _circuit_name(race["circuit"]),
                "pole": _driver_name(pole["driver"]) if pole else None, "fastest_lap": ({"driver": _driver_name(fl["driver"]), "time": fl["fl_time"], "lap": fl["fl_lap"]} if fl else None),
                "classification": [{"pos": r["pos_text"], "driver": _driver_name(r["driver"]), "team": _team_name(r["team"]), "grid": r["grid"],
                                    "laps": r["laps"], "result": r["time"] or r["status"], "points": r["points"]} for r in race["results"]],
                "source": SOURCE}
    if season:
        rs = [r for r in races if r["season"] == season]
        if not rs:
            return {"error": f"no races found for {season} (data runs 1950-{d['latest_season']})", "source": SOURCE}
        return {"season": season, "races": [{"round": r["round"], "race": r["name"], "date": r["date"], "circuit": _circuit_name(r["circuit"]),
                                              "winner": _driver_name(r["results"][0]["driver"]), "team": _team_name(r["results"][0]["team"])} for r in rs],
                "source": SOURCE}
    return {"error": "give a season, or a grand_prix name", "source": SOURCE}


def driver_career(driver: str) -> dict:
    """A driver's whole career: totals, titles, and a season-by-season table."""
    did, cands = find_driver(driver)
    if not did:
        return _ambiguous("driver", driver, cands)
    d = _data()
    rows = [r for r in _rows() if r["driver"] == did]
    if not rows:
        return {"error": f"{_driver_name(did)} has no race starts in the data", "source": SOURCE}
    st = d["standings"]
    by_season = defaultdict(list)
    for r in rows:
        by_season[r["season"]].append(r)
    table = []
    for y in sorted(by_season, reverse=True):  # newest first: if the list is cut to fit the context, the recent seasons stay
        rs = by_season[y]
        final = next((s for s in st.get(str(y), {}).get("drivers", []) if s["driver"] == did), None)
        table.append({"season": y, "teams": sorted({_team_name(r["team"]) for r in rs}), **{k: v for k, v in _summary(rs).items() if k != "fastest_laps"},
                      "championship_position": final["pos"] if final else None})
    p = d["people"]["drivers"][did]
    titles = d["titles"]["driver"].get(did, [])
    return {"driver": p["name"], "driver_id": did, "born": p["born"], "nationality": p["nationality"], "code": p["code"],
            "career": {**_summary(rows), "seasons": _seasons(rows), "world_championships": len(titles), "title_years": titles,
                       "teams": sorted({_team_name(r["team"]) for r in rows}), "first_race": f"{rows[0]['season']} {rows[0]['race']}",
                       "last_race": f"{rows[-1]['season']} {rows[-1]['race']}"},
            "by_season": table, "source": SOURCE, "note": "; ".join(NOTES.values())}


def team_history(team: str) -> dict:
    """A constructor's record: totals, titles, seasons and its winningest drivers."""
    tid, cands = find_team(team)
    if not tid:
        return _ambiguous("team", team, cands)
    d = _data()
    rows = [r for r in _rows() if r["team"] == tid]
    if not rows:
        return {"error": f"{_team_name(tid)} has no race starts in the data", "source": SOURCE}
    wins = Counter(r["driver"] for r in rows if r["win"])
    seasons = sorted({r["season"] for r in rows})
    # a driver-race is one entry; the race counts below are entries, not race weekends
    return {"team": _team_name(tid), "team_id": tid, "nationality": d["people"]["constructors"][tid]["nationality"],
            "seasons": _seasons(rows), "seasons_count": len(seasons), "race_entries": len(rows),
            "wins": sum(r["win"] for r in rows), "poles": sum(r["pole"] for r in rows), "podiums": sum(r["podium"] for r in rows),
            "constructors_titles": d["titles"]["team"].get(tid, []),
            "drivers_titles_with_team": sorted(y for dr, ys in d["titles"]["driver"].items() for y in ys
                                                if any(r["driver"] == dr and r["team"] == tid and r["season"] == y for r in rows)),
            "top_winners": [{"driver": _driver_name(k), "wins": v} for k, v in wins.most_common(5)],
            "also_matches": _also_matches(tid, team),
            "source": SOURCE, "note": "constructors' titles exist from 1958; race_entries counts every car, so a team has two a race"}


def _standings_then(year: int, after_round: int, races_in_season: int) -> dict:
    """Standings as they stood after `after_round`, from Jolpica (only used when replaying a weekend)."""
    from src.agent import f1_api

    drivers = f1_api.get_driver_standings(str(year), after_round)
    teams = f1_api.get_constructor_standings(str(year), after_round)
    by_code = {p["code"]: i for i, p in _data()["people"]["drivers"].items() if p.get("code")}
    return {"rounds": after_round, "races_in_season": races_in_season,
            "drivers": [{"pos": s["position"], "driver": by_code.get(s["code"], f"{s['given_name']} {s['family_name']}"), "teams": [],
                         "team_name": s["team"], "points": s["points"], "wins": s["wins"]} for s in drivers],
            "constructors": [{"pos": t["position"], "team": t["name"], "points": t["points"], "wins": t["wins"]} for t in teams]}


def season_summary(year: int) -> dict:
    """A season: champions, race winners, wins per driver and the final standings."""
    d = _data()
    st = d["standings"].get(str(year))
    rs = [r for r in _races() if r["season"] == year]
    snap = context.snapshot()
    if st and rs and snap and snap[0] == year:
        st = _standings_then(year, snap[1] - 1, st["races_in_season"])
    if not st or not rs:
        return {"error": f"no data for {year} (data runs 1950-{d['latest_season']})", "source": SOURCE}
    complete = bool(st["rounds"] and st["rounds"] >= st["races_in_season"])
    wins = Counter(_driver_name(r["results"][0]["driver"]) for r in rs)
    out = {"season": year, "races_held": len(rs), "races_scheduled": st["races_in_season"], "complete": complete,
           "standings_after_round": st["rounds"],
           "driver_standings": [{"pos": s["pos"], "driver": _driver_name(s["driver"]), "team": _team_name(s["teams"][-1]) if s["teams"] else s.get("team_name"),
                                 "points": s["points"], "wins": s["wins"]} for s in st["drivers"][:10]],
           "constructor_standings": [{"pos": s["pos"], "team": _team_name(s["team"]), "points": s["points"], "wins": s["wins"]} for s in st["constructors"][:10]],
           "wins_by_driver": dict(wins.most_common()),
           "races": [{"round": r["round"], "race": r["name"], "winner": _driver_name(r["results"][0]["driver"])} for r in rs], "source": SOURCE}
    if complete:
        out["drivers_champion"] = out["driver_standings"][0]["driver"]
        if out["constructor_standings"]:
            out["constructors_champion"] = out["constructor_standings"][0]["team"]
    return out


def records(metric: str, by: str = "driver", n: int = 10, driver: str | None = None, team: str | None = None, circuit: str | None = None,
            from_year: int | None = None, to_year: int | None = None) -> dict:
    """A computed leaderboard: `metric` (wins, podiums, poles, points, races,
    fastest_laps, titles) grouped `by` driver, team, circuit or season, optionally
    limited to one driver, team, circuit or a range of years. This is the ONLY
    way to answer "who has the most / how many" questions."""
    if metric not in METRICS or by not in GROUPS:
        return {"error": f"metric must be one of {METRICS} and by one of {GROUPS}", "source": SOURCE}
    d = _data()
    filters = {}
    if driver:
        did, cands = find_driver(driver)
        if not did:
            return _ambiguous("driver", driver, cands)
        filters["driver"] = did
    if team:
        tid, cands = find_team(team)
        if not tid:
            return _ambiguous("team", team, cands)
        filters["team"] = tid
    if circuit:
        cid = find_circuit(circuit)
        if not cid:
            return {"error": f"no circuit or Grand Prix found for '{circuit}'", "source": SOURCE}
        filters["circuit"] = cid

    if metric == "titles":
        if by not in ("driver", "team") or filters.get("circuit"):
            return {"error": "titles can only be grouped by driver or team and have no circuit", "source": SOURCE}
        counts = Counter()
        for who, years in d["titles"][by].items():
            if filters.get(by) and who != filters[by]:
                continue
            counts[who] = sum(1 for y in years if (not from_year or y >= from_year) and (not to_year or y <= to_year))
        name = _driver_name if by == "driver" else _team_name
        top = [{"name": name(k), "titles": v} for k, v in counts.most_common(n) if v]
        return {"metric": "titles", "by": by, "filters": {"from_year": from_year, "to_year": to_year}, "leaders": top,
                "note": "completed seasons only; constructors' titles exist from 1958", "source": SOURCE}

    key = {"wins": "win", "podiums": "podium", "poles": "pole", "fastest_laps": "fl"}.get(metric)
    totals: Counter = Counter()
    for r in _rows():
        if any(r[k] != v for k, v in filters.items()):
            continue
        if (from_year and r["season"] < from_year) or (to_year and r["season"] > to_year):
            continue
        k = r[by]
        totals[k] += 1 if metric == "races" else r["points"] if metric == "points" else int(r[key])
    name = {"driver": _driver_name, "team": _team_name, "circuit": _circuit_name, "season": str}[by]
    leaders = [{"name": name(k), metric: round(v, 1) if metric == "points" else v} for k, v in totals.most_common(n) if v]
    if by == "season":
        leaders.sort(key=lambda x: x["name"])
    out = {"metric": metric, "by": by, "filters": {**{k: v for k, v in (("driver", driver), ("team", team), ("circuit", circuit)) if v},
                                                  "from_year": from_year, "to_year": to_year}, "leaders": leaders,
           "note": NOTES.get(metric, "") + ("; no entry means none (0)" if not leaders else ""), "source": SOURCE}
    if "circuit" in filters:
        held = [r for r in _races() if r["circuit"] == filters["circuit"]]
        out["circuit"] = {"name": _circuit_name(filters["circuit"]), "races_held": len(held), "grands_prix_held_here": sorted({r["name"] for r in held})}
    return out
