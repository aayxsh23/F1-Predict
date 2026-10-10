"""Every F1 World Championship race since 1950, from Jolpica-F1 (the free
Ergast-compatible API), compiled into three small files in data/history/ that
the chat's history tools read. Pure standard library on the read side, so the
API's requirements.txt stays light.

  races.json      one entry per race: results, qualifying (1994+), circuit
  people.json     drivers, constructors, circuits (with their Wikipedia URLs)
  standings.json  each season's final driver and constructor standings

`python -m src.data.history` fetches (resumable: every response is cached
under data/raw/jolpica_cache/) and then builds. Jolpica asks for at most ~4
requests/second and ~500/hour, so a cold run takes about an hour and a half;
a re-run costs nothing but the current season.
"""
import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = ROOT / "data" / "raw" / "jolpica_cache"
OUT_DIR = ROOT / "data" / "history"
FIRST_SEASON = 1950
MIN_GAP_S = 0.35          # <= ~3 requests/second
HOURLY_CAP = 450          # under Jolpica's 500/hour sustained limit
LIMIT = 100               # Jolpica's maximum page size
HEADERS = {"User-Agent": "F1-Predict/0.1 (https://github.com/aayxsh23/F1-Predict)"}

_sent: list[float] = []   # monotonic times of uncached requests in the last hour


def _throttle() -> None:
    now = time.monotonic()
    _sent[:] = [t for t in _sent if now - t < 3600]
    if len(_sent) >= HOURLY_CAP:
        wait = 3600 - (now - _sent[0]) + 1
        print(f"  hourly cap reached; sleeping {wait / 60:.0f} min", flush=True)
        time.sleep(wait)
    if _sent and (gap := MIN_GAP_S - (time.monotonic() - _sent[-1])) > 0:
        time.sleep(gap)
    _sent.append(time.monotonic())


def get(path: str, params: dict | None = None, use_cache: bool = True) -> dict:
    """MRData for one request, from the disk cache when we've seen it before."""
    params = {"limit": LIMIT, **(params or {})}
    key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    cache_file = CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    delay = 20
    for attempt in range(8):
        _throttle()
        resp = requests.get(f"{BASE_URL}/{path}", params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            print(f"  {resp.status_code} on {path}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        resp.raise_for_status()
        data = resp.json()["MRData"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data
    raise RuntimeError(f"gave up on {key}")


def pages(path: str, table: str, listkey: str, current_season: bool = False):
    """Every item of a paged list, across pages. The current season is never
    served from cache: it changes after every race."""
    offset = 0
    while True:
        data = get(path, {"offset": offset}, use_cache=not current_season)
        yield from data[table][listkey]
        offset += LIMIT
        if offset >= int(data["total"]):
            return


def current_season() -> int:
    return date.today().year


# --- fetching ---

def fetch_all(first: int, last: int) -> None:
    print("people lists", flush=True)
    for path, table, key in (("drivers.json", "DriverTable", "Drivers"), ("constructors.json", "ConstructorTable", "Constructors"),
                             ("circuits.json", "CircuitTable", "Circuits")):
        n = sum(1 for _ in pages(path, table, key))
        print(f"  {path}: {n}", flush=True)
    for season in range(first, last + 1):
        live = season == current_season()
        races = sum(1 for _ in pages(f"{season}/results.json", "RaceTable", "Races", live))
        print(f"{season}: results pages ok ({races} race fragments)", flush=True)
        get(f"{season}/driverStandings.json", use_cache=not live)
        if season >= 1958:
            get(f"{season}/constructorStandings.json", use_cache=not live)
        if season >= 1994:
            list(pages(f"{season}/qualifying.json", "RaceTable", "Races", live))
        get(f"{season}.json", use_cache=not live)


# --- building the compact files ---

def _num(v, cast=int):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def _result(r: dict) -> dict:
    fl = r.get("FastestLap") or {}
    return {
        "pos": _num(r.get("position")), "pos_text": r.get("positionText"), "number": r.get("number"),
        "driver": r["Driver"]["driverId"], "team": r["Constructor"]["constructorId"],
        "grid": _num(r.get("grid")), "laps": _num(r.get("laps")), "status": r.get("status"),
        "points": _num(r.get("points"), float), "time": (r.get("Time") or {}).get("time"),
        "fl_rank": _num(fl.get("rank")), "fl_time": (fl.get("Time") or {}).get("time"),
        "fl_lap": _num(fl.get("lap")),
    }


def _merge_races(season: int, path: str, listkey: str, store: dict, current: bool) -> None:
    """Merge a season's race fragments (Jolpica pages by result row, so one
    race can arrive split across two pages) into `store` keyed by round."""
    for race in pages(path, "RaceTable", "Races", current):
        rnd = int(race["round"])
        entry = store.setdefault(rnd, {"season": season, "round": rnd, "name": race["raceName"], "date": race.get("date"),
                                       "time": race.get("time"), "url": race.get("url"), "circuit": race["Circuit"]["circuitId"],
                                       "results": [], "qualifying": []})
        if listkey == "Results":
            entry["results"] += [_result(r) for r in race.get("Results", [])]
        else:
            entry["qualifying"] += [{"pos": _num(q.get("position")), "driver": q["Driver"]["driverId"], "team": q["Constructor"]["constructorId"],
                                     "q1": q.get("Q1") or None, "q2": q.get("Q2") or None, "q3": q.get("Q3") or None}
                                    for q in race.get("QualifyingResults", [])]


def build(first: int, last: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    people = {"drivers": {}, "constructors": {}, "circuits": {}}
    for d in pages("drivers.json", "DriverTable", "Drivers"):
        people["drivers"][d["driverId"]] = {"name": f"{d['givenName']} {d['familyName']}", "given": d["givenName"], "family": d["familyName"],
                                            "code": d.get("code"), "number": d.get("permanentNumber"), "born": d.get("dateOfBirth"),
                                            "nationality": d.get("nationality"), "url": d.get("url")}
    for c in pages("constructors.json", "ConstructorTable", "Constructors"):
        people["constructors"][c["constructorId"]] = {"name": c["name"], "nationality": c.get("nationality"), "url": c.get("url")}
    for c in pages("circuits.json", "CircuitTable", "Circuits"):
        people["circuits"][c["circuitId"]] = {"name": c["circuitName"], "locality": c["Location"]["locality"], "country": c["Location"]["country"],
                                              "url": c.get("url")}

    races, standings = [], {}
    for season in range(first, last + 1):
        live = season == current_season()
        store: dict[int, dict] = {}
        _merge_races(season, f"{season}/results.json", "Results", store, live)
        if season >= 1994:
            _merge_races(season, f"{season}/qualifying.json", "QualifyingResults", store, live)
        schedule = get(f"{season}.json", use_cache=not live)["RaceTable"]["Races"]
        done =[store[k] for k in sorted(store) if store[k]["results"]]
        for race in done:
            race["results"].sort(key=lambda r: (r["pos"] is None, r["pos"] or 0))
            race["qualifying"].sort(key=lambda q: (q["pos"] is None, q["pos"] or 0))
        races += done

        ds = get(f"{season}/driverStandings.json", use_cache=not live)["StandingsTable"]
        entry = {"rounds": _num(ds.get("round")), "races_in_season": len(schedule), "drivers": [], "constructors": []}
        for lst in ds["StandingsLists"]:
            entry["drivers"] = [{"pos": _num(s.get("position")), "driver": s["Driver"]["driverId"], "teams": [c["constructorId"] for c in s["Constructors"]],
                                 "points": _num(s["points"], float), "wins": _num(s["wins"])} for s in lst["DriverStandings"]]
        if season >= 1958:
            cs = get(f"{season}/constructorStandings.json", use_cache=not live)["StandingsTable"]
            for lst in cs["StandingsLists"]:
                entry["constructors"] = [{"pos": _num(s.get("position")), "team": s["Constructor"]["constructorId"], "points": _num(s["points"], float),
                                          "wins": _num(s["wins"])} for s in lst["ConstructorStandings"]]
        standings[str(season)] = entry

    for name, obj in (("races", races), ("people", people), ("standings", standings)):
        (OUT_DIR / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT_DIR.glob("*.json")) / 1e6
    print(f"built {len(races)} races, {len(people['drivers'])} drivers, {len(people['constructors'])} constructors "
          f"-> {OUT_DIR} ({size:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", type=int, default=FIRST_SEASON)
    ap.add_argument("--last", type=int, default=current_season())
    ap.add_argument("--build-only", action="store_true", help="skip fetching; build from the cache")
    args = ap.parse_args()
    if not args.build_only:
        fetch_all(args.first, args.last)
    build(args.first, args.last)
