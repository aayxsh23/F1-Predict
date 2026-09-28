"""Live championship data from Jolpica-F1 (api.jolpi.ca), the free
Ergast-compatible API. Responses are cached for five minutes: standings only
change after a session, and Jolpica rate-limits."""
import time

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
TTL_S = 300
_cache: dict[str, tuple[float, dict]] = {}


def _get(path: str) -> dict:
    hit = _cache.get(path)
    if hit and time.monotonic() - hit[0] < TTL_S:
        return hit[1]
    resp = requests.get(f"{BASE_URL}/{path}", timeout=10)
    resp.raise_for_status()
    data = resp.json()["MRData"]
    _cache[path] = (time.monotonic(), data)
    return data


def get_driver_standings(season: str = "current") -> list[dict]:
    """Each: {code, given_name, family_name, team, points, wins, position}."""
    lists = _get(f"{season}/driverStandings.json")["StandingsTable"]["StandingsLists"]
    if not lists:
        return []
    return [
        {
            "code": d["Driver"].get("code") or d["Driver"]["familyName"][:3].upper(),
            "given_name": d["Driver"]["givenName"],
            "family_name": d["Driver"]["familyName"],
            "team": d["Constructors"][-1]["name"] if d["Constructors"] else None,
            "points": float(d["points"]),
            "wins": int(d["wins"]),
            "position": int(d.get("position") or d.get("positionText", 0) or 0),
        }
        for d in lists[0]["DriverStandings"]
    ]


def get_constructor_standings(season: str = "current") -> list[dict]:
    """Each: {name, points, wins, position}."""
    lists = _get(f"{season}/constructorStandings.json")["StandingsTable"]["StandingsLists"]
    if not lists:
        return []
    return [
        {"name": c["Constructor"]["name"], "points": float(c["points"]), "wins": int(c["wins"]), "position": int(c["position"])}
        for c in lists[0]["ConstructorStandings"]
    ]


def get_current_round(season: str = "current") -> int:
    """The latest completed round, the one the standings reflect."""
    return int(_get(f"{season}/driverStandings.json")["StandingsTable"].get("round", 0))


def get_season_schedule(season: str = "current") -> list[dict]:
    """Each: {round, race_name, date, is_sprint}."""
    races = _get(f"{season}.json")["RaceTable"]["Races"]
    return [{"round": int(r["round"]), "race_name": r["raceName"], "date": r["date"], "is_sprint": "Sprint" in r} for r in races]
