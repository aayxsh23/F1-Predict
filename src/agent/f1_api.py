"""Thin client for Jolpica-F1 (api.jolpi.ca) -- the free, open-source,
Ergast-compatible successor API this project identified as the live-standings
data source (see the project plan's "Current state" section). Separate from
FastF1 (Phase 1's data source): FastF1 gives rich session telemetry for
already-completed sessions, Jolpica gives simple, fast standings/schedule
lookups for "where does the championship stand right now"."""
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def get_driver_standings(season: str = "current") -> list[dict]:
    """Current points/wins per driver, most recent completed round. Each dict:
    {code, given_name, family_name, team, points, wins, position}."""
    data = requests.get(f"{BASE_URL}/{season}/driverStandings.json", timeout=10).json()
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return []
    return [
        {
            "code": d["Driver"]["code"],
            "given_name": d["Driver"]["givenName"],
            "family_name": d["Driver"]["familyName"],
            "team": d["Constructors"][0]["name"] if d["Constructors"] else None,
            "points": float(d["points"]),
            "wins": int(d["wins"]),
            "position": int(d["position"]),
        }
        for d in lists[0]["DriverStandings"]
    ]


def get_constructor_standings(season: str = "current") -> list[dict]:
    """Current points/wins per constructor. Each dict: {name, points, wins, position}."""
    data = requests.get(f"{BASE_URL}/{season}/constructorStandings.json", timeout=10).json()
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return []
    return [
        {
            "name": c["Constructor"]["name"],
            "points": float(c["points"]),
            "wins": int(c["wins"]),
            "position": int(c["position"]),
        }
        for c in lists[0]["ConstructorStandings"]
    ]


def get_current_round(season: str = "current") -> int:
    """The most recently completed round -- what the live standings above
    actually reflect. Needed to know which rounds are "remaining" for
    scenario math (src/agent/scenarios.py)."""
    data = requests.get(f"{BASE_URL}/{season}/driverStandings.json", timeout=10).json()
    return int(data["MRData"]["StandingsTable"]["round"])


def get_season_schedule(season: str = "current") -> list[dict]:
    """Every round of the season with its date and whether it's a sprint
    weekend (sprint rounds award extra points, needed for remaining-points math).
    Each dict: {round, race_name, date, is_sprint}."""
    data = requests.get(f"{BASE_URL}/{season}.json", timeout=10).json()
    races = data["MRData"]["RaceTable"]["Races"]
    return [
        {
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "date": r["date"],
            "is_sprint": "Sprint" in r,
        }
        for r in races
    ]
