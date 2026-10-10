"""Is the live forecast keeping up with the race weekend? Standard library
only, so GitHub can run it without installing anything
(.github/workflows/freshness.yml), which matters because the refresh itself
runs on a home PC that can be switched off.

Stale means, during a race weekend (an hour before the first session until
three hours after the race): the published forecast is for another round, or
a session that started more than DUE_HOURS ago isn't in it. Prints
`stale=true|false` and `reason=...` lines (GitHub step outputs).
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DUE_HOURS = 4  # an hour of session, about an hour for the data, and slack for the 30-minute refresh
CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Sprint Qualifying": "SQ",
         "Sprint Shootout": "SQ", "Sprint": "S", "Qualifying": "Q"}


def _t(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if s else None


def check(schedule: list[dict], latest: dict, now: datetime) -> tuple[bool, str]:
    for e in schedule:
        first, race = _t(e.get("Session1DateUtc")), _t(e.get("RaceStartUtc"))
        if first and race and first - timedelta(hours=1) <= now <= race + timedelta(hours=3):
            break
    else:
        return False, "not a race weekend"
    name = f"round {e['RoundNumber']} ({e.get('Location')})"
    if latest.get("round") != e["RoundNumber"] or latest.get("season") != now.year:
        return True, f"the published forecast is for round {latest.get('round')}, but {name} is underway"
    due = [CODES[e[f"Session{n}"]] for n in range(1, 6)
           if e.get(f"Session{n}") in CODES and _t(e.get(f"Session{n}DateUtc")) and _t(e[f"Session{n}DateUtc"]) + timedelta(hours=DUE_HOURS) < now]
    missing = [c for c in due if c not in latest.get("sessions", [])]
    if missing:
        return True, (f"{name}: {', '.join(missing)} ran but the published forecast doesn't include it "
                      f"(last generated {latest.get('generated_at')}). Is the home PC on and the refresh task running?")
    return False, f"{name}: forecast includes every session that has run ({latest.get('session_label')})"


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "predictions"
    now = datetime.now(timezone.utc)
    path = root / "schedule" / f"{now.year}.json"
    if not path.exists():
        print("stale=false\nreason=no schedule for this season yet")
        return
    stale, reason = check(json.loads(path.read_text(encoding="utf-8")), json.loads((root / "latest.json").read_text()), now)
    print(f"stale={'true' if stale else 'false'}\nreason={reason}")


if __name__ == "__main__":
    sys.exit(main())
