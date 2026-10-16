"""Record the API responses the frontend's browser smoke tests replay
(frontend/e2e/fixtures/), from the real FastAPI app over the committed data.
Re-run after an API change: `python scripts/e2e_fixtures.py`.
"""
import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.main import app  # noqa: E402

OUT = ROOT / "frontend" / "e2e" / "fixtures"


def name_for(path: str) -> str:
    """'/predictions/2026/17?x=1' -> 'predictions_2026_17__x_1.json' (the spec maps URLs the same way)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/")).strip("_") + ".json"


def main() -> None:
    client = TestClient(app)
    latest = client.get("/predictions/latest").json()
    season, rnd = latest["season"], latest["round"]
    races = client.get("/backtest/races").json()
    past = races[-1]
    driver = latest["drivers"][0]["driver"]
    paths = [
        "/health", "/predictions/latest", f"/predictions/{season}/{rnd}", f"/predictions/{season}/{rnd}/timeline",
        f"/races?season={season}", "/championship", "/backtest/races", "/backtest/summary",
        f"/backtest/{past['season']}/{past['round']}", "/live-record", "/model", f"/strategy/{season}/{rnd}",
        f"/strategy/{season}/{rnd}?sc_lap=15", "/regulations", "/ratings",
        f"/predictions/{season}/{rnd}/explain?driver={driver}&target=finish_position",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.json"):
        stale.unlink()
    for p in paths:
        r = client.get(p)
        if r.status_code != 200:
            raise SystemExit(f"{p}: HTTP {r.status_code}")
        (OUT / name_for(p)).write_text(json.dumps(r.json(), separators=(",", ":")), encoding="utf-8")
    meta = {"season": season, "round": rnd, "driver": driver, "past": past}
    (OUT / "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"wrote {len(paths)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
