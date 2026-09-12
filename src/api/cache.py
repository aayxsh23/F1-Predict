"""Read prediction/backtest JSON with a short in-process TTL cache. Default
source is the local data/predictions/ directory (what refresh_job.py and
backtest_export.py write, and all a local `uvicorn --reload` needs); set
PREDICTIONS_DATA_SOURCE to a raw.githubusercontent.com base URL in production
so a cold-started free-tier backend always serves whatever the GitHub Actions
workflow last committed, without needing to have been awake when it ran (see
phase7-ui-backend-plan.md's "freshness lives in GitHub, not backend uptime")."""
import json
import os
import time
from pathlib import Path

import requests

TTL_SECONDS = 60
LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"
REMOTE_BASE_URL = os.environ.get("PREDICTIONS_DATA_SOURCE")

_cache: dict[str, tuple[float, dict]] = {}


def get_json(relative_path: str) -> dict:
    """relative_path e.g. 'latest.json', '2026_14.json', 'backtest/index.json'."""
    now = time.monotonic()
    cached = _cache.get(relative_path)
    if cached and now - cached[0] < TTL_SECONDS:
        return cached[1]

    if REMOTE_BASE_URL:
        resp = requests.get(f"{REMOTE_BASE_URL}/{relative_path}", timeout=10)
        if resp.status_code == 404:
            raise FileNotFoundError(relative_path)
        resp.raise_for_status()
        data = resp.json()
    else:
        path = LOCAL_DATA_DIR / relative_path
        if not path.exists():
            raise FileNotFoundError(relative_path)
        data = json.loads(path.read_text())

    _cache[relative_path] = (now, data)
    return data
