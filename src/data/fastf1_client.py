"""Thin FastF1 wrapper: cache setup + a session loader with telemetry off by default."""
import logging
import time
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fastf1_cache"
# FastF1's own limiter (fastf1/req.py, 500 calls/h) records a refused call
# like a real one, so frequent retries keep it full forever: a 9 s retry did,
# and stalled the 2018-2021 backfill for an hour (2026-10-10). Waiting 5 min
# adds only 12 refusals an hour, so the window clears once the burst ages out.
RATE_LIMIT_WAIT_S = 300

_enabled = False
log = logging.getLogger(__name__)


def enable_cache() -> None:
    global _enabled
    if _enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _enabled = True


def load_session(season: int, round_number: int, session_code: str, laps: bool = True, weather: bool = False,
                 messages: bool = False):
    """Load a session with telemetry off; race-control messages only on request
    (they mark deleted laps and let FastF1 classify a session Ergast lacks).
    Retries with a fixed backoff on FastF1's own rate limit instead of failing the race."""
    enable_cache()
    session = fastf1.get_session(season, round_number, session_code)
    max_attempts = 36  # cap the retry loop at ~3h so a genuinely stuck race doesn't hang forever
    for attempt in range(max_attempts):
        try:
            session.load(laps=laps, telemetry=False, weather=weather, messages=messages)
            return session
        except RateLimitExceededError:
            log.warning("rate limited, waiting %ss (attempt %d/%d)", RATE_LIMIT_WAIT_S, attempt + 1, max_attempts)
            time.sleep(RATE_LIMIT_WAIT_S)
    raise RateLimitExceededError("gave up after repeated rate limiting")


def event_schedule(season: int):
    enable_cache()
    sched = fastf1.get_event_schedule(season)
    return sched[sched["RoundNumber"] > 0].reset_index(drop=True)
