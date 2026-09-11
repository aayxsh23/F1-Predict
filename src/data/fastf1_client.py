"""Thin FastF1 wrapper: cache setup + a session loader with telemetry off by default."""
import logging
import time
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fastf1_cache"
RATE_LIMIT_WAIT_S = 9  # ~500 calls/h refills at 1 call/7.2s; retry a bit above that instead of skipping races

_enabled = False
log = logging.getLogger(__name__)


def enable_cache() -> None:
    global _enabled
    if _enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _enabled = True


def load_session(season: int, round_number: int, session_code: str, laps: bool = True, weather: bool = False):
    """Load a session with telemetry/messages always off (not needed for Phase 1 features).
    Retries with a fixed backoff on FastF1's own rate limit instead of failing the race."""
    enable_cache()
    session = fastf1.get_session(season, round_number, session_code)
    max_attempts = 1200  # cap the retry loop at ~3h so a genuinely stuck race doesn't hang forever
    for attempt in range(max_attempts):
        try:
            session.load(laps=laps, telemetry=False, weather=weather, messages=False)
            return session
        except RateLimitExceededError:
            log.warning("rate limited, waiting %ss (attempt %d/%d)", RATE_LIMIT_WAIT_S, attempt + 1, max_attempts)
            time.sleep(RATE_LIMIT_WAIT_S)
    raise RateLimitExceededError("gave up after repeated rate limiting")


def event_schedule(season: int):
    enable_cache()
    sched = fastf1.get_event_schedule(season)
    return sched[sched["RoundNumber"] > 0].reset_index(drop=True)
