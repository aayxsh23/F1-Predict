"""Which moment the chat tools answer for. Normally "now", and nothing changes.

The training-data generator replays past race weekends (a stored forecast
snapshot plus the date it was made) so the fine-tuned model learns to read
whatever forecast it is given instead of memorising one weekend's numbers.
Inside `as_of(...)` every tool behaves as on that day: "this weekend" is the
snapshot's race, standings are the ones before it, history stops before it.
`variant` picks which stored version of that weekend's forecast to read
("pre" = before practice, "q" = after qualifying, "" = the original file).
Thread-local, because the generator plays many conversations at once.
"""
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone

_local = threading.local()


@contextmanager
def as_of(season: int, round: int, today: date, variant: str = ""):
    previous = getattr(_local, "state", None)
    _local.state = (season, round, today, variant)
    try:
        yield
    finally:
        _local.state = previous


def snapshot() -> tuple[int, int] | None:
    """(season, round) of the replayed weekend, or None for the live one."""
    state = getattr(_local, "state", None)
    return (state[0], state[1]) if state else None


def variant() -> str:
    state = getattr(_local, "state", None)
    return state[3] if state else ""


def today() -> date:
    state = getattr(_local, "state", None)
    return state[2] if state else datetime.now(timezone.utc).date()
