"""Spend only free credits: check before every AWS spend, record after it.

AWS posts Bedrock charges a day or two late, so its "remaining credits" figure
runs ahead of reality. The guard therefore trusts the LOWER of
  * AWS's remaining credits (catches spending outside this project), and
  * the starting credits minus everything this project has recorded spending
    (catches our own not-yet-posted charges),
and refuses a run whose planned cost would leave less than MARGIN_USD.

    python -m training.budget_guard            # show the numbers
"""
import json
import time
from pathlib import Path

import boto3

CREDITS_START_USD = 200.0  # what the account started with (AWS sign-up + activity credits)
MARGIN_USD = 15.0          # always keep this much credit unspent, for billing lag and rounding
LEDGER = Path(__file__).resolve().parent / "data" / "spend_ledger.json"


class OutOfCredits(SystemExit):
    pass


def _ledger() -> list[dict]:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []


def record(what: str, usd: float, key: str | None = None) -> None:
    """Add (or, with the same `key`, replace) one spend entry."""
    entries = [e for e in _ledger() if not key or e.get("key") != key]
    entries.append({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "usd": round(usd, 3), **({"key": key} if key else {})})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(entries, indent=1), encoding="utf-8")


def recorded_total() -> float:
    return round(sum(e["usd"] for e in _ledger()), 2)


def aws_remaining() -> float | None:
    try:
        state = boto3.client("freetier", region_name="us-east-1").get_account_plan_state()
        return float(state["accountPlanRemainingCredits"]["amount"])
    except Exception:
        return None  # API unavailable: rely on our own ledger


def effective_remaining() -> float:
    ours = CREDITS_START_USD - recorded_total()
    theirs = aws_remaining()
    return round(min(ours, theirs) if theirs is not None else ours, 2)


def check(planned_usd: float, what: str) -> None:
    """Refuse to start `what` if it could eat into the safety margin."""
    left = effective_remaining()
    if left - planned_usd < MARGIN_USD:
        raise OutOfCredits(f"REFUSED: {what} could cost up to ${planned_usd:.2f}, but only ${left:.2f} of credits are left "
                           f"(keeping ${MARGIN_USD:.0f} spare). Nothing was started.")
    print(f"credit check ok: {what} up to ${planned_usd:.2f}; ${left:.2f} of credits left before it", flush=True)


if __name__ == "__main__":
    print(json.dumps({"aws_reports_remaining": aws_remaining(), "recorded_project_spend": recorded_total(),
                      "credits_start": CREDITS_START_USD, "effective_remaining": effective_remaining(), "margin": MARGIN_USD}, indent=1))
