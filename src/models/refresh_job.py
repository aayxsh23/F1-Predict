"""Scheduled job (GitHub Actions): forecast the next or current race and write
data/predictions/, which the API serves. The only place live FastF1 calls
happen; never inside a user request.

Writes, for season S round R:
  S_R.json and latest.json   the forecast (per driver: all four predictions,
                             absolute lap/race times, and each prediction's
                             SHAP breakdown, so serving needs no XGBoost)
  timeline/S_R.json          one snapshot per weekend session, so the app can
                             show how the forecast moved after FP2 or qualifying
  schedule/S.json            the season calendar
Nothing is rewritten when the forecast hasn't changed, so a quiet run makes no
commit.

A session that has run but can't be loaded fails the job instead of being
treated as "not run yet": F1's live-timing server refuses GitHub's runners,
and a forecast silently stuck at "Before practice" is worse than a red run.
Live weekends are refreshed from a home connection (scripts/refresh_local.ps1,
`--weekend-only`); GitHub only rolls the forecast over between weekends.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.fastf1_client import event_schedule
from src.data.ingest import race_start
from src.features.build_dataset import load_raw
from src.models import estimates
from src.models.explain import shown_explanation
from src.models.features import FEATURE_COLS, stage_of
from src.models.live_predict import build_live_rows, event_info
from src.models.predict import MODEL_DIR, TARGETS, predict_all

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"
RACE_OVER_AFTER = pd.Timedelta(hours=3)
# a session's data is expected this long after its scheduled start (session
# length plus FastF1's publishing delay); later than that, missing = failed
DATA_DUE_AFTER = pd.Timedelta(hours=2)
SESSION_CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Sprint Qualifying": "SQ",
                 "Sprint Shootout": "SQ", "Sprint": "S", "Qualifying": "Q"}
LABELS = {"Q": "After qualifying", "S": "After the Sprint", "SQ": "After Sprint Qualifying"}
# every session's name and start, so scripts/check_freshness.py can tell from
# GitHub (no FastF1 there) which sessions a live forecast should already include
SESSION_TIME_COLS = [f"Session{n}DateUtc" for n in range(1, 6)]
SCHEDULE_COLS = (["RoundNumber", "EventName", "Location", "Country", "EventFormat", "EventDate"]
                 + [f"Session{n}" for n in range(1, 6)] + SESSION_TIME_COLS)


def _clean(v):
    # round() also stops float32 -> float64 widening noise (0.2 -> 0.20000000298)
    return None if pd.isna(v) else round(float(v), 4)


def next_or_current_round(season: int) -> int:
    """The first round whose race hasn't finished yet, else the last one."""
    sched = event_schedule(season)
    now = pd.Timestamp.utcnow().tz_localize(None)
    for _, e in sched.iterrows():
        start = race_start(e)
        if pd.notna(start) and start + RACE_OVER_AFTER > now:
            return int(e["RoundNumber"])
    return int(sched.iloc[-1]["RoundNumber"])


def session_label(sessions: list[str], stage: str) -> str:
    if stage == "race_day":
        return "Official grid"
    for code, label in LABELS.items():
        if code in sessions:
            return label
    fp = [s for s in sessions if s.startswith("FP")]
    return f"After {fp[-1]}" if fp else "Before practice"


def overdue_sessions(event: pd.Series, loaded: list[str], now: pd.Timestamp) -> list[str]:
    """Sessions that should have data by now but didn't load."""
    due = []
    for n in range(1, 6):
        code, start = SESSION_CODES.get(event.get(f"Session{n}")), event.get(f"Session{n}DateUtc")
        if code and pd.notna(start) and start + DATA_DUE_AFTER < now and code not in loaded:
            due.append(code)
    return due


def in_race_weekend(event: pd.Series, now: pd.Timestamp) -> bool:
    """From an hour before the first session until the race result is in."""
    return event["Session1DateUtc"] - pd.Timedelta(hours=1) <= now <= race_start(event) + RACE_OVER_AFTER


def build_prediction_payload(season: int, round_number: int) -> dict:
    rows, sessions = build_live_rows(season, round_number)
    stage = stage_of(rows)
    event = event_info(season, round_number)
    sprint_weekend = bool(rows["is_sprint_weekend"].eq(1).any())
    p = predict_all(rows).sort_values("predicted_finish_position").reset_index(drop=True)
    location = event["Location"]

    raw = load_raw()
    fastest = p["practice_fastest_s"].iloc[0]
    pole_s = estimates.pole_time(raw, location, fastest)
    duration_s = estimates.race_duration(raw, location, season, fastest)
    # gaps anchored on the predicted pole-sitter / winner, so P1 reads 0.000
    q_rel = p["predicted_qualifying_gap_pct"] - p["predicted_qualifying_gap_pct"].min()
    r_rel = p["predicted_race_gap_pct"] - p["predicted_race_gap_pct"].min()

    drivers = []
    for i, row in p.iterrows():
        drivers.append({
            "driver": row["driver"], "team": row["team"],
            "driver_number": int(row["driver_number"]) if str(row["driver_number"]).isdigit() else None,
            "grid_position": _clean(row["grid_position"]),
            "quali_gap_to_pole": _clean(row["quali_gap_to_pole"]),
            "quali_gap_pct": _clean(row["quali_gap_pct"]),
            "practice_pace": _clean(row["practice_pace"]),
            "practice_long_run_pace": _clean(row["practice_long_run_pace"]),
            "predicted_qualifying_gap_pct": _clean(row["predicted_qualifying_gap_pct"]),
            "predicted_qualifying_gap": _clean(q_rel[i] / 100 * pole_s),
            "predicted_quali_lap_s": _clean(pole_s * (1 + q_rel[i] / 100)),
            "predicted_finish_position": _clean(row["predicted_finish_position"]),
            "predicted_quali_to_race_delta": _clean(row["predicted_quali_to_race_delta"]),
            "predicted_race_gap_pct": _clean(row["predicted_race_gap_pct"]),
            "predicted_race_time_gap": _clean(r_rel[i] / 100 * duration_s),
            "feature_row": {c: row[c] if isinstance(row[c], str) else _clean(row[c]) for c in FEATURE_COLS},
            # the "why" behind each prediction, precomputed so serving needs no XGBoost
            "quali_nolap_rate": _clean(row.get("driver_quali_nolap_rate")),
            "explanations": {t: shown_explanation(t, stage, p.iloc[[i]], p, top_n=10) for t in TARGETS},
        })

    trained = json.loads((MODEL_DIR / "finish_position_metrics.json").read_text()).get("trained_at")
    return {
        "season": season, "round": round_number, "location": location,
        "event_name": event["EventName"], "country": event["Country"],
        "race_start_utc": _iso(race_start(event)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "sessions": sessions, "session_label": session_label(sessions, stage),
        "sprint_weekend": sprint_weekend,
        "known_sessions": {
            "practice": bool(p["practice_pace"].notna().any()),
            # sprint weekends only, so the app shows these two steps just when they exist
            **({"sprint_qualifying": bool(p["sprint_quali_gap_pct"].notna().any()),
                "sprint": bool(p["sprint_finish_position"].notna().any())} if sprint_weekend else {}),
            "qualifying": bool(p["quali_gap_pct"].notna().any()),
            "grid": bool(p["grid_official"].eq(1).any()),  # the FIA's official grid, penalties applied
        },
        "model_trained_at": trained,
        "race": {
            "laps": estimates.race_laps(raw, location, season),
            "expected_duration_s": _clean(duration_s),
            "pole_time_estimate_s": _clean(pole_s),
            "safety_car_probability": _clean(estimates.safety_car_probability(raw, location)),
        },
        "drivers": drivers,
    }


def _iso(ts) -> str | None:
    return None if pd.isna(ts) else pd.Timestamp(ts).isoformat()


def _comparable(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("generated_at", "model_trained_at")}


def write_payload(payload: dict) -> bool:
    """Write the forecast and its timeline snapshot; False if nothing changed."""
    path = OUT_DIR / f"{payload['season']}_{payload['round']}.json"
    if path.exists() and _comparable(json.loads(path.read_text())) == _comparable(payload):
        return False
    body = json.dumps(payload, indent=2)
    path.write_text(body)
    (OUT_DIR / "latest.json").write_text(body)

    tl_path = OUT_DIR / "timeline" / path.name
    tl_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = json.loads(tl_path.read_text()) if tl_path.exists() else []
    snap = {
        "label": payload["session_label"], "stage": payload["stage"], "generated_at": payload["generated_at"],
        "drivers": [{k: d[k] for k in ("driver", "predicted_finish_position", "predicted_qualifying_gap_pct", "predicted_race_gap_pct")}
                    for d in payload["drivers"]],
    }
    timeline = [s for s in timeline if s["label"] != snap["label"]] + [snap]
    tl_path.write_text(json.dumps(timeline, indent=2))
    return True


def write_schedule(season: int) -> None:
    sched = event_schedule(season)
    out = sched[SCHEDULE_COLS].copy()
    out["RaceStartUtc"] = [race_start(e) for _, e in sched.iterrows()]
    for c in ("EventDate", "RaceStartUtc", *SESSION_TIME_COLS):
        out[c] = out[c].apply(_iso)
    path = OUT_DIR / "schedule" / f"{season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out.to_dict("records"), indent=2, ensure_ascii=False)
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        path.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekend-only", action="store_true", help="do nothing outside a race weekend")
    args = parser.parse_args()
    season = datetime.now(timezone.utc).year
    write_schedule(season)
    round_number = next_or_current_round(season)
    event, now = event_info(season, round_number), pd.Timestamp.utcnow().tz_localize(None)
    if args.weekend_only and not in_race_weekend(event, now):
        print(f"not a race weekend (next: round {round_number}), nothing to do")
        return
    payload = build_prediction_payload(season, round_number)
    if overdue := overdue_sessions(event, payload["sessions"], now):
        raise SystemExit(f"{', '.join(overdue)} should have data by now but FastF1 couldn't load it. Not publishing a "
                         "forecast that ignores it. (F1 live timing refuses GitHub's runners: race weekends are "
                         "refreshed by scripts/refresh_local.ps1 from a home connection.)")
    changed = write_payload(payload)
    print(f"{'wrote' if changed else 'unchanged:'} season {season} round {round_number} ({payload['location']}, {payload['session_label']})")


if __name__ == "__main__":
    main()
