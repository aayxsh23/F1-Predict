"""Forecast snapshots of past race weekends, for the training-data generator.

Every "who wins this weekend?" conversation would otherwise read the same
forecast, and the fine-tuned model would learn those numbers instead of
learning to read whatever forecast it is given. So past rounds are
re-forecast with the current pipeline, at two stages, and written to
data/predictions/snapshots/ (gitignored; the app never reads it):

    python -m training.snapshots --season 2025 --rounds 2-24 --stage both

  <season>_<round>_pre.json   before practice (nothing known yet)
  <season>_<round>_q.json     after qualifying (practice and grid known)
  (<season>_<round>.json      older after-qualifying builds, still used)

Each is built "as it was": the history table is cut before that race, and for
the pre stage the practice and qualifying data are hidden too. These are
hindsight forecasts (today's models were trained on those races), which is fine
for teaching the chat model to read a forecast; they are not accuracy claims.
"""
import argparse
import json
from datetime import datetime, timezone

from src.features import build_dataset
from src.models import live_predict, refresh_job
from src.models.refresh_job import OUT_DIR, build_prediction_payload

SNAP_DIR = OUT_DIR / "snapshots"


def _history_before(season: int, rnd: int):
    """load_raw() as it was before this race. The forecast pipeline assumes the
    race hasn't happened; if its results are already in the table, the live rows
    join against them and every driver is duplicated with clashing features."""
    full = build_dataset.load_raw()
    before = full[(full["season"] < season) | ((full["season"] == season) & (full["round"] < rnd))].reset_index(drop=True)
    return lambda *a, **k: before.copy()


def _not_yet(*a, **k):
    raise RuntimeError("session not run yet")  # build_live_rows treats this as "no data", exactly as on a Wednesday


def build(season: int, rnd: int, stage: str) -> None:
    path = SNAP_DIR / f"{season}_{rnd}_{stage}.json"
    if path.exists():
        print(f"snapshot {season} R{rnd} {stage}: exists, skipped", flush=True)
        return
    originals = {"load_raw": build_dataset.load_raw, "practice": live_predict.practice_features, "quali": live_predict.quali_features}
    history = _history_before(season, rnd)
    build_dataset.load_raw = refresh_job.load_raw = history
    if stage == "pre":
        live_predict.practice_features = live_predict.quali_features = _not_yet
    try:
        payload = build_prediction_payload(season, rnd)
    finally:
        build_dataset.load_raw = refresh_job.load_raw = originals["load_raw"]
        live_predict.practice_features, live_predict.quali_features = originals["practice"], originals["quali"]
    if len(payload["drivers"]) != len({d["driver"] for d in payload["drivers"]}):
        raise RuntimeError(f"duplicate drivers in the {season} R{rnd} snapshot")
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["snapshot_note"] = "re-forecast after the fact for training data; not an accuracy claim"
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    timeline = [{"label": payload["session_label"], "stage": payload["stage"], "generated_at": payload["generated_at"],
                 "drivers": [{k: d[k] for k in ("driver", "predicted_finish_position", "predicted_qualifying_gap_pct", "predicted_race_gap_pct")}
                             for d in payload["drivers"]]}]
    (SNAP_DIR / "timeline").mkdir(exist_ok=True)
    (SNAP_DIR / "timeline" / path.name).write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print(f"snapshot {season} R{rnd} {stage}: {payload['location']}, {payload['session_label']}", flush=True)


def _rounds(spec: list[str]) -> list[int]:
    out = []
    for s in spec:
        a, _, b = s.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--rounds", nargs="+", required=True, help="e.g. 2-16 or 3 5 7")
    ap.add_argument("--stage", choices=["pre", "q", "both"], default="both")
    args = ap.parse_args()
    for r in _rounds(args.rounds):
        for stage in (["pre", "q"] if args.stage == "both" else [args.stage]):
            try:
                build(args.season, r, stage)
            except Exception as exc:  # one weekend failing must not stop the rest
                print(f"snapshot {args.season} R{r} {stage} FAILED: {type(exc).__name__}: {exc}", flush=True)
