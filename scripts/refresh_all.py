"""Bring everything on this machine up to date, in one go (the app's Refresh button runs this).

  python scripts/refresh_all.py            ingest finished races, rebuild features, retrain, republish the forecast
  python scripts/refresh_all.py --quick    skip the retrain (about a minute instead of about ten)

Steps, in order (the same commands the scheduled jobs run):
  1. ingest finished races, the FIA's upgrade lists and the race history (data/history)
  2. rebuild the feature table and retrain every model, gated like .github/workflows/retrain.yml:
     a model that no longer beats its baseline is rolled back to the shipped one, the rest of the refresh carries on
  3. republish the forecast, title odds, ratings, and score the past forecasts against the results

Progress goes to data/processed/refresh_status.json (what GET /refresh/status serves) and the full output to
data/processed/refresh.log. This only changes files on this machine; it never commits or pushes. The scheduled
jobs (scripts/refresh_local.ps1, GitHub Actions) are what publish.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
STATUS, LOG = OUT / "refresh_status.json", OUT / "refresh.log"
# what the retrain writes: restored if any retrain step fails
GUARDED = [ROOT / "src" / "models" / "saved", ROOT / "src" / "strategy" / "params.json", ROOT / "data" / "predictions" / "backtest"]
YEAR = str(datetime.now(timezone.utc).year)

# (label, module + args, fatal): a non-fatal step that fails is reported and skipped
INGEST = [
    ("Pulling finished races", ["src.data.ingest", "--seasons", YEAR], False),
    ("Reading the FIA's upgrade lists", ["src.data.fia", "--seasons", YEAR], False),
    ("Updating the race history", ["src.data.history"], False),
]
RETRAIN = [
    ("Rebuilding the feature table", ["src.features.build_dataset"]),
    ("Retraining the models", ["src.models.train", "--gate"]),
    ("Refitting the blends", ["src.models.blend"]),
    ("Checking against the shipped models", ["src.models.regression_check"]),
    ("Refitting odds and race history", ["src.models.probabilities"]),
    ("Re-running the past-race tests", ["src.models.backtest_export"]),
    ("Refitting tyre strategy", ["src.strategy.model"]),
    ("Updating the README's accuracy table", ["scripts/readme_accuracy.py"]),
]
PUBLISH = [
    ("Publishing the forecast", ["src.models.refresh_job"], True),
    ("Scoring past forecasts", ["src.models.live_record"], False),
]


def write_status(**fields) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}), encoding="utf-8")


def snapshot() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="f1-refresh-"))
    for i, p in enumerate(GUARDED):
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, tmp / str(i))
    return tmp


def restore(tmp: Path) -> None:
    for i, p in enumerate(GUARDED):
        if p.is_dir():
            shutil.rmtree(p)
            shutil.copytree(tmp / str(i), p)
        else:
            shutil.copy2(tmp / str(i), p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="skip the retrain")
    quick = ap.parse_args().quick

    total = len(INGEST) + (0 if quick else len(RETRAIN)) + len(PUBLISH)
    started, done, warnings = datetime.now(timezone.utc).isoformat(timespec="seconds"), 0, []
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = LOG.open("w", encoding="utf-8")

    def run(label: str, args: list[str]) -> bool:
        nonlocal done
        write_status(state="running", started_at=started, step=label, done=done, total=total, warnings=warnings)
        cmd = args if args[0].endswith(".py") else ["-m", *args]
        log.write(f"\n=== {label}: python {' '.join(cmd)}\n")
        log.flush()
        ok = subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUTF8": "1"}).returncode == 0  # a redirected Windows stdout is cp1252: "→" crashes it
        done += 1
        if not ok:
            warnings.append(label)
        return ok

    def fail(label: str) -> int:
        write_status(state="error", started_at=started, step=label, done=done, total=total, warnings=warnings,
                     error=f"{label} failed. The full output is in {LOG.relative_to(ROOT)}.")
        return 1

    for label, args, fatal in INGEST:
        if not run(label, args) and fatal:
            return fail(label)
    if not quick:
        backup = snapshot()
        for n, (label, args) in enumerate(RETRAIN):
            if not run(label, args):
                restore(backup)  # a half-written model set is worse than last week's
                warnings.append("models rolled back to the shipped ones")
                done += len(RETRAIN) - n - 1
                break
        shutil.rmtree(backup, ignore_errors=True)
    for label, args, fatal in PUBLISH:
        if not run(label, args) and fatal:
            return fail(label)

    write_status(state="done", started_at=started, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 step="Up to date", done=total, total=total, warnings=warnings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
