# Experiments

One script per modelling decision, so every number a decision cites can be re-run (phase 10, item 3.1; before this, PROGRESS.md often said "scratch scripts weren't kept").

Each script runs candidates through the same walk-forward as the shipped models (`src/models/compare.py`), prints the comparison and records it in `src/models/saved/leaderboard.json`. Rules (AGENTS.md "Evaluate honestly"): a choice is made on the tuning-window races; the held-out races and their race-level 95% interval only check it. A winner is switched on in `src/models/saved/model_specs.json`, which `src/models/train.py` reads.

Run from the repo root after `python -m src.features.build_dataset`, e.g. `python -m experiments.two_part_race_time`. Walk-forward predictions are cached in `data/processed/candidates/` (gitignored); pass `--fresh` to recompute.

| Script | Question | Outcome |
|---|---|---|
| `two_part_race_time.py` | Gap to the winner: one regression, or "lapped or not" then a regression per part? | Probability-weighted two-part model **shipped**: tuning-window MAE after qualifying 0.512 vs 0.527, better at every stage; held-out intervals straddle zero (0.650 vs 0.657 after qualifying). The hard switch was worse (a wrong guess costs a lap). |
| `ensemble_xgb_lgbm.py` | Does averaging XGBoost with LightGBM beat XGBoost alone? | **Shipped** for finishing position (LightGBM weight 0.7; held-out before practice 3.574 vs 3.605, interval below zero) and qualifying (0.45; held-out after practice 0.731 vs 0.740, interval below zero). Places gained: no gain, stays XGBoost. |
| `circuit_history.py` | Circuit facts measured from earlier races instead of hand-typed? | **Shipped** for finishing position and places gained (better on tuning races, held-out level); qualifying and gap to the winner keep the typed facts (measured ones worse on both windows). |
| `weather_lead_time.py` | Forecasts matched to each stage's lead time (3 days before practice, 1 day on Saturday)? | **Shipped** as a train/serve fix: held-out level or slightly better everywhere (finishing after qualifying 3.186 vs 3.189, interval below zero; gap to the winner 0.649 vs 0.657). The tuning window has almost no 2024+ races, so it couldn't choose. |
| `practice_pace.py` | Practice pace per session (weighted FP1 1 / FP2 2 / FP3 3, or the latest session) instead of one best lap across sessions? | **Not shipped**: qualifying worse on the tuning races with both (0.743 / 0.749 vs 0.734 after practice); session-weighted clearly worse for finishing position held-out (interval +0.003 to +0.022). Dropping track-limits-deleted laps is untested: it needs race-control messages the cache doesn't hold. |
