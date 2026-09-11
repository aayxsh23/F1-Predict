# LEARNING.md

This file teaches *why* the project is built the way it is, phase by phase — not just what happened. Three files, three jobs, don't confuse them:

- [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) — the **design**: what we're building and the phase plan.
- [PROGRESS.md](PROGRESS.md) — the **build log**: terse, factual, "what happened," newest first.
- **This file** — the **tutorial**: concepts, reasoning, code walkthroughs, what a naive/wrong version would have gotten wrong. Read this when you want to understand the material, not just the status.

Agents: update this file per AGENTS.md's "Multi-agent coordination" section — add depth, don't just log events.

---

## Interlude — why a venv, specifically

Every `pip install` run this session (both mine and a manual `pip install -r requirements.txt` from the user) had been going into the **global** Python interpreter, because no virtual environment had been created yet. This is a bigger deal than "it works either way, one's just tidier":

- **No isolation between projects.** Your global Python is shared by every script and tool on the machine. If this project needs `pandas 2.3` and some other project on your machine needs `pandas 1.5`, installing globally means whichever installs last wins — and the other project silently breaks the next time you run it, often with no obvious error pointing at the real cause (it just imports the wrong version).
- **No reproducibility.** `requirements.txt` is supposed to be *the* complete list of what this project needs. If you've been installing globally, it's easy to end up relying on some other package that happens to already be on your machine for unrelated reasons — the project *looks* like it works for you, but `requirements.txt` doesn't actually capture everything it depends on, and it breaks for anyone else (or you, on a fresh machine) who installs only from that file.
- **Cleanup is much harder after the fact.** Once packages are in the global interpreter, "uninstall everything this project needed" means manually working out which packages were project-specific vs. already there for some other reason — exactly the cleanup we just did (`pip uninstall -y -r requirements.txt` plus one stray dependency, `optuna`, that wasn't even tracked in `requirements.txt`). A venv sidesteps the whole problem: it's just a directory, and "clean slate" is `rm -rf .venv && python -m venv .venv`.

One extra gotcha worth remembering because it cost real time here: `python -m venv <dir>` does **not** wipe an existing directory — if `<dir>` already has files in it (say, from an earlier half-finished attempt, or an IDE auto-creating one), running `venv` again just layers the venv structure on top without clearing what's there. The symptom is confusing: a "freshly created" venv that already has unrelated packages installed. The fix is to actually delete the directory first if you're not sure of its history, then recreate and verify with `pip list` that it's genuinely empty before trusting it.

## Phase 2.5 (unplanned) — from backtesting to a real live prediction

### Backtesting vs. forecasting — a distinction worth being precise about

Everything through Phase 2 answers one question: *"if this model had existed before a race that already happened, using only pre-race information, would it have called that race correctly?"* That's what time-based CV measures, and it's the right way to prove a model has real skill before trusting it on anything. But it's a fundamentally different question from *"what will happen in a race that hasn't been run yet?"* — the second one needs current data (not just a fixed historical window) and a way to assemble a feature row for a race with no known outcome, which nothing in Phase 1-2 actually built. Recognizing that distinction is what "when can I predict Madrid" surfaced — the model itself was always capable of it, the *plumbing* to feed it a live row wasn't there yet.

### Why `live_predict.py` doesn't reimplement anything

The tempting-but-wrong approach here would be writing a parallel "live feature builder" that recomputes driver form, team form, etc. from scratch for one race. That's a trap: any subtle difference between that logic and the training-time feature layers (`driver_features.py`, `team_features.py`, ...) would silently make live predictions inconsistent with what the model was actually trained on — a live prediction that "looks like" a training row but isn't quite.

The fix: [src/features/build_dataset.py](src/features/build_dataset.py)'s `build()` was refactored to accept an already-assembled raw table instead of hardcoding `load_raw()`. `live_predict.py` then does the minimal possible new work — build a real-world raw row for the upcoming race in the exact shape `ingest_race()` produces for a historical one — and appends it to the historical table before calling the *identical* `build()` pipeline. Every rolling/historical feature (`recent_form`, `track_form`, `team_track_type_form`, etc.) is now computed by the literal same function calls used in training, on a table that happens to have one extra row at the end. There's no way for training and live prediction to drift apart, because there's only one implementation.

This also means the "3-stage" mental model (post-practice / post-quali / pre-race) from Phase 2's `mask_for_stage()` doesn't actually need to be built as a rigid stage system for real live use — it was a *simulation* of partial information for testing on a historical row. For a genuinely upcoming race, you don't need to simulate "pretend this isn't known yet," because it actually isn't known yet: attempting to load FP2 before FP2 has run just fails naturally (`fastf1.exceptions`/`DataNotLoadedError`), gets caught, and the column stays `NaN` — the real world does the masking for you. The staged framing was the right way to *think* about and *test* the mechanism; the live implementation is simpler than that framing suggests, because reality is already discrete about what it does and doesn't know yet.

### Two columns that resist "just pull whatever's available"

`grid_position` and `starting_tire_compound` are structurally different from `practice_pace`/`quali_gap_to_pole`: even with perfect real-time data access, they don't *exist* until specific external events happen — grid position isn't final until qualifying resolves and any penalties are applied by the stewards (sometimes hours after the session ends), and compound choice isn't announced until race day. No amount of "pull more session data" fixes this; it's not a data-access limitation, it's that the information genuinely doesn't exist yet. `live_predict.py` handles this honestly: qualifying classification is used as a *proxy* for grid position (explicitly documented as pre-penalty), and both columns accept manual overrides for exactly the moment their real values become known from an announcement rather than from FastF1's structured data.

### A real-world surprise: FastF1's location names aren't stable across seasons

Pulling 2025-2026 data to test all this surfaced something the circuit reference table wasn't built to expect: FastF1 renamed the *same physical circuit's* location string between seasons — Monaco appears as `"Monaco"` through 2025 and `"Monte Carlo"` starting 2026; Miami similarly shifts to `"Miami Gardens"`. Since [circuit_reference.py](src/features/circuit_reference.py)'s `resolve()` matches purely on the `location` string, this would have raised a hard error (by design — it's supposed to fail loudly on an unmapped circuit rather than silently produce garbage) for every Monaco/Miami race in the new data, even though nothing about the actual track changed. The fix is a small `LOCATION_ALIASES` dict applied only to the *matching* step, not the data itself — the row keeps its original `location` value (needed later to re-query FastF1 for that exact race), it's only the lookup key that gets normalized. This is a good instance of a general pattern: when two systems both claim to identify "the same real-world thing," don't assume their identifiers agree with each other over time, and build the seam (an alias/normalization layer) rather than hoping it never comes up.

## Phase 1 — Data foundation

### The problem this phase solves

A model can't learn from "Verstappen, Bahrain, 2023" — it needs numeric signals. Phase 1 is entirely about turning raw race history into a table of leakage-safe numbers.

### Ingestion — [src/data/ingest.py](src/data/ingest.py), [src/data/fastf1_client.py](src/data/fastf1_client.py)

Pulls one row per driver per race from FastF1: grid position, qualifying gap-to-pole (plus raw Q1/Q2/Q3 times), FP1-FP3 practice pace, race weather, and the actual outcome (finish position, race pace, pit stops, starting compound). Each race is cached as its own parquet file under `data/raw/races/` — this makes the whole backfill **resumable**: a crash, a rate limit, or a Ctrl-C never loses completed work, because re-running just skips files that already exist.

**A real bug we hit and fixed here:** FastF1 self-enforces a ~500-calls/hour courtesy limit on its free public data sources (it's a client-side sliding-window counter, not a server ban). The first ingestion attempt burned through it in 7 minutes with no retry logic and *silently* skipped 55 of 66 races — `ingest_race()`'s exception handling only caught race-session failures, not qualifying/practice failures, so a rate-limited call just returned `None` and the loop moved on. The fix was in `fastf1_client.load_session()`: catch `RateLimitExceededError` specifically and retry with a backoff instead of giving up.

The backoff interval itself taught a second lesson: the first fix used a flat 60-second sleep per retry attempt. That's far too conservative — the real refill rate is ~1 call per 7.2 seconds (500/hour), so once a race needed many genuinely-fresh calls in a row (e.g. FP2/FP3 never touched before), the 60s cadence meant we only gained about one successful call per minute even though the API would legitimately allow much more. Dropping the backoff to 9 seconds (just above the natural refill interval) fixed it — this is *using the allowance we're actually entitled to*, not evading the limit. The lesson generalizes: when you retry against a rate limit, match your retry cadence to the real refill rate, not an arbitrary "safe-sounding" number.

### Circuit reference table — [src/features/circuit_reference.csv](src/features/circuit_reference.csv), [src/features/circuit_reference.py](src/features/circuit_reference.py)

Some facts about a race aren't computable from that race's data at all — "how hard is it to overtake at Monaco" is a property of the track, stable across years (mostly). So this table is **hand-curated**, not derived: 25 circuits × 10 columns (overtaking difficulty, pit lane loss time, safety car frequency, etc.), sourced from real F1 knowledge as reasoned proxies.

The interesting design choice is the key: `(location, configuration_id, valid_from_year, valid_to_year)`, not just `location`. Why? Tracks get reconfigured — Singapore removed a chicane section for 2023. If you keyed purely by name, a pre-2023 Singapore race and a post-2023 Singapore race would get identical "track characteristics," which is simply false — they're meaningfully different circuits wearing the same name. `resolve()` in `circuit_reference.py` picks the era row whose year range contains the race's season, so this is handled correctly rather than approximated away.

### The feature-layer hierarchy

Built in the order the project plan specifies, each layer merging onto the previous:

```
circuit → driver → team → relative (driver vs teammate) → weather/strategy → final matrix
```

- [driver_features.py](src/features/driver_features.py) — grid position, quali gap, practice pace (all pre-race actuals) plus recency-weighted driver form
- [team_features.py](src/features/team_features.py) — constructor-level form, including form *specific to track type* (a team can be strong on power tracks and weak on street circuits)
- [relative_features.py](src/features/relative_features.py) — teammate comparisons, which isolate driver skill from car performance since teammates share equipment
- [strategy_features.py](src/features/strategy_features.py) — expected pit stops and compound-performance history

### The one idea that matters most: leakage

**If you're predicting a race, you cannot use anything that only becomes known during or after that race.** This sounds obvious until you're elbow-deep in feature engineering and realize "driver's average finish position" is dangerous — average *over which races*? If it includes the race you're predicting, the model is being handed the answer.

The rule in practice:
- Grid position, qualifying time, practice pace → safe to use as-is, because they exist *before* the race is run (that's the whole point of the plan's "rolling prediction" idea — call the model again as each of these becomes real)
- Finish position, actual race pace, pit stop count → only usable as **historical averages from previous races**, never from the race being predicted

This is enforced by one shared function — [src/features/rolling.py](src/features/rolling.py)'s `recent_form()` — not reimplemented per feature. It sorts a group by date and, for each row, averages only over rows *strictly before* it:

```python
for i in range(1, len(vals)):
    prior = vals[:i][::-1]          # everything before row i, most recent first
    weights = decay ** np.arange(len(prior))
    out.loc[idx[i]] = np.average(prior[valid], weights=weights[valid])
```

Row 0 in any group is always `NaN` — there's no prior data for a driver's or team's first-ever race in the dataset. That's not a bug to patch; it's the correct, honest answer, and XGBoost (Phase 2) handles `NaN` natively rather than needing imputation.

**Recency weighting** compounds this: a driver's result from 3 years ago shouldn't count as much as last race. `recent_form` uses exponential decay by *race count* (`decay ** k`, k races back) for general form; `track_form` uses decay by *year gap* for circuit-specific history (last year ≈ 0.7 weight, two years back ≈ 0.49), matching the plan's explicit example.

Because this one function is the single point of failure for the entire leakage guarantee, it's also the most directly tested thing in the repo — [tests/test_features.py](tests/test_features.py) builds tiny synthetic data with a hand-computed expected answer and asserts `recent_form`/`track_form` never touch the current-or-future row. Test the choke point, not every downstream column that depends on it.

### Output

[data/processed/model_matrix.parquet](data/processed/model_matrix.parquet) — 1359 rows (68 races × ~20 drivers, seasons 2022-2024), 33 leakage-safe features + 2 targets (`target_finish_position`, `target_quali_to_race_delta`).

---

## Phase 2 — Predictor v1 (finishing position)

### Why regression, not classification

Finishing position is 1st-20th. You *could* treat this as 20-way classification, but that throws away ordering — predicting 2nd when the truth is 3rd is a tiny error; predicting 2nd when the truth is 19th is a huge one. Classification treats both as "just wrong." **Regression** (predict a continuous number like 3.04) naturally captures "how close," which is both a better training signal and a more honest way to report a prediction (a number plus an implicit error bar reads better than a single categorical guess).

[src/models/train_finish_position.py](src/models/train_finish_position.py) uses `xgb.XGBRegressor(objective="reg:absoluteerror")` — MAE as the training objective, because being off by 2 positions should be twice as bad as being off by 1 (squared error would punish big misses disproportionately, which isn't obviously right for a bounded 1-20 scale).

### Feature prep — [src/models/features.py](src/models/features.py)

Two things happen beyond just selecting the 33 Phase 1 columns:

**One interaction feature**, `grid_x_overtaking_difficulty = grid_position × overtaking_difficulty`. The plan is explicit that XGBoost learns interactions on its own — trees split on one feature at a time but a *sequence* of splits across grid_position and overtaking_difficulty effectively encodes their interaction already. So why add it? Because SHAP (below) explains a model in terms of the columns it's given. If the interaction is implicit, SHAP has to spread its effect awkwardly across two separate features. Giving the model the interaction explicitly as its own column makes the *explanation* cleaner, which matters a lot here since this project's whole second half is about generating those explanations. This is purely an interpretability aid, not a performance requirement.

**A real bug and its fix: fixed-category categoricals.** `starting_tire_compound` is a string (`SOFT`/`MEDIUM`/.../`WET`). XGBoost's native categorical support (`enable_categorical=True`) needs a pandas `category` dtype, and the naive approach is `X[col].astype("category")` — which **infers the category list from whatever data is in front of it**. That works fine on the full training set (all 5 compounds are present), but breaks during a single-row rolling prediction where the compound isn't known yet and gets set to `NaN`: a one-row, all-`NaN` column has *zero* categories, and XGBoost throws `Check failed: n_categories > 0`. It's not a hypothetical edge case — it's exactly the situation Phase 2's own rolling-prediction feature needs to handle. The fix is to fix the category list explicitly and reuse it everywhere:

```python
COMPOUND_CATEGORIES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
X["starting_tire_compound"] = pd.Categorical(X["starting_tire_compound"], categories=COMPOUND_CATEGORIES)
```

This isn't just about avoiding the crash — it matters for *correctness* too. XGBoost encodes categories as integer codes internally, and those codes are baked into the trained tree's split conditions. If training used one inferred category ordering and prediction used a different one (say, alphabetical order changed because a batch happened to be missing a compound), the same integer code would mean a *different compound* at inference time than it meant during training — silently wrong predictions, no error raised. Using one fixed, explicit category list for both training and prediction is what actually guarantees train/predict consistency. (We retrained after this fix, since the previously-saved model's categories weren't guaranteed to match the new fixed list.)

### Time-based cross-validation — the second-most-important idea in this project

Standard k-fold CV shuffles rows randomly into folds. That's wrong here for two independent reasons:

1. **Temporal leakage**: if a 2023 race ends up in your training fold and a 2022 race ends up in your test fold, you're validating a model on the "past" using information from the "future" — completely unrealistic, since in production you'll only ever have past races to train on.
2. **Race leakage**: all ~20 drivers in one race share the same circuit row, weather row, etc. If you randomly split *rows* instead of *races*, some drivers from a race land in train and others from the *same race* land in test — the model gets a sneak preview of that race's exact conditions during training.

`time_based_splits()` in [train_finish_position.py](src/models/train_finish_position.py) fixes both: it de-duplicates to unique `(season, round)` races, sorts by date, and runs `sklearn.model_selection.TimeSeriesSplit` **on the races**, not the rows:

```python
races = df[["season", "round", "race_date"]].drop_duplicates().sort_values("race_date")
for train_races, test_races in TimeSeriesSplit(n_splits=5).split(races):
    # map race-level indices back to every row belonging to those races
```

`TimeSeriesSplit` produces *expanding-window* folds: fold 1 trains on the earliest chunk of races and tests on the next chunk; fold 2 trains on everything up through fold 1's test set and tests on the chunk after that; and so on. Every test set is strictly chronologically after its own training set — the same discipline as Phase 1's leakage rule, just applied at the model-validation level instead of the feature-engineering level. This is genuinely the same idea twice: don't let the model see the future, whether that's "the future" during feature computation or "the future" during evaluation.

### Hyperparameter tuning

`RandomizedSearchCV` samples 40 random combinations from distributions over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda` — the standard XGBoost tree/regularization knobs — and scores each using the *same* time-based folds from above (passed in directly as `cv=folds`, a plain list of `(train_idx, test_idx)` arrays, since sklearn's CV objects accept any iterable of index pairs). We used sklearn's existing `RandomizedSearchCV` rather than adding Optuna as a new dependency: scikit-learn was already required, and a randomized search over 8 hyperparameters with 40 iterations doesn't need Optuna's more sophisticated Bayesian sampling to be effective at this dataset size (1359 rows).

`refit=True` means the final `search.best_estimator_` is retrained on *all* the data (not held back by any fold) using the best-found hyperparameters — that's the model we actually save and use. The reported CV metric, on the other hand, has to come from a *separate* loop (`fold_mae()`) that retrains fresh per fold, because `search.best_score_` reflects the score *during the search*, and mixing "the model trained on everything" with "the score from models trained on subsets" would conflate two different things.

### Reading the results

- **Baseline MAE 3.48**: if you just guessed "you'll finish where you started" (grid position = finish position), you're off by 3.48 positions on average. This is a legitimately strong baseline in F1 — grid position matters a lot — which is exactly why it's the right baseline to beat, not a strawman.
- **Tuned model MAE 3.04**: a genuine ~0.44-position improvement. Modest-sounding, but consistent with how much signal grid position alone already captures; beating a strong baseline by any real margin is meaningful.
- **Calibration table**: bins predictions into quintiles and compares mean predicted vs. mean actual within each bin. If a model is "calibrated," these two columns track closely — which they do here (e.g. one bin: predicted 15.95, actual 17.16). A model that's *accurate on average but miscalibrated* would show the predicted and actual columns diverging systematically at the extremes (e.g. always underpredicting for backmarkers), which would be a red flag this check is specifically designed to catch.

### SHAP — explaining the model, not just scoring it

SHAP (SHapley Additive exPlanations) assigns each feature, for each individual prediction, a numeric contribution — "grid_position pushed this driver's predicted finish 2 places worse than average, driver_recent_form pushed it 1 place better," etc., such that they sum to the actual prediction. `shap.TreeExplainer(model)` computes this efficiently for tree models by walking the actual decision paths rather than needing to re-run the model thousands of times.

The plan poses a specific, falsifiable sanity check: **circuit features should matter more where the plan says they should** — e.g. overtaking difficulty should matter more at Monaco-like tracks than Monza-like tracks, because that's the actual real-world mechanism (grid position is nearly destiny at Monaco; it matters much less at Monza where overtaking is easy). `shap_circuit_check()` in the training script tests this directly: it computes mean `|SHAP value|` for the `grid_x_overtaking_difficulty` interaction feature, split by whether `overtaking_difficulty ≥ 0.7` (hard tracks) or `≤ 0.3` (easy tracks). Result: **0.32 at hard tracks vs. 0.18 at easy tracks** — the model learned the right mechanism on its own, not just a correlation that happens to fit. This is a much stronger validation than a plain accuracy number, because it checks *why* the model is right, not just *that* it's right — a model can hit a good MAE for the wrong reasons, and this check is specifically designed to catch that.

This SHAP output is also the actual bridge to Phase 4: the plan's core idea is that SHAP values become the *query* into the RAG explainer ("grid position mattered a lot because overtaking is hard here — let's retrieve regulation/history text about overtaking difficulty at this circuit"). Phase 2's SHAP check isn't just an evaluation nicety; it's proof that the signal Phase 4 will depend on is real.

### Rolling re-prediction — one model, not three

The plan wants the same race to get re-predicted after FP1-3, after qualifying, and pre-race, using progressively more real information. The naive approach would be training three separate models for three separate feature sets. [src/models/predict.py](src/models/predict.py) does something much simpler, enabled by a property we already needed for the categorical fix above: **XGBoost handles missing values (`NaN`) natively** — a tree's split logic has an explicit "which way does a missing value go" rule learned during training, it isn't just imputed with a placeholder.

That means "the model at the post-practice stage" and "the model at the pre-race stage" can be the *exact same trained model* — the only thing that changes is which columns are filled in vs. `NaN`:

```python
def mask_for_stage(rows, stage):
    if stage == "post_practice":
        for col in POST_QUALI_ONLY_COLS + PRE_RACE_ONLY_COLS:
            rows[col] = np.nan   # grid position, quali gap, starting compound: not real yet
    elif stage == "post_quali":
        for col in PRE_RACE_ONLY_COLS:
            rows[col] = np.nan   # only starting compound still unknown
    # pre_race: everything filled in, no masking
```

[tests/test_rolling_predict.py](tests/test_rolling_predict.py) verifies this actually works on a real row rather than assuming it: predictions on one sample genuinely moved 7.22 → 4.39 → 4.05 across the three stages, proving the model is responding to the new information at each stage rather than ignoring it (a model that returned the same number regardless of what was masked would indicate the masked columns weren't actually load-bearing, which would be a real problem worth catching).

### Why this design choice mattered

Three separate models would have meant: three training runs, three hyperparameter searches, three sets of SHAP explanations that don't necessarily agree with each other on *why* a prediction is what it is, and three times the maintenance burden every time the feature table changes. One model plus native missing-value handling gets the same rolling-prediction *behavior* the plan asks for, with none of that duplication — and it's *more* consistent, not less, because the "why" behind a post-practice prediction and a pre-race prediction for the same race comes from the same learned tree structure, just evaluated with different known/unknown inputs.
