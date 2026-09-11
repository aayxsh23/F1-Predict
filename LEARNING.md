# LEARNING.md

This file teaches *why* the project is built the way it is, phase by phase — not just what happened. Three files, three jobs, don't confuse them:

- [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) — the **design**: what we're building and the phase plan.
- [PROGRESS.md](PROGRESS.md) — the **build log**: terse, factual, "what happened," newest first.
- **This file** — the **tutorial**: concepts, reasoning, code walkthroughs, what a naive/wrong version would have gotten wrong. Read this when you want to understand the material, not just the status.

Agents: update this file per AGENTS.md's "Multi-agent coordination" section — add depth, don't just log events.

---

## Phase 4+5 (merged) — starting a local-only RAG + fine-tuning explainer

### Why skip the plan's "prove it with an API first" step

The original plan staged Phase 4 and 5 deliberately: build the RAG pipeline against an external LLM API first (cheap to iterate, proves retrieval works), *then* swap in a locally fine-tuned model in Phase 5, changing only the generation call. That staging exists to de-risk two hard things at once — but it assumes you're going to want the API-backed version to exist at all. Given the explicit goal here is a fully local, self-hosted explainer with no external API dependency at runtime, building the API version first and then discarding it would be pure throwaway work, not de-risking. So the two phases are being merged: the local model is the generation step from day one, first as the un-fine-tuned base model (still validates retrieval + prompting work end-to-end, same de-risking value the plan wanted from the API step), then swapped for the fine-tuned adapter once Phase 5's training is done. The retrieval logic itself is completely unaffected by this choice either way — it was always going to be "chunks in, LLM call out," and *which* LLM answers that call was always meant to be swappable.

### The `torch` CPU-vs-CUDA gotcha, and why it's easy to miss

`pip install torch` succeeding with no errors makes it very easy to assume you have a working GPU-accelerated PyTorch. On Windows (and Linux), it doesn't — PyPI's default wheel for `torch` is CPU-only; the CUDA-enabled builds live behind PyTorch's own package index (`--index-url https://download.pytorch.org/whl/cu128`, versioned by CUDA release). The failure mode if you don't catch this isn't an error, it's silence: `import torch` works fine, model loading works fine, everything works — just 10-50x slower than it should be, and a 1B-parameter model fine-tune that should take minutes on a GPU would instead take hours on CPU with no indication anything is wrong until you notice the wall-clock time. The fix is cheap once you know to check: `torch.cuda.is_available()` right after install, before writing a single line of model code that depends on it. This is the same category of lesson as the venv incident earlier in this project — a tool reporting success doesn't mean it configured itself the way you assumed.

### Why the RAG corpus leans on real FIA PDFs instead of hand-written summaries for regulations/decisions

For the regulation and steward-decision corpus, real official FIA PDFs were downloaded directly rather than transcribing/paraphrasing their content by hand. This matters for more than authenticity: the actual Phase 4 ingestion pipeline is going to use `pypdf` to parse PDFs into chunks, because that's the realistic shape of "a RAG corpus" for a project like this (regulations and legal-style decisions are published as PDFs, not clean web text) — so sourcing real PDFs now means the ingestion pipeline gets built and tested against the same kind of document it'll actually face, rather than against hand-typed text files that happen to be easier to parse than the real thing. A pipeline that only ever saw pre-cleaned text wouldn't actually prove pypdf-based chunking works.

The circuit-characteristics write-ups in `race_summaries/` are the deliberate exception — there's no equivalent "official PDF" source for track-characteristics narrative, so those are authored directly, but explicitly cross-referenced against the existing hand-curated `circuit_reference.csv` values rather than written independently of them. The reasoning: those numeric proxy values (`overtaking_difficulty`, `safety_car_frequency`, etc.) are exactly what SHAP is going to explain the importance of, so the RAG corpus needs prose that *names and justifies* those same numbers, not a separately-sourced narrative that might describe a circuit in a way that doesn't line up with what the feature table already claims about it.

### One-per-circuit, not one-per-race, for the "race_summaries" folder

The folder name (inherited from `AGENTS.md`'s original repo-layout sketch) suggests per-race recaps, but the actual retrieval need is different: a SHAP-driven query looks like *"why did grid position matter this much at this circuit"* or *"why did overtaking difficulty push the prediction this direction,"* not *"what happened in the 2023 running of this specific race."* A circuit only changes its underlying characteristics on the rare occasion its layout is reconfigured (Singapore 2023, Zandvoort's 2021 return) — writing 105 near-duplicate per-race blurbs would mostly restate the same circuit facts 4-5 times each (once per season) while diluting retrieval relevance, since a query about "why is Monaco hard to pass at" would then have to rank one useful chunk against several redundant ones. 25 circuit-level documents (one per row in `circuit_reference.csv`, Marina Bay's two configuration eras merged into a single document since they discuss the same underlying track) give denser, less redundant grounding for the kind of question this system actually needs to answer.

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

## Phase 3 — Predictor v2 (quali-to-race delta)

### The second target reuses everything except the target column

`target_quali_to_race_delta = grid_position - finish_position` was already sitting in `model_matrix.parquet` since Phase 1 (every target the plan needs was computed up front, once). So Phase 3's actual new work is almost entirely about *evaluating a second target properly*, not building new infrastructure — same feature table, same time-based CV, same tuning search. The one thing worth pausing on: `grid_position` is a **feature** for this target, not just for finish-position. That's not an accident or an oversight to fix — it's the entire reason this target is interesting. A driver starting P1 has almost no room to *gain* (there's nowhere better to go) but plenty of room to *lose*; a driver starting P18 has the opposite shape. The model needs to see grid_position to learn that asymmetric ceiling/floor, exactly the way it needs grid_position to predict absolute finishing position.

### Extracting `training_common.py` — reuse forced by a second consumer

Phase 2's training script had `time_based_splits`, `tune`, `fold_mae`, `calibration_table`, and `shap_circuit_check` all defined inline. Writing Phase 3's script by copy-pasting those five functions and changing one target column would have been the fastest way to get *a* second model working, and the wrong way to do it: two copies of the same CV/tuning logic will inevitably drift — someone fixes a bug or tweaks a hyperparameter range in one file and forgets the other, and now the two "identically evaluated" models secretly aren't. This is the textbook case for extracting a shared module rather than duplicating: not "this might be reused someday" (which would be premature), but "this is being used a second time, right now, by code being written in this same change." `training_common.py` exists because Phase 3 needed it, not in anticipation of a Phase 4 that might.

After the extraction, Phase 2's training script was re-run to confirm it produced byte-for-byte identical metrics to before — a pure refactor should never quietly change behavior, and the only way to be sure is to actually check, not just reason that it "should" be equivalent.

### The baseline MAE being identical between the two targets isn't a coincidence

Phase 2's baseline is "predicted finish = grid" (MAE 3.482); Phase 3's baseline is "predicted delta = 0" (also MAE 3.482, exactly). These aren't independently-arrived-at numbers that happen to match — they're the same computation. `target_quali_to_race_delta = grid - finish`, so `|actual_delta - 0| = |(grid - finish) - 0| = |finish - grid|`, which is exactly what Phase 2's baseline computes. When two "different" baselines come out identical, it's worth checking *why* before treating it as a satisfying coincidence — here the why is simple algebra, and it's a useful sanity check that both targets are being evaluated consistently.

### The delta model tuned to a shallower tree, and its calibration shows it

`RandomizedSearchCV` landed on `max_depth=2` for the delta model, versus depth choices that let Phase 2's finish-position model fit more structure. A depth-2 tree can only carve the feature space into a handful of coarse regions, which shows up directly in the calibration table: the model's predictions cluster more tightly around the mean than the actual deltas do (predicting -2.0 when the true average for that bin is -3.3, predicting 3.8 against an actual 4.8) — a mild "regression to the mean" pattern. This isn't a bug to chase down; it's what a shallow, regularization-favoring tree looks like when tuned on a smaller dataset (1357 rows) where a deeper tree risks overfitting the search itself. It's exactly the kind of thing worth re-checking once the model is retrained on the much larger 2022-2026 dataset — more data typically lets the tuner justify more depth without overfitting.

### The SHAP result that *didn't* replicate — and why that's the interesting finding, not a failure

The expectation, going in, was that the finish-position model's hard-vs-easy-track SHAP pattern for `grid_x_overtaking_difficulty` (0.32 vs 0.18 — circuit difficulty matters more when it's hard to overtake) would show up again for the delta model. It didn't: 0.083 vs 0.090, essentially flat, if anything slightly reversed. The instinct here should not be "the feature engineering must be broken" — it's worth first asking whether the *target* actually implies a different relationship than assumed:

For absolute finishing position, "grid position matters more at hard-to-overtake tracks" is a direct, first-order effect — you largely stay where you started. For the *delta* specifically, a hard-to-overtake track compresses the whole *range* of possible deltas for every driver (nobody moves much, regardless of where they started), so there's simply less delta-variance left for any grid-related interaction feature to explain there — the effect isn't absent, it's that the target itself has less room to move in that regime, so the interaction term's marginal contribution shrinks along with everything else. Meanwhile, the delta model's actual top feature is `grid_vs_expected_position` — "how far did you qualify from what your recent form predicted" — at more than 4x the importance of anything else. That's almost tautologically related to the delta target (qualify worse than your form suggests, and there's more room, and more expectation, to recover), and it makes far more sense as the dominant signal for *this* target than a circuit-interaction feature borrowed from the other model's story.

The lesson: a SHAP sanity check that confirms your prior hypothesis is reassuring, but a SHAP result that *doesn't* match a carried-over assumption is often telling you the two targets have genuinely different underlying mechanisms — worth understanding on its own terms rather than treated as noise or something to force back into agreement.

### More data didn't straightforwardly help — and that's worth sitting with, not explaining away

Once the 2025-2026 backfill finished, both models were retrained on the full dataset (2124 rows, up from 1357). The instinct is "more data = better model" — what actually happened was the *opposite* of that instinct on the metric that matters most: the improvement over the naive baseline shrank for both targets (finish-position: 0.44 → 0.25 positions; delta: 0.54 → 0.37 positions), even though the baseline itself got slightly better (grid position became a *stronger* naive predictor on the fuller dataset, not weaker).

The wrong response here would be to quietly not mention this, or to reach for the first plausible-sounding excuse and call it settled. The right response is to check the obvious suspect first: cold-start rows for the several genuinely new teams that joined the grid in 2026 (Cadillac, Audi) — if those rows have `NaN` history features, they're both hard to predict *and* diluting the average. That turned out **not** to be the answer: by the time of the most recent race (round 13), even a brand-new team already had a dozen races behind it, so `team_recent_form`/`driver_recent_form` null rates for 2026 rows are under 1% — barely different from any other season. Ruling out the tidy explanation is itself useful information, not a dead end.

The leading hypothesis instead is subtler and *can't* be seen by checking for `NaN`s: 2025-2026 had unusually heavy driver-market churn (Antonelli, Hadjar, Colapinto, Doohan, Bortoleto all new to F1 within this window, on top of the two new teams). None of those drivers' form features are *missing* — they're just *thinner*: a rookie with 5 races of recency-weighted history is a noisier signal than a veteran with 60, even though both produce a real number, not a `NaN`. A dataset that's `NaN`-complete can still be quietly harder to learn from if a larger share of it represents genuinely less-established signal. This wasn't independently verified further this session (that would take a per-fold or per-driver-tenure error breakdown) — it's reported as the most plausible open hypothesis, not a proven conclusion, which is the honest place to leave a finding you haven't fully chased down.

The actual takeaway isn't "the retrain failed" — both models still meaningfully beat their baseline, and 2x the data is unambiguously valuable for other reasons (a far more current, accurate live driver lineup, for one, confirmed by re-running the Madrid prediction). It's that "more data" and "better held-out accuracy" are not the same claim, and reporting a metric that moved the *wrong* way, with your best honest guess at why, is worth more than only reporting the numbers that moved the way you expected.
