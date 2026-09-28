# AGENTS.md

Canonical instructions for any coding agent working in this repo (Claude, Gemini, or otherwise). [CLAUDE.md](CLAUDE.md) and [GEMINI.md](GEMINI.md) are thin, tool-specific pointers to this file — **this is the source of truth**, not them. If you find a rule duplicated and diverging between files, fix it here first and re-sync the others.

> **Always keep [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) in context.** It's the design history and phase order. Re-read it before planning any non-trivial change. Where it and this file disagree, this file wins (it records the later scope decisions).

Start with [README.md](README.md) for what the app does and how to run it, and [docs/MODELS.md](docs/MODELS.md) for what each model predicts and how accurate it is.

## What this is

An F1 race-weekend forecaster: four XGBoost predictors (qualifying, finishing position, places gained, gap to the winner), odds and ranges sampled from them, a tyre-strategy simulator fitted on real stint data, championship title odds, and a Gemini-backed analyst chat that answers only from the app's own data and cites the FIA rules it uses.

## Hard scope constraints (do not violate without the user explicitly changing scope)

- **Exactly 4 predictor models**, all XGBoost, all built on the same circuit/driver/team feature layers. No per-circuit or per-cluster models; circuit characteristics are ordinary columns. The 4 targets:
  1. `finish_position`: race finishing position.
  2. `quali_delta`: places gained from the grid (grid minus finish).
  3. `qualifying`: gap to pole as a % of the pole lap, predicted *before qualifying happens*. It must never see a column only known once qualifying or the race has happened: `QUALI_SAFE_FEATURE_COLS` in `src/models/features.py` is the enforced list (no grid, quali times, race-day tyres or weather). Getting this wrong is a leakage bug, not a style choice.
  4. `race_time`: gap to the winner as a % of the winner's race time, lapped cars included.
- **Everything else is derived, not a fifth model:** odds and ranges are sampled from the finish and qualifying models (`src/models/probabilities.py`); lap times and race length are historical ratios (`src/models/estimates.py`); title odds simulate the season with the race sampler (`src/agent/scenarios.py`); the strategy model is a fitted tyre-wear model plus a simulator (`src/strategy/model.py`). Keep new capabilities in that shape.
- **One model serves every weekend stage.** Training stacks one copy of each row per stage with not-yet-known columns blanked (`features.mask_for_stage`). Don't add per-stage models, and don't train on columns a live forecast never has.
- **Evaluate honestly.** Hyperparameters are tuned on the earliest 60% of races; reported accuracy is walk-forward on later races (each predicted by a model trained only on earlier ones). Never report or display in-sample numbers as accuracy.
- **No deep learning in the predictors**: XGBoost only. The fine-tuned local Llama explainer (`src/rag/llm.py`, `finetune.py`) stays as an optional offline mode.
- **Scope changes, 2026-09-28 (user: "start all"):** the strategy simulator, previously deferred, is in scope and built. The chat uses the **Gemini API** (the user's choice; `GEMINI_API_KEY`, model `GEMINI_MODEL`, default `gemini-3.8-flash`), replacing the local-only rule for the chat and the written explanations. Hosting is **Vercel** (frontend and API in one project).
- The chat agent must be built on **LangGraph** (via `langchain.agents.create_agent`) with LangChain tools; every number it states must come from a tool.
- **Serving stays light:** the API must not import pandas, XGBoost, torch, FastF1 or a PDF parser (Vercel's function size limit). Anything needing them is precomputed by the scheduled jobs and shipped as JSON. `requirements.txt` is the serving set; keep it that way.
- Automation via **GitHub Actions** only (`refresh-predictions.yml`, `retrain.yml`, `ci.yml`), no paid scheduler.

## Repo layout

```
api/index.py              # Vercel entry point for the FastAPI app
data/
  raw/races/              # one parquet per race (committed: CI has no re-ingest budget)
  raw/laps/               # lap-by-lap stint data per race (committed; strategy model input)
  raw/fastf1_cache/       # FastF1's own cache, gitignored
  processed/              # feature matrix + walk-forward predictions, gitignored (rebuild)
  predictions/            # what the API serves: forecasts, timeline/, schedule/, backtest/
  corpus/                 # FIA regulations, steward decisions, circuit write-ups + index.json
docs/MODELS.md            # model cards: targets, inputs, held-out accuracy, limits
src/
  data/                   # FastF1 ingestion, Open-Meteo forecasts
  features/               # feature layers + circuit reference table
  models/                 # train.py, predict, probabilities, estimates, refresh_job, backtest_export, catalog
  strategy/               # tyre-wear fit + strategy simulator (params.json committed)
  agent/                  # chat (LangGraph + Gemini), tools, championship maths, Jolpica client
  rag/                    # corpus index + BM25 search; local Llama explainer (optional, offline)
  api/                    # FastAPI routes, read-time enrichment, JSON cache
frontend/                 # React app (see frontend/DESIGN.md, PRODUCT.md)
tests/
.github/workflows/        # refresh (live forecast), retrain (weekly, gated), ci
```

## Python environment

- **Never install packages into the global/system Python.** Use the venv at `.venv/` (gitignored), invoking `.venv/Scripts/python.exe` / `.venv/Scripts/pip.exe` directly by path on Windows (tool calls don't keep an activated shell). Create it once with `python -m venv .venv` if missing.
- Requirements are split and pinned: `requirements.txt` (the API, what Vercel installs), `requirements-dev.txt` (pipeline, training, tests; includes the first), `requirements-llm.txt` (optional local Llama, needs a CUDA GPU). Local development normally wants `pip install -r requirements-dev.txt`.
- This rule exists because it was violated once (requirements installed into the global interpreter). See PROGRESS.md/LEARNING.md for that incident.

## Conventions

- The circuit reference table (`src/features/circuit_reference.csv`) is keyed by circuit + configuration era. A new venue needs a hand-curated row before it can be predicted (the resolver raises rather than guessing). Sepang was added this way for the 2026 Bahrain GP.
- The SHAP breakdown drives retrieval for written explanations (the top inputs plus the circuit form the search query); treat predictor and explainer as connected.
- Fan-facing names for model inputs live in one place: `src/models/catalog.py` `FEATURE_LABELS`. The app never shows a column name.

## Multi-agent coordination

More than one agent (Claude via [CLAUDE.md](CLAUDE.md), Gemini via [GEMINI.md](GEMINI.md), possibly others reading this file directly) may work on this repo, in the same or different sessions. To keep them from drifting out of sync or contradicting each other:

- **This file is canonical.** Shared rules (scope constraints, repo layout, conventions) live here once. CLAUDE.md/GEMINI.md hold only tool-specific notes (e.g. which slash commands or CLI flags that tool uses) — never a second copy of a shared rule.
- **Changing scope or a hard constraint?** Edit it here first, in the same change also check whether CLAUDE.md/GEMINI.md need their short pointer text updated (usually they don't — they just point here).
- **Update [PROGRESS.md](PROGRESS.md) after every meaningful change** — new phase started/finished, a scope decision, a file layout change, a blocker hit. This is the handoff mechanism between agents and sessions: whichever agent picks up next reads PROGRESS.md first to know what state the repo is actually in, rather than re-deriving it or trusting stale memory. Keep entries short (a few lines), newest first.
- **Update [LEARNING.md](LEARNING.md) after every meaningful change too, but write it differently.** PROGRESS.md is a terse build log (what happened); LEARNING.md is a teaching document (why, and how the concepts work) — the user is using this project to learn, not just to ship it. When a phase advances or a real decision/bugfix lands, add or extend that phase's section in LEARNING.md explaining the concept in depth: what problem it solves, why this approach over the obvious alternative, how the code implements it, and what a wrong/naive version would have gotten wrong. Write for someone learning the material, not for a changelog reader.
- Don't assume the other agent's last session is fully reflected in the project plan file — the plan is the *design*, PROGRESS.md is the *build log*, LEARNING.md is the *tutorial*. Check all three.

## Repo hygiene

Before ending any turn that created, edited, or deleted files, leave the repo clean:

- Run `git status` and account for every entry — nothing untracked or modified that shouldn't be there (stray scratch files, editor/OS cruft, leftover debug output).
- Any new generated or local-only path (build output, caches, `__pycache__/`, notebook checkpoints, local env files, model artifacts, a new data subfolder under `data/raw` or `data/processed`, a new vector-store directory, etc.) gets added to `.gitignore` in the same change that introduces it — don't leave it to `git status` noise for the next session to puzzle over.
- Don't commit anything unless the user explicitly asks (see the repo-wide convention on this) — "clean" means `git status` is legible and intentional, not that changes are auto-committed.
- If a file you scaffolded turns out unused (e.g. an empty stub nothing ever filled in and nothing depends on it), delete it rather than leaving dead placeholders around.
