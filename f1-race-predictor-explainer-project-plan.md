# F1 Race Predictor + Strategy Explainer — Project Plan

## What we're building

A project that combines two things that are usually built separately:

1. **A predictor** — a classic ML model that forecasts race outcomes (finishing position, quali-to-race delta) using historical F1 data.
2. **An explainer** — a RAG (Retrieval-Augmented Generation) system that takes the predictor's output and explains *why* in plain English, grounded in real F1 regulations, historical precedent, and steward decisions — not a hallucinated guess.

The core idea: most portfolio projects do ML *or* GenAI. This does both, connected meaningfully — the model's own explanation (via SHAP) becomes the query that drives the RAG retrieval, so the "why" is generated from real documents, not just from the LLM's general knowledge.

**One-line pitch:** "Predicts what will happen in a race, then explains why — using real regulations and history, not guesses."

---

## Why this project (context)

This came out of a broader analysis of the Indian AI internship market for Summer 2027, which found that:
- Applied AI / GenAI roles are the largest and fastest-growing internship category
- RAG, LLM orchestration, and agentic patterns are heavily represented in real job postings
- Recruiters weight a strong, deployed, defendable GitHub project over tutorials or certificates
- Projects that combine *multiple* high-demand skills (classic ML + GenAI + backend) stand out more than single-skill projects

This project is designed to hit: Python, classic ML (XGBoost + hyperparameter tuning + SHAP), **real deep learning (LoRA fine-tuning of a local 1B LLM)**, embeddings, RAG, **LLM orchestration (LangChain + LangGraph)**, backend (FastAPI), and agentic tool-calling.

**Note on deep learning:** the predictor deliberately uses XGBoost/LightGBM, not a neural net — tabular race-summary data doesn't need DL, and XGBoost is the right tool here, tuned properly via hyperparameter search. The genuine deep learning component now comes from **fine-tuning Llama 3.2 1B locally** (Phase 5) to generate explanations, rather than only calling an external LLM API — this is real neural-network training (via LoRA/QLoRA), not just tabular ML.

---

## Current state (as of this plan)

- **Status:** Concept finalized, no code written yet.
- **Data source identified:** FastF1 Python library (lap times, tire compounds, pit stops, weather, telemetry — detailed from ~2018 onward) + **Jolpica-F1** (free, open-source, Ergast-compatible successor API — the original Ergast API shut down at the end of 2024; Jolpica mirrors its endpoints, so historical race results back to 1950 are still accessible, just via `api.jolpi.ca` instead of the old `ergast.com` URL) + FIA regulation documents + steward decision PDFs (for later explainer grounding) + track layout/revamp history (e.g. circuit changes like Spa's Eau Rouge run-off in 2022, Silverstone/Zandvoort reconfigurations) so historical comparisons at a track account for layout changes rather than treating all years as the same circuit.
- **Scope decided:**
  - Predictor: originally **two** targets sharing one feature pipeline — race finishing position, and quali-to-race delta (how much a driver gains/loses from grid to finish). **Expanded to four on 2026-09-11** (user-requested scope change, see the dedicated note after Phase 3 below) — added a qualifying-result predictor and a race-time predictor.
  - **Rolling predictions** — the model re-runs after each session (FP1/FP2/FP3 → Qualifying → pre-race) rather than predicting once, using progressively more complete features (practice pace → actual grid position → final weather/strategy assumptions). The explainer can then answer "why did the prediction change after qualifying?"
  - **Track-aware features, time-aware (single model, no clustering)** — circuit characteristics are added as ordinary columns to the same feature table used by the predictors below, via a small circuit reference table merged onto every race row. The table is keyed by **circuit + configuration era** (not just circuit name), so a pre-revamp Spa and a post-2022-revamp Spa don't inherit identical characteristics. Still just **4 models total** (see the scope-expansion note after Phase 3); the tree-based models learn interactions like "grid position matters more when overtaking difficulty is high" on their own. No per-circuit or per-cluster models, and no extra components connected to the explainer or agent beyond the same 4 predictors. Full feature list (circuit, driver, team, relative, weather, strategy groups) is detailed in Phase 1.
  - Explainer: RAG over regulations + historical precedent, driven by the predictor's own SHAP feature importances, built with **LangChain/LangGraph** rather than a manual pipeline.
  - **Local fine-tuned LLM for explanations** — instead of relying solely on an external API for generating explanations, **Llama 3.2 1B** is fine-tuned locally via **LoRA/QLoRA** on a small hand-built dataset of (prediction + retrieved context) → (good explanation) pairs. Chosen over reasoning-style distilled models (e.g. DeepSeek-R1-Distill) because the task here is consistent, plain-English explanation writing, not multi-step reasoning traces — Llama 3.2 1B is easier to control and has more direct fine-tuning precedent for this kind of task. RAG still does the fact-grounding/retrieval; fine-tuning teaches the model the explanation *style and task format*, and gives genuine hands-on deep-learning/fine-tuning experience. Runs locally via Hugging Face `transformers`/`peft`, served with `Ollama` or `vLLM`.
  - **Live standings agent pulled into v1** — this is the natural home for LangGraph's tool-calling and multi-step reasoning (e.g. "what does Norris need to win the title this weekend?").
  - **Live upcoming-race prediction, added after Phase 2** — the model-development work in Phases 1-2 validates the pipeline by backtesting on historical races (time-based CV: "would this model, given only pre-race information, have called this already-decided race correctly?"). That's necessary but distinct from actually forecasting a real race that hasn't happened yet, which needs current-season data and a live feature-row assembler rather than replaying historical rows. Added as its own capability (`src/models/live_predict.py`) rather than folded into Phase 2's Output bar, since it depends on ingestion being kept current (not just the original 3-5 season backfill) and on new calendar entries (e.g. Madrid, 2026) having a circuit reference row before they're predictable. Reuses the exact same feature layers and trained model as training — no separate "live" model or duplicated feature logic, and no fixed set of named prediction "stages": whatever session data (FP1-3, qualifying) genuinely exists for the target race at the time gets pulled in, whatever doesn't exist yet stays `NaN`, exactly the way the model's native missing-value handling already works for the historical rolling-re-prediction check in Phase 2.
  - **Free automated hosting** — GitHub Actions (scheduled workflow, timed to known session end times) triggers re-prediction automatically instead of running manually; the app itself is hosted on a free tier (Render/Railway/Fly.io for FastAPI, or Streamlit Community Cloud for a simpler UI). Free tiers spin down when idle, which is fine for a portfolio project.
  - Strategy simulation (pit-stop "what-if" modeling) and cross-prediction reasoning are still deferred to a later phase — not part of v1.

---

## Phase-wise build plan

### Phase 1 — Data foundation (Week 1)

**Pipeline structure:** think of the feature table as a hierarchy, built in layers and merged together — this keeps the pipeline clean and makes it much easier to explain in the README/architecture diagram:

```
Circuit features (per circuit + configuration era)
        ↓
Driver features (per driver, per race)
        ↓
Team/car features (per constructor, per race)
        ↓
Relative features (driver vs. teammate, actual vs. expected)
        ↓
Weather + strategy features (per race weekend)
        ↓
Final model matrix
```

- Pull historical race data via FastF1 / Jolpica-F1 for a meaningful set of seasons (start with recent 3–5 seasons for data quality, extend further back if needed).

- **Circuit reference table, time-aware.** Key it by **circuit + configuration era** (`circuit`, `configuration_id`, `valid_from_year`, `valid_to_year`), not just circuit name — so a pre-2022 Spa and a post-Eau-Rouge-runoff Spa don't inherit identical characteristics. v1 columns:
  - `overtaking_difficulty` (proxy, e.g. historical avg. on-track overtakes per race)
  - `is_street_circuit`
  - `pit_lane_loss_time`
  - `safety_car_frequency`
  - `dnf_rate`
  - `longest_straight_m`
  - `braking_zone_count`
  - `tyre_degradation_level`
  - `rain_race_frequency`
  - `track_length_km`

  Merge onto the main race dataset by `circuit` + race year (resolved to the correct configuration era) — every driver in the same race gets the same circuit-feature values.

  **Ongoing maintenance, not a one-time table:** a new calendar entry (e.g. Madrid joining in 2026) needs a hand-curated row added before any race there can be predicted — the resolver deliberately raises rather than silently guessing at an unmapped circuit. An inaugural circuit's row is necessarily a rougher estimate than one backed by years of real results (public layout facts plus reasonable proxies for the rest), and should be revisited once real race data exists for it. Separately, FastF1's own location naming has drifted between seasons for the same physical circuit (Monaco -> "Monte Carlo", Miami -> "Miami Gardens" starting in 2026) — handled via an alias map rather than duplicate rows, so watch for this with any newly-added season.

- **Driver features** (per driver, per race):
  - `grid_position`, `quali_gap_to_pole`, `practice_pace`
  - `driver_recent_form` (rolling avg. finish/points over recent races)
  - `driver_track_form` (historical performance at this specific circuit)
  - `driver_positions_gained_form` (rolling avg. grid→finish delta)
  - `driver_dnf_rate`

  **Recency-weight** the "form" and "track form" rolling stats rather than treating all past seasons equally (e.g. this year's race at a track weighted ~1.0, last year's ~0.7, two years back ~0.5) — a driver's result from 5 years ago shouldn't count as much as last season's. This matters more than adding extra raw columns.

- **Team/car features** (per constructor, per race): `team_recent_form`, `team_quali_pace`, `team_race_pace`, `team_reliability`, `team_track_type_form` (e.g. constructor's average performance on high-downforce vs. street-circuit vs. low-downforce tracks — captures that a car's competitiveness isn't constant across circuit types).

- **Relative features** (driver vs. teammate — same car, so this isolates driver-specific performance from car performance): `teammate_quali_gap`, `teammate_race_pace_gap`, `grid_vs_expected_position` (how far grid position deviated from what recent form/pace would predict).

- **Weather features**: `air_temp`, `track_temp`, `rain_probability`, `wind_speed`, `wet_track_probability`. These matter especially for the rolling-prediction concept, since weather forecasts firm up between FP1 → Quali → pre-race.

- **Strategy features**: `starting_tire_compound`, `expected_stops`, `tyre_degradation` (from the circuit table), `historical_compound_performance` — gives strategic context without building the full strategy simulator (deferred, see Future phases).

- **Prevent leakage — apply this to every rolling/historical feature above.** For a race being predicted, every "recent form," "track form," or circuit-history stat must be computed using **only races before it** — never using the target race itself or future races. This is the same principle behind the time-based CV splits used in Phase 2; it needs to be enforced at the feature-engineering stage too, not just at model-validation time.

- **Output:** a clean, ~30–45-column tabular dataset ready for modeling — deliberately not 100+ columns; a few well-chosen, leakage-safe, recency-aware features beat a wall of raw columns, and it keeps the eventual SHAP explanations interpretable enough to feed cleanly into the RAG explainer.

### Phase 2 — Predictor v1 (Week 1–2)
- Train **race finishing position** model (XGBoost/LightGBM — tabular data, no need for deep learning here) as a **single model** using the full feature table from Phase 1, including the merged-in circuit characteristics. No clustering or per-track models — the tree-based model learns feature interactions (e.g. "grid position matters more when overtaking difficulty is high") directly from the data.
- **Tune hyperparameters** (tree depth, learning rate, number of estimators, regularization) via `RandomizedSearchCV`/Optuna, using **time-based cross-validation splits** (not random splits — you don't want the model validated on "future" races relative to its training data).
- Evaluate properly (not just accuracy — check calibration, compare against a naive baseline like "grid position = finish position", and check via SHAP that circuit features are actually being used meaningfully, e.g. contributing more at Monaco-like rows than Monza-like rows). As part of this, sanity-check for leakage — confirm no rolling/historical feature for a given race was computed using that race or later ones.
- Optionally add a handful of **interpretable interaction features** (e.g. `grid_position × overtaking_difficulty`, `team_form × circuit_type`) — XGBoost learns interactions on its own, so this isn't required for model performance, but a few manually-created ones can make SHAP output more directly readable when it feeds the RAG explainer later.
- Build the **rolling re-prediction** capability: the same model can be called again with updated inputs after FP1/FP2/FP3 and after Qualifying, producing a fresh prediction each time as more real session data becomes available (practice pace → actual grid position → final weather/strategy assumptions).
- **Output:** a working, evaluated, tuned, track-aware single model for target #1, callable at multiple points across a race weekend.

### Phase 3 — Predictor v2 (Week 2)
- Add **quali-to-race delta** model, reusing the Phase 1/2 pipeline — including the merged circuit features and rolling re-prediction, since both targets share the same feature base. At the time, still just 2 models — see the scope-expansion note directly below for the two that were added afterward.
- Add SHAP explainability to both models — this is what will drive the RAG layer later, and will also explain *why* a prediction shifted between sessions (e.g. "grid position became the dominant feature after qualifying") or across circuits (e.g. "overtaking difficulty pushed grid position's importance up at this track").
- **Output:** two working, track-aware, session-updatable predictors, both with SHAP-based feature importance output.

### Scope expansion (2026-09-11) — qualifying and race-time predictors added

User-requested, explicit scope change from the plan's original "exactly 2 predictor models" — see AGENTS.md's hard scope constraints for the canonical, current list of all 4. Both new models reuse the Phase 1-3 machinery (`training_common.py`'s time-based CV/tuning/SHAP check) rather than inventing new evaluation methodology:

- **`qualifying`** — predicts qualifying gap-to-pole *before qualifying happens*, from practice pace and historical form only. This is a genuinely different information stage from the other three models (which all predict something that resolves at the race, so "everything known pre-race" is fair game) — it needed its own restricted, leakage-safe feature subset (`QUALI_SAFE_FEATURE_COLS` in `src/models/features.py`), excluding grid_position, quali_gap_to_pole, teammate_quali_gap, grid_vs_expected_position, starting_tire_compound, historical_compound_performance, and all weather columns (ingest.py captures weather from the *race* session, not qualifying — Sunday's weather isn't known on Saturday, and no separate qualifying-weather column exists). Top SHAP features: practice_pace, team_quali_pace, driver_recent_form — exactly the sensible pre-quali story.
- **`race_time`** — predicts each driver's race time as a gap to the winner in seconds (0 for the winner), from FastF1's `results['Time']` column (winner's absolute duration; everyone else's gap to that winner already, no extra computation needed). DNF/not-classified rows have no defined time and are dropped, not imputed. Uses the same full pre-race feature set as finish_position/quali_delta, since it resolves at the race like they do.
- Both required a raw-data addition (`gap_to_winner_seconds` in `src/data/ingest.py`) and a full re-ingestion of all 105 cached races to backfill it — see PROGRESS.md for a real gotcha hit during that backfill (the naive `--force` re-ingest tried to re-fetch every session from scratch rather than just the new column, and got stuck rate-limited chasing future not-yet-run races).

### Phase 4 — RAG explainer with LangChain (Week 3–4)
- Collect and chunk the grounding corpus: FIA regulations, steward decision documents, historical race summaries.
- Build the retrieval pipeline using **LangChain**: embed the corpus, store in a vector DB (start simple — Chroma), retrieve relevant chunks based on the SHAP output as the query.
- Use LangChain's chains to connect retrieval output + predictor output to an **external LLM API** (OpenAI/Anthropic/Gemini) and generate a plain-English explanation — this proves the RAG pipeline works before adding the complexity of a local fine-tuned model.
- **Output:** given a prediction, the system explains it in natural language, citing real regulation/history text — built on a framework, not manual glue code.

### Phase 5 — Local LLM fine-tuning with Llama 3.2 1B (Week 4–5)
- Build a **small hand-curated dataset** (~100–300 examples) of (prediction + retrieved regulation/history context) → (good explanation) pairs — draft with a larger API model (GPT-4/Claude) first, then review and clean by hand for quality and consistent style.
- Fine-tune **Llama 3.2 1B** locally via **LoRA/QLoRA** using Hugging Face `transformers` + `peft` — feasible on a single consumer GPU (even ~8GB VRAM) given the model's small size.
- Swap the fine-tuned local model in as the generation step in the Phase 4 pipeline (retrieval logic stays identical — only the final "write the explanation" call changes), and serve it locally via `Ollama` or `vLLM`.
- Compare outputs qualitatively (and ideally with a small held-out eval set) against the Phase 4 API-based version to confirm the fine-tuned model is actually competitive in explanation quality.
- **Output:** a locally fine-tuned, self-hosted LLM generating explanations — genuine hands-on deep-learning/fine-tuning experience, not just API calls.

### Phase 4+5, as actually built (2026-09-11) — done, deviated from the plan text above by explicit user choice

Both phases were merged and built local-only from day one rather than staged through an external-API version first — see PROGRESS.md/LEARNING.md for the full reasoning; the short version is that building a throwaway API-backed version first only de-risks the retrieval logic, and the base local model does that same de-risking job just as well. Concrete differences from the plan text above:

- **No external LLM API at any point** — embeddings (`sentence-transformers`), retrieval (Chroma), and generation (`meta-llama/Llama-3.2-1B-Instruct`, 4-bit) are all local from the first working version.
- **The fine-tuning dataset (108 examples, not 100-300 hand-curated) was agent-drafted, not GPT-4/Claude-API-drafted-then-hand-cleaned** — one example per (circuit, target) combination, built from real predictions/SHAP/retrieved context run through the actual inference pipeline, with template-composed completions authored to have the exact grounding discipline the base model's own test run proved it lacked (repetition, one internally-contradictory claim). Lives in `src/rag/training_data/explanations.jsonl`.
- **Served via plain `transformers`+`peft` in-process, not Ollama/vLLM** — an always-on local server is a Phase 7 deployment decision, not needed for building/testing the pipeline itself.
- **Result, qualitatively verified**: same Monaco prediction, base model vs. fine-tuned adapter — base model produced a rambling explanation that at one point contradicted its own premise (claimed a "strong chance of overhauling other drivers" immediately after correctly stating Monaco is the hardest circuit on the calendar to overtake at, for a driver predicted to finish 15th). The fine-tuned adapter produced a concise, internally-consistent explanation with correct SHAP-direction language, and generalized the learned style to a genuinely held-out row/target combination rather than memorizing training examples verbatim.
- **Output achieved:** yes — grounded, local, natural-language explanations of all 4 predictors' outputs, with a real (not just claimed) qualitative improvement from the fine-tune.

### Phase 6 — Live standings agent with LangGraph (Week 5–6)
- Build a **LangGraph** agent with a tool for fetching live/current standings data (Jolpica-F1/F1 API).
- Give it multi-step reasoning ability: e.g. "what does Norris need to win the title this weekend?" requires pulling current points, computing remaining-race scenarios, and reasoning about outcomes — not a single lookup.
- Add basic state/memory so it can handle follow-up questions in the same conversation.
- **Output:** an agent that answers dynamic, scenario-based questions using live data and tool calls — the clearest demonstration of orchestration-framework skills in the project.

### Phase 7 — Backend + automation + polish (Week 6–7)
- Wrap everything in a FastAPI backend with clear endpoints (predict, explain, ask-agent).
- Automate the rolling re-prediction with a **GitHub Actions scheduled workflow**, timed to known session end times for a race weekend, instead of running the script manually — it calls the prediction pipeline and writes the latest result to a small DB or JSON file the app reads from.
- Deploy the app on a **free hosting tier** — Render, Railway, or Fly.io for the FastAPI backend, or Streamlit Community Cloud if a simpler built-in UI is preferred over a custom frontend. Free tiers spin down when idle and take a few seconds to wake up on first request — acceptable for a portfolio demo, not something to worry about fixing. **Note:** the locally fine-tuned model needs to run somewhere with GPU access or be served via a small inference endpoint — free-tier web hosts typically won't have GPU, so plan to either run inference on your own machine and expose it, or fall back to the Phase 4 API model for the hosted demo while showcasing the fine-tuned model separately (e.g. in a notebook/video) in your README.
- Build a simple UI or at least a clean demo script/notebook. **Detailed forward-looking design** (custom React frontend + FastAPI backend, API contract, design tokens, GitHub-Actions-as-datastore hosting pattern, phased build order) already drafted in [phase7-ui-backend-plan.md](phase7-ui-backend-plan.md) — written once Phase 3 was done, before Phase 4-6 started, so the design work is ready when this phase actually begins rather than starting from scratch.
- Write a strong README: problem, architecture in plain words, demo GIF, what you'd improve next.
- **Output:** a deployed, self-updating, demoable, documented v1 project — no manual re-runs required.

---

## Future phases (stretch goals, not required for v1)

| Stretch feature | What it adds | Why it's deferred |
|---|---|---|
| **Strategy simulation** | "What if he pits lap 20 vs lap 30?" — a different kind of model (simulation, not classification/regression) | Genuinely different problem type from the two predictors; not a quick add-on |
| **Cross-prediction reasoning** | Explainer compares two predictions at once (e.g. why race position improves but quali doesn't) | Needs multi-input reasoning, harder than single-output explanation — natural v2 feature |
| **Penalty likelihood model** | Predicts penalty risk from incident type | Needs a labeled steward-decision dataset, which is harder to build than lap-time data |
| **Dedicated telemetry deep learning model** | A neural net over lap-by-lap telemetry (sequence data — e.g. tire degradation or pace-drop prediction) | Telemetry is genuinely sequence-shaped and suits DL better than race-summary tables do, but it's a separate modeling effort from the LLM fine-tuning already in v1 — only worth it if you want DL on structured time-series data specifically |

## Future feature extensions (beyond the v1 ~30–45 column set)

The v1 feature set in Phase 1 is deliberately curated, not exhaustive. If the model needs more signal later, or as a "what I'd add next" talking point for interviews, these are the next features worth adding — grouped the same way as Phase 1, roughly in order of expected value:

| Feature group | Additional features | Why deferred from v1 |
|---|---|---|
| Circuit | `num_corners`, `high_speed_corner_ratio`, `elevation_change_m`, `avg_lap_distance_km`, `race_laps`, `avg_safety_car_laps` (disruption magnitude, not just SC occurrence), `red_flag_frequency`, `pit_stop_frequency`, `track_evolution_proxy` (how much lap time improves through the weekend) | Diminishing returns beyond the core 10 circuit columns; XGBoost already gets strong signal from overtaking difficulty, degradation, and street-circuit flag |
| Driver | Split `driver_recent_form` into explicit windows (`driver_avg_finish_last_3/5`, `driver_points_last_5`, `driver_quali_avg_last_5`, `driver_race_pace_avg_last_5`), plus `q1_time_gap`/`q2_time_gap`/`q3_time_gap` instead of a single quali-gap-to-pole | More granular than v1 needs; useful once you're tuning for marginal accuracy gains rather than proving the concept |
| Team | `constructor_avg_finish_last_3/5`, `constructor_points_rate`, `team_dnf_rate` broken out separately from driver DNF rate | Adds precision but overlaps significantly with `team_reliability`/`team_recent_form` already in v1 |
| Car development | `upgrade_introduced_this_race` (flag) and/or `races_since_last_upgrade`, sourced from a hand-curated upgrade calendar (which team brought what, at which race — paddock-journalism sourced, e.g. RaceFans/Motorsport.com upgrade trackers, not available via FastF1/Jolpica) | v1's `team_recent_form`/`team_race_pace` are recency-weighted rolling averages, so they *do* eventually reflect an upgrade's effect — but only as a lagging indicator over several subsequent races, since the race an upgrade is introduced at is predicted using pre-upgrade form (correctly, per the leakage rule — the model has no visibility into a change starting exactly at the race being predicted). A step-change upgrade is currently a real, undetected blind spot for its first few races; fixing it needs an explicit upgrade-tracking dataset, which is a genuinely separate data-curation effort on the scale of the circuit reference table, not a quick column add |
| Historical track performance | Full rolling stat set per driver-at-circuit (`driver_avg_grid_at_track`, `driver_avg_positions_gained_at_track`, `driver_points_at_track`, `driver_podium_rate_at_track`, `driver_avg_pace_at_track`), all recency-weighted | v1's single recency-weighted `driver_track_form` captures most of this value with far less feature-engineering overhead |
| Weather | `humidity`, `wind_direction`, `rain_intensity`, derived `temperature_vs_historical_avg`, `crosswind_strength` | Marginal beyond the 5 v1 weather columns; worth adding if weather-sensitivity turns out to matter a lot in early evaluation |
| Strategy | `one_stop_probability`/`two_stop_probability` as explicit predicted outputs, `compound_laps`, `tyre_age_at_predicted_finish` | Edges toward the deferred strategy-simulation feature above; fine as tabular columns, but easy to over-invest here before the core predictor is even validated |
| Reliability/incidents | `mechanical_failure_rate` split from general DNF rate, `penalty_count_recent`, `incident_rate_recent` | Keep penalties/incidents separate from the main predictor if the explainer later uses steward documents directly — avoids the model and the RAG layer working from overlapping but inconsistent penalty signals |

---

## Summary: what "done" looks like for v1

A live, deployed, self-updating system where you can:
1. Pick an upcoming or historical race
2. Get four predictions (qualifying gap-to-pole, finishing position, quali-to-race delta, race time gap-to-winner) with model confidence, built on a ~30–45 feature table spanning circuit (time-aware, per configuration era), driver, team, relative, weather, and strategy signals — aware of the specific circuit's overtaking characteristics (e.g. Monaco vs. Monza behave differently), from **properly tuned** models. The qualifying prediction is available even before qualifying happens; the other three sharpen as more real session data arrives.
3. See the prediction automatically update after each practice/qualifying session, without running anything manually
4. Get a plain-English explanation of *why*, and *why it changed*, grounded in real FIA regulations and historical precedent — generated entirely by your **own locally fine-tuned Llama 3.2 1B**, no external API call in the loop
5. Ask dynamic, scenario-based questions ("what does X need to do to win the title?") and get an agentic answer using live data

That (Phases 1–7) is a complete, strong, defendable project that hits classic ML with hyperparameter tuning, track-aware feature engineering, RAG, **real LLM fine-tuning (genuine deep learning)**, LangChain/LangGraph orchestration, agentic tool-calling, and free automated deployment — all built on just **4 predictors + 1 fine-tuned explainer LLM + 1 agent**, not a sprawl of extra models. Everything in "Future phases" is optional upside, not a requirement to call this finished.