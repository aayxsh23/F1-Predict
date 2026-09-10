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

This project is designed to hit: Python, classic ML (XGBoost + SHAP), embeddings (the one place deep learning shows up, via the vector DB's embedding model), RAG, **LLM orchestration (LangChain + LangGraph)**, backend (FastAPI), and agentic tool-calling — the last two now pulled into v1 scope rather than deferred.

**Note on deep learning:** the predictor deliberately uses XGBoost/LightGBM, not a neural net — tabular race-summary data doesn't need DL, and XGBoost is the right tool here. The only DL you get "for free" is via embeddings in the RAG pipeline. If deep learning specifically matters for your resume, see the optional stretch model in the Future Phases table below — it's a genuinely separate piece of work, not something to force into the core pipeline.

---

## Current state (as of this plan)

- **Status:** Concept finalized, no code written yet.
- **Data source identified:** FastF1 Python library (lap times, tire compounds, pit stops, weather, telemetry — detailed from ~2018 onward) + **Jolpica-F1** (free, open-source, Ergast-compatible successor API — the original Ergast API shut down at the end of 2024; Jolpica mirrors its endpoints, so historical race results back to 1950 are still accessible, just via `api.jolpi.ca` instead of the old `ergast.com` URL) + FIA regulation documents + steward decision PDFs (for later explainer grounding) + track layout/revamp history (e.g. circuit changes like Spa's Eau Rouge run-off in 2022, Silverstone/Zandvoort reconfigurations) so historical comparisons at a track account for layout changes rather than treating all years as the same circuit.
- **Scope decided:**
  - Predictor: **two** targets sharing one feature pipeline — race finishing position, and quali-to-race delta (how much a driver gains/loses from grid to finish).
  - **Rolling predictions** — the model re-runs after each session (FP1/FP2/FP3 → Qualifying → pre-race) rather than predicting once, using progressively more complete features (practice pace → actual grid position → final weather/strategy assumptions). The explainer can then answer "why did the prediction change after qualifying?"
  - **Track-aware features (single model, no clustering)** — circuit characteristics (overtaking difficulty, street vs. permanent flag, historical average positions gained/lost, safety car frequency, pit-lane loss time) are added as ordinary columns to the same feature table used by the two predictors below — a one-time circuit reference table (~24 rows, one per track) merged onto every race row via the circuit name. Still just **2 models total**; the tree-based model learns interactions like "grid position matters more when overtaking difficulty is high" on its own. No per-circuit or per-cluster models, and no extra components connected to the explainer or agent beyond the same 2 predictors.
  - Explainer: RAG over regulations + historical precedent, driven by the predictor's own SHAP feature importances, built with **LangChain/LangGraph** rather than a manual pipeline.
  - **Live standings agent pulled into v1** — this is the natural home for LangGraph's tool-calling and multi-step reasoning (e.g. "what does Norris need to win the title this weekend?").
  - **Free automated hosting** — GitHub Actions (scheduled workflow, timed to known session end times) triggers re-prediction automatically instead of running manually; the app itself is hosted on a free tier (Render/Railway/Fly.io for FastAPI, or Streamlit Community Cloud for a simpler UI). Free tiers spin down when idle, which is fine for a portfolio project.
  - Strategy simulation (pit-stop "what-if" modeling) and cross-prediction reasoning are still deferred to a later phase — not part of v1.

---

## Phase-wise build plan

### Phase 1 — Data foundation (Week 1)
- Pull historical race data via FastF1 / Jolpica-F1 for a meaningful set of seasons (start with recent 3–5 seasons for data quality, extend further back if needed).
- Build a clean, reusable feature set: grid position, practice/quali pace, historical track performance per driver, weather, tire strategy, team/car form.
- Build a small **circuit reference table** (~24 rows, one per track) with track-characteristic columns: overtaking-difficulty proxy (e.g. historical avg. on-track overtakes per race), `is_street_circuit` flag, historical average |grid − finish| position change, safety car frequency, pit-lane loss time. Merge this onto the main race dataset by circuit name — every driver in the same race gets the same circuit-feature values, only their own grid position/pace/etc. differ.
- **Output:** a clean tabular dataset ready for modeling, with track characteristics included as ordinary columns.

### Phase 2 — Predictor v1 (Week 1–2)
- Train **race finishing position** model (XGBoost/LightGBM — tabular data, no need for deep learning here) as a **single model** using the full feature table from Phase 1, including the merged-in circuit characteristics. No clustering or per-track models — the tree-based model learns feature interactions (e.g. "grid position matters more when overtaking difficulty is high") directly from the data.
- Evaluate properly (not just accuracy — check calibration, compare against a naive baseline like "grid position = finish position", and check via SHAP that circuit features are actually being used meaningfully, e.g. contributing more at Monaco-like rows than Monza-like rows).
- Build the **rolling re-prediction** capability: the same model can be called again with updated inputs after FP1/FP2/FP3 and after Qualifying, producing a fresh prediction each time as more real session data becomes available (practice pace → actual grid position → final weather/strategy assumptions).
- **Output:** a working, evaluated, track-aware single model for target #1, callable at multiple points across a race weekend.

### Phase 3 — Predictor v2 (Week 2)
- Add **quali-to-race delta** model, reusing the Phase 1/2 pipeline — including the merged circuit features and rolling re-prediction, since both targets share the same feature base. Still just the same 2 models, no additional ones.
- Add SHAP explainability to both models — this is what will drive the RAG layer later, and will also explain *why* a prediction shifted between sessions (e.g. "grid position became the dominant feature after qualifying") or across circuits (e.g. "overtaking difficulty pushed grid position's importance up at this track").
- **Output:** two working, track-aware, session-updatable predictors, both with SHAP-based feature importance output.

### Phase 4 — RAG explainer with LangChain (Week 3–4)
- Collect and chunk the grounding corpus: FIA regulations, steward decision documents, historical race summaries.
- Build the retrieval pipeline using **LangChain**: embed the corpus, store in a vector DB (start simple — Chroma), retrieve relevant chunks based on the SHAP output as the query.
- Use LangChain's chains to connect retrieval output + predictor output to an LLM and generate a plain-English explanation.
- **Output:** given a prediction, the system explains it in natural language, citing real regulation/history text — built on a framework, not manual glue code.

### Phase 5 — Live standings agent with LangGraph (Week 4–5)
- Build a **LangGraph** agent with a tool for fetching live/current standings data (Jolpica-F1/F1 API).
- Give it multi-step reasoning ability: e.g. "what does Norris need to win the title this weekend?" requires pulling current points, computing remaining-race scenarios, and reasoning about outcomes — not a single lookup.
- Add basic state/memory so it can handle follow-up questions in the same conversation.
- **Output:** an agent that answers dynamic, scenario-based questions using live data and tool calls — the clearest demonstration of orchestration-framework skills in the project.

### Phase 6 — Backend + automation + polish (Week 5–6)
- Wrap everything in a FastAPI backend with clear endpoints (predict, explain, ask-agent).
- Automate the rolling re-prediction with a **GitHub Actions scheduled workflow**, timed to known session end times for a race weekend, instead of running the script manually — it calls the prediction pipeline and writes the latest result to a small DB or JSON file the app reads from.
- Deploy the app on a **free hosting tier** — Render, Railway, or Fly.io for the FastAPI backend, or Streamlit Community Cloud if a simpler built-in UI is preferred over a custom frontend. Free tiers spin down when idle and take a few seconds to wake up on first request — acceptable for a portfolio demo, not something to worry about fixing.
- Build a simple UI or at least a clean demo script/notebook.
- Write a strong README: problem, architecture in plain words, demo GIF, what you'd improve next.
- **Output:** a deployed, self-updating, demoable, documented v1 project — no manual re-runs required.

---

## Future phases (stretch goals, not required for v1)

| Stretch feature | What it adds | Why it's deferred |
|---|---|---|
| **Strategy simulation** | "What if he pits lap 20 vs lap 30?" — a different kind of model (simulation, not classification/regression) | Genuinely different problem type from the two predictors; not a quick add-on |
| **Cross-prediction reasoning** | Explainer compares two predictions at once (e.g. why race position improves but quali doesn't) | Needs multi-input reasoning, harder than single-output explanation — natural v2 feature |
| **Penalty likelihood model** | Predicts penalty risk from incident type | Needs a labeled steward-decision dataset, which is harder to build than lap-time data |
| **Dedicated deep learning model** | A neural net over lap-by-lap telemetry (sequence data — e.g. tire degradation or pace-drop prediction) | Telemetry is genuinely sequence-shaped and suits DL better than race-summary tables do, but it's a separate modeling effort, not a quick add-on. Only worth it if DL specifically is a resume priority. |

---

## Summary: what "done" looks like for v1

A live, deployed, self-updating system where you can:
1. Pick an upcoming or historical race
2. Get two predictions (finishing position, quali-to-race delta) with model confidence, aware of the specific circuit's overtaking characteristics (e.g. Monaco vs. Monza behave differently)
3. See the prediction automatically update after each practice/qualifying session, without running anything manually
4. Get a plain-English explanation of *why*, and *why it changed*, grounded in real FIA regulations and historical precedent — not a generic LLM guess
5. Ask dynamic, scenario-based questions ("what does X need to do to win the title?") and get an agentic answer using live data

That (Phases 1–6) is a complete, strong, defendable project that hits classic ML, track-aware feature engineering, RAG, LangChain/LangGraph orchestration, agentic tool-calling, and free automated deployment — all built on just **2 predictors + 1 explainer + 1 agent**, not a sprawl of per-circuit models. Everything in "Future phases" is optional upside, not a requirement to call this finished.