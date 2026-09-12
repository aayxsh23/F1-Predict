# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + React + TypeScript (static SPA, no server-rendering need — all data comes from a separate FastAPI backend). react-router-dom for routing. TanStack Query as the data layer (no Redux/Zustand — the query cache is the state). Tailwind CSS + shadcn/ui for primitives. Recharts for the SHAP/backtest charts. Decided in [phase7-ui-backend-plan.md](phase7-ui-backend-plan.md) with each choice justified against a simpler alternative — not delegated, not re-opened here.

## Users

F1 fans who want to know what's likely to happen in an upcoming race weekend, and why — not F1 insiders needing broadcast-grade telemetry, and not a technical audience needing the ML pipeline explained to them. They arrive already knowing the sport (driver names, team names, what qualifying/grid/practice mean) but not knowing anything about SHAP, XGBoost, or RAG. The product is a real prediction tool first; it never frames itself as a portfolio demo or explains its own ML internals as a feature.

## Product Purpose

Predicts F1 race weekend outcomes (qualifying gap-to-pole, finishing position, grid-to-finish delta, race-time gap-to-winner) from historical and current-season data, then explains *why* in plain English grounded in real FIA regulations, steward precedent, and circuit history — not a hallucinated guess. Also answers live, scenario-based championship questions ("what does X need to do to win the title this weekend?"). Success is a fan opening the app during a race weekend, understanding what's predicted and why in under a minute, and trusting the explanation enough to repeat it to someone else.

## Positioning

Most prediction content (broadcast graphics, punter tip sites) states a number with no reasoning, or reasoning that's just a pundit's opinion. This system's explanation is generated from the same model that made the prediction (SHAP feature attribution drives what gets retrieved and explained) grounded in real regulation/precedent text — the "why" is traceable to actual documents and actual feature contributions, not vibes.

## Operating Context

- Predictions update automatically on a schedule (GitHub Actions, every 30 min) and sharpen through a race weekend as real practice/qualifying/grid data becomes available — a fan checking Thursday and checking Saturday should see a visibly more informed prediction, not the same static number.
- The qualifying-gap prediction is available before qualifying happens; the other three predictions exist from the moment a race is added to the calendar and only get more accurate as the weekend progresses — there's no "not available yet" state for any of the four, only "less informed yet."
- Historical/backtest data lets a fan check the model's track record on already-decided races, not just trust an unfalsifiable live number.
- A live-standings agent answers open-ended, scenario-based championship questions in the same session, with memory for follow-ups.

## Capabilities and Constraints

- Backend: FastAPI, already built (`src/api/main.py`) — `/races`, `/predictions/latest`, `/predictions/{season}/{round}`, `/predictions/{season}/{round}/explain` (per-driver SHAP numbers), `/backtest/races`, `/backtest/{season}/{round}`, `POST /explain` (full natural-language explanation, real — not a stub), `POST /ask-agent` (real live-standings agent with conversation memory via `conversation_id`).
- All four predictions and the natural-language explanation are real and working end to end today, verified against live 2026 season data — nothing in this product is a "coming soon" placeholder, unlike the state phase7-ui-backend-plan.md was originally written against.
- No user accounts, no write actions anywhere in the product — everything is read-only public F1 data.
- Free-tier hosting: the backend cold-starts after idling: the frontend must not blank the screen or read as broken while a first request wakes it.
- No live telemetry/timing feed (lap-by-lap, sector times) — the product operates at the level of session results and rolling form, not live race telemetry.

## Brand Commitments

No existing name, logo, or visual identity beyond the working title "F1 Race Predictor + Strategy Explainer" (the project plan's own name) — free to establish in new-work. Not FIA/F1-licensed or affiliated; must read as an independent analysis tool, not an official property (no team livery reproduction, no claiming official branding).

## Evidence on Hand

Real trained XGBoost models (4 targets, metrics in `src/models/saved/*_metrics.json`), a real historical backtest across 105 races (2022–2026), a real RAG corpus (FIA regulations, steward decisions, 25 circuit write-ups), and real live 2026-season data verified end-to-end (see PROGRESS.md's Phase 6/7 entries) — no invented sample data, testimonials, or case studies; every number the UI will show has a real backing source.

## Product Principles

1. Show real state, never a fabricated one — the "known sessions" strip reflects actual FastF1 data availability (practice/qualifying/grid/compound), not an invented progress bar; a cold-started backend shows a loading state, not a blank or fake-cached screen.
2. Every prediction is one click from its own reasoning — the SHAP breakdown and the natural-language explanation are always reachable from wherever a prediction is shown, not buried behind a separate flow.
3. Design for a fan's vocabulary, not a data scientist's — "grid position," "quali gap," "recent form" over "feature," "SHAP value," "target variable."
4. Currency is a first-class fact, not a footnote — every prediction view states when it was generated, since the product's core promise is that it updates through the weekend.
5. The four predictions and history/backtest are the product; the agent chat is a real, secondary surface, not a gimmick bolted on — it gets a genuine, uncluttered space, not a corner widget.

## Accessibility & Inclusion

No user-specific accessibility requirement was stated; hold to WCAG AA as a baseline given real numeric/tabular data (contrast, focus rings, `prefers-reduced-motion`) per phase7-ui-backend-plan.md's own accessibility section.
