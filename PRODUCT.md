# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + React + TypeScript (static SPA, no server-rendering need — all data comes from a separate FastAPI backend). react-router-dom for routing. TanStack Query as the data layer (no Redux/Zustand — the query cache is the state). Tailwind CSS v4 with hand-built components (no component library), Framer Motion for sheet and tab motion, React Three Fiber + drei for the podium stage (lazy-loaded, with a CSS 3D fallback); charts are hand-drawn SVG/CSS. All workbench state lives in URL params, and so does the pit wall's (the race in the path, `?lens=` for the target). The original stack was decided in [phase7-ui-backend-plan.md](phase7-ui-backend-plan.md); the 2026-09-25 workbench rebuild replaced shadcn/Radix/Recharts (see PROGRESS.md).

## Users

F1 fans who want to know what's likely to happen in an upcoming race weekend, and why — not F1 insiders needing broadcast-grade telemetry, and not a technical audience needing the ML pipeline explained to them. They arrive already knowing the sport (driver names, team names, what qualifying/grid/practice mean) but not knowing anything about SHAP, XGBoost, or RAG. The product is a real prediction tool first; it never frames itself as a portfolio demo or explains its own ML internals as a feature.

## Product Purpose

Predicts F1 race weekend outcomes (qualifying gap-to-pole, finishing position, grid-to-finish delta, race-time gap-to-winner) from historical and current-season data, then explains *why* in plain English grounded in real FIA regulations, steward precedent, and circuit history — not a hallucinated guess. Also answers live, scenario-based championship questions ("what does X need to do to win the title this weekend?"). Success is a fan opening the app during a race weekend, understanding what's predicted and why in under a minute, and trusting the explanation enough to repeat it to someone else.

## Positioning

Most prediction content (broadcast graphics, punter tip sites) states a number with no reasoning, or reasoning that's just a pundit's opinion. This system's explanation is generated from the same model that made the prediction (SHAP feature attribution drives what gets retrieved and explained) grounded in real regulation/precedent text — the "why" is traceable to actual documents and actual feature contributions, not vibes.

## Operating Context

- Forecasts refresh automatically (GitHub Actions: every 30 minutes Thursday to Sunday, daily otherwise) and sharpen through a race weekend as practice, qualifying and grid data arrive. A fan checking Thursday and again Saturday night sees a visibly better-informed forecast, and the "Why" tab shows how it moved session by session.
- Every prediction exists from the moment a race is next on the calendar; there is no "not available yet", only "less informed yet". Qualifying is predicted before it happens.
- Past races show what the model predicted before the race next to what happened, from a model trained only on earlier races, so the track record can be checked, not taken on trust.
- The analyst chat answers open questions (forecasts, strategy, the title fight, the rules) with every number drawn from the app's own data.

## Capabilities and Constraints

- Backend: FastAPI (`src/api/main.py`), deployable as one Vercel serverless function beside the static frontend (`vercel.json`, `api/index.py`) or self-hosted. It serves precomputed JSON (forecasts with their SHAP breakdowns, the walk-forward archive, the regulations index) and does light live work: odds sampling, the strategy simulator, championship odds and the assistant (`POST /chat`, server-sent events) -- a deterministic LangGraph router for almost every intent, with a local, LoRA-fine-tuned Llama 3.2 1B for exactly two ("why is this predicted?" and open-ended questions), used only to write prose around numbers the router already computed, never to choose data or do arithmetic. No external LLM API anywhere in the product. Full route list: README.md.
- Predictions: qualifying (gap and lap time, pole/Q3/Q1-exit odds), finishing position (win/podium/points odds, likely range, retirement risk, beats-teammate), places gained, and gap to the winner with an estimated race length; tyre strategy with pit windows and a safety-car scenario; championship title odds.
- No user accounts and nothing written by users; the only user input is chat text, rate-limited per IP and never stored (the conversation lives in the browser).
- Free-tier hosting cold-starts after idling: the frontend must never blank the screen or read as broken while the first request wakes it.
- No live timing feed: the product works at the level of sessions, stints and rolling form, not lap-by-lap telemetry during a session.

## Brand Commitments

No existing name, logo, or visual identity beyond the working title "F1 Race Predictor + Strategy Explainer" (the project plan's own name) — free to establish in new-work. Not FIA/F1-licensed or affiliated; must read as an independent analysis tool, not an official property (no team livery reproduction, no claiming official branding).

## Evidence on Hand

Real trained XGBoost models (4 targets, held-out metrics in `src/models/saved/*_metrics.json`, explained in docs/MODELS.md), a walk-forward archive of 97 races (2022–2026), lap-by-lap stint data behind the strategy model, a regulations index (FIA 2026 regulations, steward decisions, 27 circuit write-ups), and live 2026-season data verified end to end — no invented sample data, testimonials, or case studies; every number the UI will show has a real backing source.

## Product Principles

1. Show real state, never a fabricated one — the "known sessions" strip reflects actual FastF1 data availability (practice/qualifying/grid/compound), not an invented progress bar; a cold-started backend shows a loading state, not a blank or fake-cached screen.
2. Every prediction is one click from its own reasoning — the SHAP breakdown and the natural-language explanation are always reachable from wherever a prediction is shown, not buried behind a separate flow.
3. Design for a fan's vocabulary, not a data scientist's — "grid position," "quali gap," "recent form" over "feature," "SHAP value," "target variable."
4. Currency is a first-class fact, not a footnote — every prediction view states when it was generated, since the product's core promise is that it updates through the weekend.
5. The predictions and their track record are the product; the analyst chat is a real way into them, not a gimmick: it gets its own uncluttered space, answers only from the app's data, and cites the rules it uses.

## Accessibility & Inclusion

No user-specific accessibility requirement was stated; hold to WCAG AA as a baseline given real numeric/tabular data (contrast, focus rings, `prefers-reduced-motion`) per phase7-ui-backend-plan.md's own accessibility section.
