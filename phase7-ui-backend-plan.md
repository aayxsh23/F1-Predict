# Phase 7 — UI + Backend Architecture Plan

**Status: forward-looking design only, nothing here is built yet.** Phases 4-6 (RAG explainer, LLM fine-tuning, LangGraph agent) come first per [AGENTS.md](AGENTS.md)'s build-in-order convention and [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md)'s phase order — this document exists so the design work doesn't have to be redone from scratch when Phase 7 actually starts. Written by a dedicated planning pass (2026-09-11) once Phase 3 was complete; see [PROGRESS.md](PROGRESS.md) for that entry.

## Context

The user asked for a UI + backend integration plan while Phase 4-6 work is still pending, using the ui-ux-pro-max / frontend-design / impeccable design skills — the intent is a professional, minimal-aesthetic custom frontend (not the project plan's Streamlit fallback), designed against the **full end-state** product (predictions, explainer, agent) so it doesn't need a redesign once Phase 4/6 land, while being honest that only the two predictors (Phase 2/3) are real today.

Two things the exploration found that change the plan and can't be glossed over:

1. **The trained model files are gitignored** (`.gitignore` lines 31-32: `models/saved/`, `src/models/saved/`). A fresh deploy checkout has zero models. They're small (`finish_position_xgb.json` ≈ 964KB, `quali_delta_xgb.json` ≈ 83KB, verified) — cheap to commit instead.
2. **`data/processed/model_matrix.parquet`** (needed for a history/backtest view) is also gitignored and is the full ~100+ race training table — too large/slow to regenerate on a free-tier deploy. A backtest view needs its own small precomputed export, not a direct read of the training parquet.

## Core architectural decision: where "fresh" data lives

The project plan's own framing — a GitHub Actions job writes the latest prediction to a small file the app reads from — is right; this just makes it concrete: **freshness lives in GitHub, not in backend uptime.**

- The repo is public, so GitHub Actions minutes are free/unlimited — run the refresh job every 30 minutes without budget-tuning cron timing.
- The scheduled workflow computes predictions + SHAP and **commits the JSON straight back into the repo** (`data/predictions/latest.json`, `data/predictions/{season}_{round}.json`).
- The FastAPI backend fetches that JSON from `raw.githubusercontent.com` on each request, with a 60-second in-process TTL cache (a dict + timestamp — no Redis). A cold-started backend (free tier, ~15 min idle spindown) is instantly fresh, because it never needs to have been awake when the workflow ran. Backend uptime and data freshness are fully decoupled.
- Model files, by contrast, load once at process startup from disk (committed to the repo per point 1 above) — they only change on retrain, so there's no per-request fetch for those.

**Open consideration, not yet decided:** committing every 30 minutes to `main` forever will add a steady stream of small commits to the project's git history. That's a real, common pattern for "database-free" side projects, but worth a deliberate choice rather than a default — alternatives if it bothers you later: a dedicated `data` branch instead of `main`, or GitHub Pages/an artifact instead of commits at all. Similarly, `raw.githubusercontent.com` is a CDN for file content, not an official production data API — fine at this project's scale, but worth knowing it's not what it's "meant" for if traffic ever became a concern.

This means the backend does almost no heavy work per request: predictions are a cache read, SHAP is an in-memory computation on an already-embedded feature row, the schedule endpoint is a cheap FastF1 metadata call. The one slow thing in the whole system — `live_predict.predict_upcoming_race`, which calls out to FastF1 live — only ever runs inside the GitHub Actions job, never inside a user-facing request.

## Information architecture / pages

| Route | Purpose | Data source |
|---|---|---|
| `/` | Dashboard: current/next race predictions, race picker | `GET /races`, `GET /predictions/latest` |
| `/race/:season/:round` | Predictions table (grid, quali, practice, both targets) + "known sessions" badges | `GET /predictions/{season}/{round}` |
| `/race/:season/:round/driver/:code` | SHAP "why" breakdown for both targets | `GET /predictions/{season}/{round}/explain` |
| `/history` | Past races with backtest data | `GET /backtest/races` |
| `/history/:season/:round` | Actual vs. predicted, one historical race | `GET /backtest/{season}/{round}` |
| `/explainer` | **Stub** — "coming in Phase 4" card, calls `POST /explain` for real, renders its 501 gracefully | `POST /explain` |
| `/agent` | **Stub** — chat shell, disabled input, calls `POST /ask-agent` for real | `POST /ask-agent` |

Nav: sidebar (desktop) / bottom nav (mobile), 4 sections — Predictions, History, Explainer, Agent — the last two carry a small "Phase 4" / "Phase 6" badge in the nav itself, marked rather than hidden or faked.

The "known sessions" strip (Practice / Qualifying / Grid / Compound badges) is driven directly by boolean fields the API computes via `.notna()` checks on the same live row `live_predict.py` already produces — no fake progress bar, no invented stage enum, consistent with [LEARNING.md](LEARNING.md)'s Phase 2.5 point that reality is already discrete about what it does and doesn't know yet.

## API contract

No version prefix — single consumer, trivial to add later, unrequested scaffolding to add now.

**Reused as-is, no new model/data logic:**
- `GET /races?season=2026` → `src.data.fastf1_client.event_schedule(season)`
- `GET /predictions/latest`, `GET /predictions/{season}/{round}` → cache-read of JSON produced by `live_predict.predict_upcoming_race` via the new `refresh_job.py` (never computed live in-request)
- `GET /health` → liveness only

**New work, explicitly scoped — this is the actual Phase 7 implementation effort:**

`src/models/explain.py` (new) — single-prediction SHAP, extending the pattern already proven in `training_common.shap_circuit_check` (`shap.TreeExplainer(model)`) to one row instead of a circuit-level aggregate:
```python
def shap_explanation(model, row: pd.Series, top_n: int = 8) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_as_df)  # single row
    # -> sorted by |shap| desc, top_n signed contributions + base_value + predicted_value
```
`GET /predictions/{season}/{round}/explain?driver=VER&target=finish_position` wraps this, reading the feature row already embedded in the cached prediction JSON — so it never touches FastF1 or rebuilds the feature matrix, staying fast even on a cold-started backend.

Response shape:
```json
{
  "driver": "VER", "target": "finish_position",
  "predicted_value": 3.41, "base_value": 11.2,
  "top_contributions": [
    {"feature": "grid_position", "value": 2.0, "shap": -4.1},
    {"feature": "driver_recent_form", "value": 0.92, "shap": -1.8},
    {"feature": "grid_x_overtaking_difficulty", "value": 0.5, "shap": 0.6}
  ]
}
```

`src/models/backtest_export.py` (new) — iterates historical rows via `build_dataset.build()` (the same function training already uses) and dumps actual-vs-predicted per race to `data/predictions/backtest/{season}_{round}.json`. Runs periodically (e.g. weekly, or manually after a retrain) — not every 30 minutes like live predictions.

`GET /backtest/races` → index of races with backtest data. `GET /backtest/{season}/{round}` → per-driver actual vs. predicted, both targets.

**Reserved seams for Phase 4/6 — stubs only, deliberately not designed further here:**

`POST /explain` — request `{season, round, driver, target}`; today returns `501` with `{"status": "not_implemented", "message": "RAG explainer lands in Phase 4"}`. Later: `{explanation, sources}`.

`POST /ask-agent` — request `{message, conversation_id?}`; same `501` stub today. Later: `{reply, tool_calls, conversation_id}`.

Prediction JSON payload shape (`refresh_job.py` writes it, the `/predictions` endpoints serve it, one driver shown):
```json
{
  "season": 2026, "round": 14, "location": "Madrid",
  "generated_at": "2026-09-11T14:00:00Z",
  "known_sessions": {"practice": true, "qualifying": false, "grid": false, "compound": false},
  "drivers": [{
    "driver": "VER", "team": "Red Bull",
    "grid_position": null, "quali_gap_to_pole": null, "practice_pace": 1.023,
    "predicted_finish_position": 3.41, "predicted_quali_to_race_delta": -0.8,
    "feature_row": { "...33 feature columns from prepare_features...": "..." }
  }]
}
```

## Backend build items (new files)

- `src/api/main.py` — FastAPI app + routes, `CORSMiddleware` scoped to the deployed frontend origin only.
- `src/api/schemas.py` — pydantic models mirroring the shapes above.
- `src/api/cache.py` — GitHub-raw JSON fetch + 60s in-memory TTL (dict + timestamp).
- `src/models/explain.py` — new per-prediction SHAP logic (the actual gap this plan fills).
- `src/models/backtest_export.py` — new historical actual-vs-predicted export.
- `src/models/refresh_job.py` — thin glue: calls the *unchanged* `predict_upcoming_race` + the new `explain.py` per driver, writes the JSON above. The only place invoking the slow FastF1 path, and it only ever runs inside GitHub Actions.
- `.github/workflows/refresh-predictions.yml` — cron `*/30 * * * *`, `pip install -r requirements.txt` (cached), `python -m src.models.refresh_job`, commit + push.
- `.gitignore` change: exempt `src/models/saved/*_xgb.json` (deploy input) from the existing ignore rule.

Everything else — `live_predict.py`, `predict.py`, `build_dataset.py`, `training_common.py`, `circuit_reference.py` — gets wrapped unchanged.

## Frontend architecture

Stack, each choice justified against a simpler alternative rather than defaulted to:
- **Vite + React + TypeScript**, not Next.js — no SSR need (all data comes from a separate API), so a static SPA is simpler to build and free-host than running a Next.js server.
- **react-router-dom** for the routes above.
- **TanStack Query** — the one dependency actually justified by the free-tier problem: `staleTime`, `retry` with backoff, `placeholderData: keepPreviousData` directly solve "backend might be cold, don't blank the screen." No Redux/Zustand — the query cache *is* the data layer; a small React context handles only the light/dark toggle (persisted to `localStorage`).
- Skip a query-persister package — a plain `localStorage.setItem` in `onSuccess` plus reading it back as `initialData` is ~5 lines for the one cached object this app has.
- **Tailwind CSS + shadcn/ui** for table/card/badge/tabs/skeleton primitives — avoids hand-building accessible dropdowns/dialogs.
- **Recharts** (what shadcn/ui's chart component wraps) for the SHAP bar chart and any backtest chart — one hero chart per prediction, not a chart-library sprawl.

Folder structure:
```
frontend/
  src/
    main.tsx, App.tsx
    routes/Dashboard.tsx, RaceDetail.tsx, DriverExplain.tsx,
           History.tsx, HistoryDetail.tsx, ExplainerStub.tsx, AgentStub.tsx
    components/
      layout/Sidebar.tsx, TopBar.tsx
      predictions/DriverTable.tsx, KnownSessionsBadges.tsx
      charts/ShapBarChart.tsx
      ui/            (shadcn-generated)
    lib/api.ts, queryClient.ts, theme.tsx
    styles/tokens.css, globals.css
  tailwind.config.ts, vite.config.ts
```
`VITE_API_BASE_URL` env var, always explicit — frontend and backend are never co-hosted.

## Design system (concrete tokens)

**Primitives** (`styles/tokens.css`, `:root`):
- Neutrals: `--gray-0:#fff --gray-50:#f8f9fb --gray-100:#eef0f3 --gray-200:#dde1e6 --gray-300:#c3c9d1 --gray-400:#9aa2ad --gray-500:#717a87 --gray-600:#4f5761 --gray-700:#363c44 --gray-800:#22262b --gray-900:#131519 --gray-950:#0a0b0d`
- Accent, used sparingly — primary actions/highlights only, not chrome (a controlled nod to F1 red, not a full livery): `--red-400:#ff6b5e --red-500:#e8483a --red-600:#c22e22 --red-700:#961f17`
- Data-signal colors for signed SHAP values: `--green-500:#1f9d55` (pushes prediction better), reuse `--red-500` for worse.
- Spacing (4px base): `--space-1:4 --space-2:8 --space-3:12 --space-4:16 --space-5:24 --space-6:32 --space-7:48 --space-8:64` (px)
- Type scale, smaller than a marketing site since this is dense tabular data: `--text-xs:12/16 --text-sm:13/18 --text-base:14/20 --text-md:16/24 --text-lg:18/26 --text-xl:22/28 --text-2xl:28/34 --text-3xl:36/42` (px, size/line-height)
- Fonts: `--font-sans:"Inter",system-ui,sans-serif` for UI text; `--font-mono:"JetBrains Mono",ui-monospace,monospace` for numeric table cells (driver codes, positions, deltas) — monospaced numerals stop column jitter in a dense table, a functional reason, not just a look.
- Radius: `--radius-sm:4 --radius-md:8 --radius-lg:12` (px) — tight, not bubbly.
- Shadow: `--shadow-sm:0 1px 2px rgba(0,0,0,.06)`, `--shadow-md:0 4px 12px rgba(0,0,0,.12)`.

**Semantic** (light `:root`, dark `[data-theme=dark]`):
- Light: `--bg-canvas:var(--gray-50) --bg-surface:var(--gray-0) --border-default:var(--gray-200) --text-primary:var(--gray-900) --text-secondary:var(--gray-600) --text-muted:var(--gray-400) --accent-default:var(--red-500) --accent-hover:var(--red-600)`
- Dark: `--bg-canvas:var(--gray-950) --bg-surface:var(--gray-900) --border-default:var(--gray-700) --text-primary:var(--gray-50) --text-secondary:var(--gray-300) --text-muted:var(--gray-500) --accent-default:var(--red-400) --accent-hover:var(--red-500)`
- Data-viz: `--chart-positive:var(--green-500) --chart-negative:var(--red-500) --chart-known:var(--green-500) --chart-pending:var(--gray-400)`

**Component tokens** (representative, not exhaustive):
- Table: `--table-row-height:40px --table-header-bg:var(--bg-surface) --table-row-hover-bg:var(--gray-100)`/dark `var(--gray-800)` `--table-cell-padding-x:var(--space-3)`; numeric columns set `font-family:var(--font-mono)`.
- Known/pending badge: `--badge-known-bg:color-mix(in srgb, var(--green-500) 15%, transparent) --badge-known-text:var(--green-500) --badge-pending-bg:var(--gray-100) --badge-pending-text:var(--gray-500)`.
- Card: `--card-padding:var(--space-5) --card-radius:var(--radius-md) --card-bg:var(--bg-surface) --card-border:var(--border-default)`.

Accessibility: WCAG AA — use `--red-600` (not `--red-500`) for accent text at small sizes on light backgrounds to clear 4.5:1; every focusable element keeps a visible focus ring (never `outline:none` without a replacement); respect `prefers-reduced-motion` on chart transitions.

Wiring: Tailwind's `theme.extend.colors` references these CSS custom properties directly (e.g. `text-primary: 'var(--text-primary)'`), not duplicated hex values in `tailwind.config.ts` — one source of truth.

## Hosting/deployment

- **Frontend:** static `vite build` output → Cloudflare Pages (or Netlify/Vercel) free tier — pure static, no cold-start concept at all.
- **Backend:** Render free web service, `uvicorn src.api.main:app`. Free-tier spindown after 15 min idle is fine — see "Core architectural decision" above for why.
- **Automation:** `.github/workflows/refresh-predictions.yml`, cron every 30 min, runs `refresh_job.py`, commits `data/predictions/*.json`.
- **Model artifacts:** commit `src/models/saved/*_xgb.json` — the one `.gitignore` change required.
- **CORS:** locked to the deployed frontend origin.
- **Secrets:** none needed for the stub endpoints; Phase 4/6 will need an LLM API key added to Render's env dashboard when they land — not designed now.

## Phased build order

**7a — API skeleton + predictions-only UI**
1. Un-ignore + commit the two model JSON files; write `refresh_job.py` + the GH Actions workflow.
2. `src/api/main.py`/`schemas.py`: `/health`, `/races`, `/predictions/latest`, `/predictions/{season}/{round}` (cache-read only, no live compute).
3. Frontend scaffold (Vite/React/Tailwind/shadcn, token CSS, Sidebar/TopBar, Dashboard + RaceDetail, TanStack Query, theme toggle).

Output: a real, deployed "pick a race, see predictions" product, proving the whole free-tier data flow end to end before anything fancier is built.

**7b — SHAP explanation endpoint + "why" panel**
4. `src/models/explain.py` + `/predictions/{season}/{round}/explain`.
5. `DriverExplain` page + `ShapBarChart`.

Output: the per-prediction breakdown — also the exact payload shape Phase 4's RAG retrieval will eventually consume, so this validates that contract early.

**7c — Backtest/history view**
6. `src/models/backtest_export.py` + `/backtest/*` endpoints.
7. `History`/`HistoryDetail` pages.

**7d — Explainer/agent stub seams (deliberately last and thin)**
8. `POST /explain`, `POST /ask-agent` returning the `501` stub shape.
9. `ExplainerStub`/`AgentStub` pages + nav badges, wired to the real stub endpoints (not faked client-side) — Phase 4/6 only ever change the backend handler body, never the frontend contract.

Skipped deliberately: API versioning prefix (single consumer, add later if needed), Redis/DB (GitHub-as-datastore + a dict TTL cache covers this request volume), a query-persister package (5 lines of `localStorage` does the job), Next.js SSR (no server-rendering need).

## Critical files (existing code this plan wraps, not modifies)

- [src/models/live_predict.py](src/models/live_predict.py) — `predict_upcoming_race`, the live prediction entry point
- [src/models/predict.py](src/models/predict.py) — `load_model`, `predict`
- [src/models/training_common.py](src/models/training_common.py) — `shap_circuit_check`, the SHAP pattern `explain.py` extends
- [src/features/build_dataset.py](src/features/build_dataset.py) — `build`, reused by `backtest_export.py`
- [src/data/fastf1_client.py](src/data/fastf1_client.py) — `event_schedule`
- [.gitignore](.gitignore) — needs the model-artifact exemption described above

## Verification (when this is actually built)

- `refresh_job.py` run manually once → confirm `data/predictions/latest.json` matches `live_predict.predict_upcoming_race`'s output for a real upcoming race.
- Backend: `uvicorn src.api.main:app --reload`, hit each endpoint with `curl`/httpie against a race with real cached JSON; confirm `/explain` and `/ask-agent` return the `501` stub shape, not a 404 or crash.
- Frontend: `npm run dev` against the local backend; walk every route in the table above, confirm the "known sessions" badges match a race's actual FastF1 state (test against one race with zero sessions run, per the Madrid precedent from this session).
- Deploy dry run: push to a test branch, confirm Render cold-starts correctly and the GitHub Actions workflow's commit shows up as fresh data without the backend needing to be awake when it ran.
- Accessibility: run the deployed frontend through axe DevTools or Lighthouse for the WCAG AA contrast/focus-ring checks called out above.
