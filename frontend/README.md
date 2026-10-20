# Pit Wall — frontend

Vite + React + TypeScript SPA for the F1 race predictor. Talks to the FastAPI backend in `../src/api/` — nothing here computes predictions or calls FastF1/the LLM directly.

## Run locally

```
npm install
npm run dev
```

Requires the backend running (`python -m uvicorn src.api.main:app` from the repo root) at `http://localhost:8000` by default. Point at a different backend with `VITE_API_BASE_URL` (see `.env.example`).

## Build

```
npm run build
```

Static output in `dist/` — no server-rendering, deployable to any static host (Cloudflare Pages, Netlify or Vercel).

## Checks

```
npm run lint && npm test          # oxlint, Vitest (src/**/*.test.ts)
npm run size                      # first-load JavaScript must stay under 150 KB gzipped
npx playwright test               # smoke + axe on desktop and phone, against e2e/fixtures/
```

`python scripts/e2e_fixtures.py` (repo root) re-records the fixtures from the real API. The design system is [DESIGN.md](DESIGN.md).
