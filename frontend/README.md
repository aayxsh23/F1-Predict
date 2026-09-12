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

Static output in `dist/` — no server-rendering, deployable to any static host (Cloudflare Pages/Netlify/Vercel per phase7-ui-backend-plan.md).
