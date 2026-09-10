# PROGRESS.md

Living build log — the actual current state of the repo, as opposed to [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) (the design) or [AGENTS.md](AGENTS.md) (the rules). Every agent updates this file after any meaningful change, before ending its session, newest entry first. Keep entries short.

Read this file first when picking up work in a new session — it tells you what actually happened, not just what was planned.

---

## 2026-09-10 — Repo scaffolded

- Created folder structure for Phases 1-6 (`data/`, `src/{data,features,models,rag,agent,api}`, `notebooks/`, `tests/`, `.github/workflows/`) — no code yet.
- Added `requirements.txt`, expanded `.gitignore` (FastF1 cache, model artifacts, vector store, IDE/OS cruft).
- Configured git remote `origin` to `https://github.com/aayxsh23/F1-Predict.git`.
- Added AGENTS.md (canonical instructions), CLAUDE.md + GEMINI.md (thin pointers to it), and this file.
- **Status:** Phase 1 (data foundation) not started.
- **Next:** pull historical race data via FastF1/Jolpica-F1, build the feature pipeline + circuit reference table.
