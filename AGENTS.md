# AGENTS.md

Canonical instructions for any coding agent working in this repo (Claude, Gemini, or otherwise). [CLAUDE.md](CLAUDE.md) and [GEMINI.md](GEMINI.md) are thin, tool-specific pointers to this file — **this is the source of truth**, not them. If you find a rule duplicated and diverging between files, fix it here first and re-sync the others.

> **Always keep [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) in context.** It's the single source of truth for scope and phase order. Re-read it before planning any non-trivial change, and don't reintroduce anything its "Future phases" table defers.

## What this is

F1 Race Predictor + Strategy Explainer: an XGBoost predictor (finishing position + quali-to-race delta) whose SHAP output drives a LangChain RAG explainer, plus a LangGraph live-standings agent. Full rationale and phase breakdown: the project plan above.

## Hard scope constraints (do not violate without the user explicitly changing scope)

- **Exactly 2 predictor models**, both sharing one feature pipeline. No per-circuit or per-cluster models — circuit characteristics are ordinary columns in the shared feature table, not a reason to fork models.
- **No deep learning** in the core pipeline — XGBoost/LightGBM only. Embeddings (via the vector DB) are the only DL surface. A telemetry DL model is a deferred stretch goal, not core work.
- **Strategy simulation and cross-prediction reasoning are deferred** — don't build them into v1 even if they seem like a natural extension.
- Explainer must be built on **LangChain**, agent on **LangGraph** — not manual pipeline glue.
- Hosting must stay on **free tiers** (Render/Railway/Fly.io/Streamlit Community Cloud); automation via **GitHub Actions scheduled workflow**, not a manual script or a paid scheduler.

## Repo layout

```
data/
  raw/                  # pulled FastF1 / Jolpica-F1 data, gitignored — regenerate, don't commit
  processed/             # cleaned feature tables, gitignored
  corpus/
    regulations/         # FIA regulation docs (Phase 4 RAG corpus)
    steward_decisions/   # steward decision PDFs (Phase 4 RAG corpus)
    race_summaries/       # historical race summaries (Phase 4 RAG corpus)
src/
  data/                  # Phase 1 — FastF1/Jolpica ingestion
  features/               # Phase 1 — feature pipeline incl. circuit reference table
  models/                  # Phase 2-3 — the 2 predictors + SHAP
  rag/                      # Phase 4 — LangChain retrieval/explainer
  agent/                     # Phase 5 — LangGraph live-standings agent
  api/                        # Phase 6 — FastAPI endpoints (predict, explain, ask-agent)
notebooks/                     # exploration, not production code
tests/
.github/workflows/               # Phase 6 — scheduled re-prediction workflow
```

## Python environment

- **Never install packages into the global/system Python.** This project uses a venv at `.venv/` (already gitignored). All `pip install` and `python -m ...` invocations must go through it:
  - Windows agents/shells: `.venv/Scripts/python.exe` / `.venv/Scripts/pip.exe` (or, for a persistent interactive shell only — not across separate tool calls — activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/Scripts/activate` in Git Bash).
  - If `.venv/` doesn't exist yet, create it once with whichever Python is on PATH: `python -m venv .venv` — that bootstrap step is the only time the global interpreter is touched.
  - An agent's tool calls don't persist shell activation state between calls, so always invoke the venv's `python.exe`/`pip.exe` directly by path rather than relying on `activate` having stuck from a previous command.
- This rule exists because it was violated once already — `requirements.txt` got installed straight into the global interpreter (both by an agent running bare `pip install` and by the user re-running the same command), requiring a manual `pip uninstall` cleanup before `.venv/` was introduced. See PROGRESS.md/LEARNING.md for that incident.

## Conventions

- Build phases in order (1→6); each phase's Output in the plan is the acceptance bar before moving on.
- The circuit reference table (~24 rows) is merged onto race rows by circuit name — keep it in `src/features/`, not duplicated per model.
- SHAP output is the query into the RAG layer (Phase 4) — treat predictor and explainer as connected, not independent modules.

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
