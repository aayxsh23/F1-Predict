"""FastAPI backend. Serves what the pipeline precomputed (data/predictions/,
written by the scheduled refresh job and the training run) plus a few cheap
live computations: odds, explanations, strategy, championship odds and chat.

Nothing here imports pandas, XGBoost, torch, FastF1 or a PDF parser (the
refresh job precomputes everything that needs them, SHAP breakdowns included),
so it fits a serverless function: see api/index.py for the Vercel entry point.
Run locally: `uvicorn src.api.main:app --reload`.
"""
import builtins
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.cache import get_json
from src.api.enrich import with_circuit, with_probabilities
from src.api.schemas import ChatRequest, ExplainRequest
from src.models.catalog import CANONICAL_PRED_COLS, FEATURE_LABELS, MODEL_DIR
from src.rag import corpus

app = FastAPI(title="Pit Wall F1 API")
# Deployed, the app calls the API on its own origin (/api), which needs no CORS
# at all; only the Vite dev server on localhost, or a FRONTEND_ORIGIN set on
# purpose, may call it from another origin. (Was "*": any website could use the chat.)
_origins = os.environ.get("FRONTEND_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins.split(",") if _origins else [],
    allow_origin_regex=None if _origins else r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

LLM_LIMIT, LLM_WINDOW_S = 20, 600
GLOBAL_LIMIT = 300  # per instance and window, all clients together: a bound even if IPs rotate
# GET responses only change when a scheduled job publishes new files, so a CDN
# (Vercel's edge) may serve them for a minute and keep serving a stale copy
# for ten while it refetches: most reads never wake the serverless function
CACHE_CONTROL = "public, max-age=0, s-maxage=60, stale-while-revalidate=600"


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200 and request.url.path not in ("/health",):
        response.headers.setdefault("Cache-Control", CACHE_CONTROL)
    return response
_calls: dict[str, deque] = defaultdict(deque)


def _limit(request: Request) -> None:
    """20 chat/explain calls per 10 minutes per client IP, and GLOBAL_LIMIT for everyone."""
    # ponytail: per-instance memory. Exact where the local model runs (one process on the owner's PC); on Vercel
    # each serverless instance counts separately, acceptable while that tier has no model; use Upstash/Vercel KV if abused
    # Vercel sets x-real-ip itself; anywhere else a client could forge it, so the socket address is used
    ip = request.headers.get("x-real-ip") if os.environ.get("VERCEL") else None
    ip = ip or (request.client.host if request.client else "?")
    now = time.monotonic()
    for key, cap in ((ip, LLM_LIMIT), ("*", GLOBAL_LIMIT)):
        q = _calls[key]
        while q and now - q[0] > LLM_WINDOW_S:
            q.popleft()
        if len(q) >= cap:
            raise HTTPException(429, "Too many questions in a short time. Try again in a few minutes.")
    _calls[ip].append(now)
    _calls["*"].append(now)


def _json(path: str, missing: str):
    try:
        return get_json(path)
    except FileNotFoundError:
        raise HTTPException(404, missing)


def _forecast(season: int, round: int) -> dict:
    return _json(f"{season}_{round}.json", "No forecast for this race yet.")


def _driver(payload: dict, driver: str) -> dict:
    row = next((d for d in payload["drivers"] if d["driver"] == driver.upper()), None)
    if row is None:
        raise HTTPException(404, f"No driver '{driver}' in this race's forecast.")
    return row


def stored_explanation(driver_row: dict, target: str) -> dict:
    """The SHAP breakdown the refresh job stored, with fan labels added."""
    exp = driver_row.get("explanations", {}).get(target)
    if exp is None:
        raise HTTPException(404, "This forecast has no stored breakdown (it predates them).")
    return {**exp, "top_contributions": [{**c, "label": FEATURE_LABELS.get(c["feature"], c["feature"])} for c in exp["top_contributions"]]}


@app.get("/health")
def health():
    try:
        generated = get_json("latest.json").get("generated_at")
    except FileNotFoundError:
        generated = None
    age = None if generated is None else (datetime.now(timezone.utc) - datetime.fromisoformat(generated)).total_seconds() / 3600
    from src.rag.llm import is_available

    return {"status": "ok", "latest_forecast_at": generated, "latest_forecast_age_hours": None if age is None else round(age, 1),
            "chat_model": is_available()}  # the local model answers open questions here (never on the serverless API)


@app.get("/races")
def races(season: int):
    return _json(f"schedule/{season}.json", f"No calendar for {season}.")


@app.get("/predictions/latest")
def predictions_latest():
    return with_circuit(with_probabilities(_json("latest.json", "No forecasts yet -- run src.models.refresh_job.")))


@app.get("/predictions/{season}/{round}")
def predictions_for_race(season: int, round: int):
    return with_circuit(with_probabilities(_forecast(season, round)))


@app.get("/predictions/{season}/{round}/explain")
def predictions_explain(season: int, round: int, driver: str, target: str = "finish_position"):
    if target not in CANONICAL_PRED_COLS:
        raise HTTPException(400, f"Unknown target '{target}'; expected one of {list(CANONICAL_PRED_COLS)}.")
    d = _driver(_forecast(season, round), driver)
    return {"driver": d["driver"], "target": target, **stored_explanation(d, target)}


@app.get("/predictions/{season}/{round}/timeline")
def predictions_timeline(season: int, round: int):
    try:
        return get_json(f"timeline/{season}_{round}.json")
    except FileNotFoundError:
        return []


@app.get("/backtest/races")
def backtest_races():
    return _json("backtest/index.json", "No backtest yet -- run src.models.backtest_export.")


@app.get("/backtest/summary")
def backtest_summary():
    return _json("backtest/summary.json", "No backtest yet -- run src.models.backtest_export.")


@app.get("/backtest/{season}/{round}")
def backtest_for_race(season: int, round: int):
    return _json(f"backtest/{season}_{round}.json", "No backtest for this race.")


@app.get("/live-record")
def live_record():
    """The forecasts the app published, scored against the results (src/models/live_record.py)."""
    return _json("live_record.json", "No published forecast has been scored yet.")


@app.get("/model")
def model_card():
    """Held-out accuracy of each model (the numbers the track-record view shows)."""
    out = {}
    for t in CANONICAL_PRED_COLS:
        m = json.loads((MODEL_DIR / f"{t}_metrics.json").read_text())
        out[t] = {k: m[k] for k in ("unit", "baseline", "evaluation", "stages", "data_through", "trained_at")}
        out[t]["top_features"] = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "weight": w} for f, w in m["top_features"].items()]
    # the odds on held-out races: "when we said 30%, it happened ...%" (phase 10 item 6.4)
    checks = json.loads((MODEL_DIR / "pl_calibration.json").read_text()).get("checks", {})
    out["odds"] = {stage: {k: c[k] for k in ("held_out_races", "win_reliability", "brier", "ranked_probability_score",
                                              "ranked_probability_score_if_guessing_evenly", "calibration_error") if k in c}
                   for stage, c in ((s.removeprefix("race_"), c) for s, c in checks.items() if s.startswith("race_"))}
    return out


@app.get("/whatif/{season}/{round}")
def what_if(season: int, round: int, driver: str, grid: int):
    """One driver's odds if they started from `grid`, the rest of the field as
    forecast. The finish for every slot was predicted by the real models when
    the forecast was published (src/models/whatif.py); this only re-runs the
    race sampler, so it costs milliseconds."""
    from src.models.probabilities import race_probabilities

    payload = _forecast(season, round)
    table = payload.get("whatif_grid") or {}
    row = _driver(payload, driver)
    slots = table.get(row["driver"])
    if not slots:
        raise HTTPException(404, "No what-if for this race's forecast.")
    if not 1 <= grid <= len(slots):
        raise HTTPException(422, f"grid must be between 1 and {len(slots)}")
    drivers = payload["drivers"]
    i = next(k for k, d in enumerate(drivers) if d["driver"] == row["driver"])
    now = [d["predicted_finish_position"] for d in drivers]
    then = now[:i] + [slots[grid - 1]] + now[i + 1:]
    seed = season * 100 + round
    teams = [d["team"] for d in drivers]

    def odds(pred):  # builtins.round: the route's `round` (the race) shadows it
        r = race_probabilities(pred, stage="post_quali", teams=teams, seed=seed)
        return {"win": builtins.round(float(r["win"][i]), 4), "podium": builtins.round(float(r["podium"][i]), 4),
                "points": builtins.round(float(r["top10"][i]), 4), "expected_position": builtins.round(float(r["expected_position"][i]), 2)}

    # "current" is what the app shows now; the what-if also assumes the grid is settled, as it is after qualifying
    shown = with_probabilities(payload)["drivers"][i]
    current = {"win": shown["probabilities"]["win"], "podium": shown["probabilities"]["podium"],
               "points": shown["probabilities"]["top10"], "expected_position": shown["expected_position"]}
    current_grid = row.get("grid_position")
    return {"driver": row["driver"], "grid": grid, "current_grid": int(current_grid) if current_grid else None,
            "current": current, "what_if": odds(then), "predicted_finish": {"current": now[i], "what_if": slots[grid - 1]},
            "note": None if current_grid else "Before qualifying: the other qualifying inputs are the model's own predictions."}


@app.get("/regulations")
def regulations_list():
    return corpus.list_documents()


# declared before /regulations/{filename}, which would otherwise swallow "search"
@app.get("/regulations/search")
def regulations_search(query: str, k: int = 5):
    return corpus.search_documents(query, k=min(k, 20))


@app.get("/regulations/{filename}")
def regulations_get(filename: str):
    try:
        return {"filename": filename, "text": corpus.get_document_text(filename)}
    except FileNotFoundError:
        raise HTTPException(404, f"No document named '{filename}'.")


@lru_cache(maxsize=64)
def _strategy(location: str, laps: int, sc_lap: int | None) -> dict:
    from src.strategy.model import simulate

    return simulate(location, laps, sc_lap=sc_lap)


@app.get("/strategy/{season}/{round}")
def strategy(season: int, round: int, sc_lap: int | None = None):
    p = _forecast(season, round)
    laps = int(p.get("race", {}).get("laps") or 57)
    if sc_lap is not None and not 1 <= sc_lap < laps:
        raise HTTPException(400, f"sc_lap must be between 1 and {laps - 1}.")
    return {"season": season, "round": round, **_strategy(p["location"], laps, sc_lap)}


@app.get("/ratings")
def ratings():
    """Drivers' qualifying ratings from teammate comparisons (src/features/ratings.py)."""
    return _json("ratings.json", "No driver ratings published yet.")


@app.get("/championship")
def championship():
    """Standings and title odds: precomputed by the refresh job
    (championship.json); computed live only if that file is missing."""
    try:
        return get_json("championship.json")
    except FileNotFoundError:
        pass
    from src.agent.tools import championship_payload

    try:
        return championship_payload()
    except Exception as exc:  # Jolpica down or rate-limited
        raise HTTPException(503, f"Live standings unavailable right now ({type(exc).__name__}).")


@app.post("/explain")
def explain_endpoint(req: ExplainRequest, request: Request):
    """A plain-English paragraph on why the model predicts what it does,
    written by the local model from the stored SHAP breakdown (503 where the
    model isn't running, e.g. the serverless API)."""
    from src.agent.chat import ExplainerUnavailable, explain_text

    _limit(request)
    d = _driver(_forecast(req.season, req.round), req.driver)
    exp = stored_explanation(d, req.target)
    try:
        text, sources = explain_text(req.season, req.round, d["driver"], req.target)
    except ExplainerUnavailable as exc:
        raise HTTPException(503, str(exc))
    return {"prediction": exp["predicted_value"], "target": req.target, "explanation": text, "sources": sources}


@app.post("/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    """Server-sent events: one `data: {json}` line per event (see src/agent/chat.py)."""
    from src.agent.chat import stream_chat

    if req.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be the user's.")
    _limit(request)
    history = [m.model_dump() for m in req.messages]
    context = req.context.model_dump() if req.context else None

    def events():
        for e in stream_chat(history, context):
            yield f"data: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
