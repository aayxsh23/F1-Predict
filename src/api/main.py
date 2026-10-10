"""FastAPI backend. Serves what the pipeline precomputed (data/predictions/,
written by the scheduled refresh job and the training run) plus a few cheap
live computations: odds, explanations, strategy, championship odds and chat.

Nothing here imports pandas, XGBoost, torch, FastF1 or a PDF parser (the
refresh job precomputes everything that needs them, SHAP breakdowns included),
so it fits a serverless function: see api/index.py for the Vercel entry point.
Run locally: `uvicorn src.api.main:app --reload`.
"""
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
from src.api.enrich import with_probabilities
from src.api.schemas import ChatRequest, ExplainRequest
from src.models.catalog import CANONICAL_PRED_COLS, FEATURE_LABELS, MODEL_DIR
from src.rag import corpus

app = FastAPI(title="Pit Wall F1 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TARGET_DIRECTION = {  # what a positive contribution means, per target
    "finish_position": "towards a worse finish", "qualifying": "further from pole",
    "quali_delta": "towards gaining more places", "race_time": "further behind the winner",
}
LLM_LIMIT, LLM_WINDOW_S = 20, 600
_calls: dict[str, deque] = defaultdict(deque)


def _limit(request: Request) -> None:
    """20 LLM calls per 10 minutes per client IP."""
    # ponytail: per-instance memory, so each serverless instance counts separately; use Upstash/Vercel KV if abused
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    q, now = _calls[ip], time.monotonic()
    while q and now - q[0] > LLM_WINDOW_S:
        q.popleft()
    if len(q) >= LLM_LIMIT:
        raise HTTPException(429, "Too many questions in a short time. Try again in a few minutes.")
    q.append(now)


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
    return {"status": "ok", "latest_forecast_at": generated, "latest_forecast_age_hours": None if age is None else round(age, 1)}


@app.get("/races")
def races(season: int):
    return _json(f"schedule/{season}.json", f"No calendar for {season}.")


@app.get("/predictions/latest")
def predictions_latest():
    return with_probabilities(_json("latest.json", "No forecasts yet -- run src.models.refresh_job."))


@app.get("/predictions/{season}/{round}")
def predictions_for_race(season: int, round: int):
    return with_probabilities(_forecast(season, round))


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
    return out


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


@app.get("/championship")
def championship():
    from src.agent.tools import championship_state

    try:
        c = championship_state()
    except Exception as exc:  # Jolpica down or rate-limited
        raise HTTPException(503, f"Live standings unavailable right now ({type(exc).__name__}).")
    from src.agent.f1_api import get_constructor_standings

    odds = c["odds"]
    return {
        "remaining": {"races": len(c["remaining"]), "sprints": sum(r["is_sprint"] for r in c["remaining"])},
        "drivers": [{**s, "title_chance": round(float(odds["champion"][i]), 4),
                     "expected_points": round(float(odds["expected_points"][i]), 1)} for i, s in enumerate(c["standings"])],
        "constructors": [{**t, "title_chance": round(float(odds["team_champion"].get(t["name"], 0.0)), 4)}
                         for t in get_constructor_standings()],
    }


@app.post("/explain")
def explain_endpoint(req: ExplainRequest, request: Request):
    """A plain-English paragraph on why the model predicts what it does: the
    SHAP breakdown is the query into the corpus, and the local model writes from both."""
    from src.agent.chat import ExplainerUnavailable, explain_text

    _limit(request)
    payload = _forecast(req.season, req.round)
    d = _driver(payload, req.driver)
    exp = stored_explanation(d, req.target)
    top = exp["top_contributions"][:5]
    labels = [c["label"] for c in top]
    circuit = payload["location"]
    query = f"{circuit} " + " ".join(labels)
    hits = corpus.search(query, k=3, circuit=circuit)
    lines = [f"- {lab}: value {c['value']}, effect {c['shap']:+.2f} ({'pushes ' + TARGET_DIRECTION[req.target] if c['shap'] > 0 else 'pulls the other way'})"
             for lab, c in zip(labels, top)]
    facts = (f"Driver {d['driver']} ({d['team']}) at {circuit}, {payload['season']} round {payload['round']}.\n"
             f"Prediction ({req.target.replace('_', ' ')}): {exp['predicted_value']:.2f}; an average driver here: {exp['base_value']:.2f}.\n"
             "Inputs that moved it most:\n" + "\n".join(lines) + "\n\nContext:\n"
             + "\n\n".join(f"[{h['source']}{' art. ' + h['article'] if h['article'] else ''}]\n{h['text'][:1200]}" for h in hits))
    try:
        text = explain_text(facts)
    except ExplainerUnavailable as exc:
        raise HTTPException(503, str(exc))
    return {
        "prediction": exp["predicted_value"], "target": req.target, "circuit": circuit,
        "top_features": [{"feature": c["feature"], "shap_value": c["shap"], "phrase": lab} for c, lab in zip(top, labels)],
        "retrieval_query": query, "sources": [h["source"] for h in hits], "explanation": text,
    }


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
