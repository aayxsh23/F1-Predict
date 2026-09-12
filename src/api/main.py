"""FastAPI backend -- Phase 7. Wraps Phases 1-6 unchanged (predict.py,
live_predict.py, rag/explain.py, agent/graph.py); the only new logic is here
and in explain.py/backtest_export.py/refresh_job.py. See
phase7-ui-backend-plan.md for the full design.

Deviation from that plan, worth being explicit about: it was written before
Phases 4-6 existed, so it designed /explain and /ask-agent as 501 stubs
("RAG explainer lands in Phase 4"/"Phase 6"). Both are real now -- wiring
them to the actual rag.explain.explain()/agent.graph.ask() implementations
below is strictly more correct than reverting to a stub for functionality
that already works, so that's what this does instead."""
import os
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent.graph import ask as agent_ask
from src.agent.graph import build_graph
from src.api.cache import get_json
from src.api.schemas import AskAgentRequest, ExplainRequest
from src.data.fastf1_client import event_schedule
from src.models.explain import shap_explanation
from src.models.features import row_from_dict
from src.models.predict import CANONICAL_PRED_COLS, load_model
from src.rag.explain import explain as rag_explain

app = FastAPI(title="F1 Race Predictor API")

# no frontend deployed yet to scope this to -- tighten to the real origin
# once one exists (phase7-ui-backend-plan.md's "CORS locked to the deployed
# frontend origin"); nothing behind this API is sensitive or writes state.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "*")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_agent_graph = None


def _get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    return _agent_graph


@lru_cache(maxsize=4)
def _cached_model(target: str):
    return load_model(target)


def _driver_feature_row(payload: dict, driver: str) -> dict:
    row = next((d for d in payload["drivers"] if d["driver"] == driver), None)
    if row is None:
        raise HTTPException(404, f"no driver '{driver}' in this race's cached prediction")
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/races")
def races(season: int):
    sched = event_schedule(season)
    cols = ["RoundNumber", "EventName", "Location", "EventDate"]
    return sched[cols].to_dict("records")


@app.get("/predictions/latest")
def predictions_latest():
    try:
        return get_json("latest.json")
    except FileNotFoundError:
        raise HTTPException(404, "no predictions have been generated yet -- run src.models.refresh_job")


@app.get("/predictions/{season}/{round}")
def predictions_for_race(season: int, round: int):
    try:
        return get_json(f"{season}_{round}.json")
    except FileNotFoundError:
        raise HTTPException(404, "no cached prediction for this race yet")


@app.get("/predictions/{season}/{round}/explain")
def predictions_explain(season: int, round: int, driver: str, target: str = "finish_position"):
    if target not in CANONICAL_PRED_COLS:
        raise HTTPException(400, f"unknown target '{target}', expected one of {list(CANONICAL_PRED_COLS)}")
    try:
        payload = get_json(f"{season}_{round}.json")
    except FileNotFoundError:
        raise HTTPException(404, "no cached prediction for this race yet")

    driver_row = _driver_feature_row(payload, driver)
    row = row_from_dict(driver_row["feature_row"])
    result = shap_explanation(_cached_model(target), row, target=target)
    return {"driver": driver, "target": target, **result}


@app.get("/backtest/races")
def backtest_races():
    try:
        return get_json("backtest/index.json")
    except FileNotFoundError:
        raise HTTPException(404, "no backtest data yet -- run src.models.backtest_export")


@app.get("/backtest/{season}/{round}")
def backtest_for_race(season: int, round: int):
    try:
        return get_json(f"backtest/{season}_{round}.json")
    except FileNotFoundError:
        raise HTTPException(404, "no backtest data for this race")


@app.post("/explain")
def explain_endpoint(req: ExplainRequest):
    try:
        payload = get_json(f"{req.season}_{req.round}.json")
    except FileNotFoundError:
        raise HTTPException(404, "no cached prediction for this race yet")

    driver_row = _driver_feature_row(payload, req.driver)
    row = row_from_dict({**driver_row["feature_row"], "location": payload["location"]})
    return rag_explain(row, target=req.target)


@app.post("/ask-agent")
def ask_agent_endpoint(req: AskAgentRequest):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    reply = agent_ask(_get_agent_graph(), req.message, thread_id=conversation_id)
    return {"reply": reply, "conversation_id": conversation_id}
