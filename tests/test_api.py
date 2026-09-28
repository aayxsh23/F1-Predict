"""The FastAPI routes, against fixture forecast files plus the repo's committed
models, corpus index and strategy params. No network and no LLM: the chat and
explanation routes are checked for validation, the unconfigured-key path and
rate limiting. Run with `python tests/test_api.py`."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.cache as cache
import src.api.main as main
from src.agent import chat
from src.models.explain import _native, shap_explanation
from src.models.features import FEATURE_COLS, row_from_dict
from src.models.predict import load_model

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"
client = TestClient(main.app)


def _feature_rows(n: int) -> list[dict]:
    df = pd.read_parquet(DATA_PATH)
    rows = df[df["target_finish_position"].notna()].sample(n, random_state=7)
    return [{c: _native(r[c]) for c in FEATURE_COLS} for _, r in rows.iterrows()]


def _fixture() -> dict:
    drivers = []
    for i, (code, team, fr) in enumerate(zip(["VER", "HAM", "NOR", "PIA"], ["Red Bull", "Ferrari", "McLaren", "McLaren"], _feature_rows(4))):
        drivers.append({"driver": code, "team": team, "driver_number": i + 1, "grid_position": None,
                        "predicted_qualifying_gap_pct": 0.1 * i, "predicted_finish_position": 2.0 + i,
                        "predicted_quali_to_race_delta": 0.0, "predicted_race_gap_pct": 0.2 * i, "feature_row": fr})
    return {"season": 2026, "round": 14, "location": "Madrid", "generated_at": "2026-09-11T14:00:00+00:00",
            "stage": "post_practice", "known_sessions": {"practice": True, "qualifying": False, "grid": False, "compound": False},
            "race": {"laps": 57}, "drivers": drivers}


def _with_fixture_dir(fn):
    def run():
        with tempfile.TemporaryDirectory() as tmp:
            old = cache.LOCAL_DATA_DIR
            cache.LOCAL_DATA_DIR, root = Path(tmp), Path(tmp)
            cache._cache.clear()
            payload = _fixture()
            for name in ("latest.json", "2026_14.json"):
                (root / name).write_text(json.dumps(payload))
            (root / "timeline").mkdir()
            (root / "timeline" / "2026_14.json").write_text(json.dumps([{"label": "After FP2", "drivers": []}]))
            (root / "schedule").mkdir()
            (root / "schedule" / "2026.json").write_text(json.dumps([{"RoundNumber": 14, "Location": "Madrid"}]))
            try:
                fn()
            finally:
                cache.LOCAL_DATA_DIR = old
                cache._cache.clear()
    run.__name__ = fn.__name__
    return run


def test_shap_explanation_adds_up():
    result = shap_explanation(load_model("finish_position"), row_from_dict(_feature_rows(1)[0]), top_n=40)
    total = result["base_value"] + sum(c["shap"] for c in result["top_contributions"])
    assert abs(total - result["predicted_value"]) < 0.01, "contributions + base must equal the prediction"
    json.dumps(result)


def test_cache_ttl_and_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        old, cache.LOCAL_DATA_DIR = cache.LOCAL_DATA_DIR, Path(tmp)
        cache._cache.clear()
        (Path(tmp) / "latest.json").write_text(json.dumps({"round": 14}))
        assert cache.get_json("latest.json") == {"round": 14}
        (Path(tmp) / "latest.json").write_text(json.dumps({"round": 15}))
        assert cache.get_json("latest.json") == {"round": 14}, "TTL cache should hold the first read"
        try:
            cache.get_json("missing.json")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
        cache.LOCAL_DATA_DIR = old
        cache._cache.clear()


@_with_fixture_dir
def test_forecast_routes():
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["latest_forecast_at"] == "2026-09-11T14:00:00+00:00"
    latest = client.get("/predictions/latest").json()
    assert latest["location"] == "Madrid"
    assert all("probabilities" in d and "quali_odds" in d and "beats_teammate" in d for d in latest["drivers"])
    assert client.get("/predictions/2026/999").status_code == 404
    assert client.get("/predictions/2026/14/timeline").json()[0]["label"] == "After FP2"
    assert client.get("/predictions/2025/1/timeline").json() == []
    assert client.get("/races", params={"season": 2026}).json()[0]["Location"] == "Madrid"
    assert client.get("/races", params={"season": 1999}).status_code == 404


@_with_fixture_dir
def test_explain_route_labels_and_validation():
    body = client.get("/predictions/2026/14/explain", params={"driver": "VER", "target": "finish_position"}).json()
    assert body["driver"] == "VER" and all(c["label"] and "_" not in c["label"] for c in body["top_contributions"])
    assert client.get("/predictions/2026/14/explain", params={"driver": "ALO"}).status_code == 404
    assert client.get("/predictions/2026/14/explain", params={"driver": "VER", "target": "nope"}).status_code == 400


@_with_fixture_dir
def test_strategy_route():
    body = client.get("/strategy/2026/14", params={"sc_lap": 20}).json()
    assert body["strategies"] and body["scenario"]["sc_lap"] == 20
    assert client.get("/strategy/2026/14", params={"sc_lap": 99}).status_code == 400


def test_model_card_and_regulations():
    card = client.get("/model").json()
    assert set(card) == {"qualifying", "finish_position", "quali_delta", "race_time"}
    assert "post_quali" in card["finish_position"]["stages"]
    docs = client.get("/regulations").json()
    assert len(docs) == 8
    hits = client.get("/regulations/search", params={"query": "unsafe release"}).json()
    assert hits and "article" in hits[0]
    assert client.get(f"/regulations/{docs[0]['filename']}").json()["text"]
    assert client.get("/regulations/..%2F..%2Fsecrets").status_code == 404


@_with_fixture_dir
def test_llm_routes_validate_fail_clearly_and_rate_limit():
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    chat._model.cache_clear()
    main._calls.clear()
    try:
        bad = client.post("/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
        assert bad.status_code == 400
        assert client.post("/chat", json={"messages": [{"role": "user", "content": "x" * 5000}]}).status_code == 422
        r = client.post("/chat", json={"messages": [{"role": "user", "content": "Who wins?"}], "context": {"season": 2026, "round": 14}})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert json.loads(r.text.strip().removeprefix("data: "))["type"] == "error"
        assert client.post("/explain", json={"season": 2026, "round": 14, "driver": "VER"}).status_code == 503
        codes = [client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code for _ in range(main.LLM_LIMIT)]
        assert codes[-1] == 429, "the limiter should cut in after LLM_LIMIT calls"
    finally:
        os.environ.update({k: v for k, v in saved.items() if v})
        main._calls.clear()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
