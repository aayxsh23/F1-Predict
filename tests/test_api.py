"""Self-check for the Phase 7 backend: cache TTL/local-read behavior, the
single-row SHAP explanation shape, and the FastAPI routes that only ever
touch cached JSON (fast, no FastF1/LLM). /races (live FastF1 network call),
POST /explain and POST /ask-agent (load the local LLM, seconds+ per call,
same category as test_explainer.py) are deliberately NOT covered here --
exercise those manually per phase7-ui-backend-plan.md's own verification
section instead.

Run with `python tests/test_api.py` (requires the 4 models trained first).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.cache as cache
from src.api.main import app
from src.models.explain import _native, shap_explanation
from src.models.features import FEATURE_COLS, row_from_dict
from src.models.predict import load_model

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"


def _sample_feature_row() -> dict:
    """A real driver's raw feature values, JSON-serializable -- the same
    shape refresh_job.py writes into a prediction payload's feature_row."""
    df = pd.read_parquet(DATA_PATH)
    row = df[df["target_finish_position"].notna()].sample(1, random_state=7).iloc[0]
    return {c: _native(row[c]) for c in FEATURE_COLS}


def test_shap_explanation_shape():
    feature_row = _sample_feature_row()
    model = load_model("finish_position")
    result = shap_explanation(model, row_from_dict(feature_row), target="finish_position", top_n=5)

    assert isinstance(result["predicted_value"], float)
    assert isinstance(result["base_value"], float)
    assert len(result["top_contributions"]) == 5
    contributions_sorted = sorted((abs(c["shap"]) for c in result["top_contributions"]), reverse=True)
    assert contributions_sorted == [abs(c["shap"]) for c in result["top_contributions"]], "contributions not sorted by |shap| desc"
    # json.dumps must not choke on numpy scalar types leaking through
    json.dumps(result)


def test_cache_ttl_and_local_read(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache.LOCAL_DATA_DIR = Path(tmp)
        cache._cache.clear()
        (Path(tmp) / "latest.json").write_text(json.dumps({"season": 2026, "round": 14}))

        first = cache.get_json("latest.json")
        assert first == {"season": 2026, "round": 14}

        # rewrite the file; a cache hit within TTL must still return the old value
        (Path(tmp) / "latest.json").write_text(json.dumps({"season": 2026, "round": 15}))
        assert cache.get_json("latest.json") == first, "TTL cache did not hold a cached value"

        try:
            cache.get_json("missing.json")
            assert False, "expected FileNotFoundError for a missing file"
        except FileNotFoundError:
            pass


def test_predictions_endpoints_with_fixture_data():
    import tempfile

    feature_row = _sample_feature_row()
    fixture = {
        "season": 2026, "round": 14, "location": "Madrid",
        "generated_at": "2026-09-11T14:00:00Z",
        "known_sessions": {"practice": True, "qualifying": False, "grid": False, "compound": False},
        "drivers": [{
            "driver": "VER", "team": "Red Bull",
            "grid_position": None, "quali_gap_to_pole": None, "practice_pace": 1.02,
            "predicted_qualifying_gap": 0.8, "predicted_finish_position": 3.4,
            "predicted_quali_to_race_delta": -0.5, "predicted_race_time_gap": 12.1,
            "feature_row": feature_row,
        }],
    }

    with tempfile.TemporaryDirectory() as tmp:
        cache.LOCAL_DATA_DIR = Path(tmp)
        cache._cache.clear()
        (Path(tmp) / "latest.json").write_text(json.dumps(fixture))
        (Path(tmp) / "2026_14.json").write_text(json.dumps(fixture))

        client = TestClient(app)

        assert client.get("/health").json() == {"status": "ok"}

        assert client.get("/predictions/latest").json()["location"] == "Madrid"
        assert client.get("/predictions/2026/14").json()["drivers"][0]["driver"] == "VER"
        assert client.get("/predictions/2026/999").status_code == 404

        resp = client.get("/predictions/2026/14/explain", params={"driver": "VER", "target": "finish_position"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["driver"] == "VER" and body["target"] == "finish_position"
        assert len(body["top_contributions"]) > 0

        assert client.get("/predictions/2026/14/explain", params={"driver": "HAM", "target": "finish_position"}).status_code == 404
        assert client.get("/predictions/2026/14/explain", params={"driver": "VER", "target": "not_a_target"}).status_code == 400


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
