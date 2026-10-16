"""The prediction the app shows is a blend of models and anchors, and its Why
breakdown must still add up exactly to it, for every target and stage, with
whatever families the targets use (ensembles, two-part) and the teammate
split blend, whose breakdown mixes in the teammate's. Uses the committed
models and blend weights on a real recent race from the feature matrix.
Run with `python -m pytest tests/test_blend_exact.py`."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.catalog import CANONICAL_PRED_COLS  # noqa: E402
from src.models.explain import shown_explanation  # noqa: E402
from src.models.features import mask_for_stage  # noqa: E402
from src.models.predict import predict_all  # noqa: E402

MATRIX = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"


def _race(stage: str) -> pd.DataFrame:
    m = pd.read_parquet(MATRIX)
    last = m[m["season"] == m["season"].max()]
    race = last[last["round"] == last["round"].max()].reset_index(drop=True)
    return predict_all(mask_for_stage(race, stage))


@pytest.mark.parametrize("stage", ["pre_weekend", "post_practice", "post_quali"])
def test_every_shown_prediction_is_an_exact_sum(stage):
    if not MATRIX.exists():
        pytest.skip("build the feature matrix first")
    p = _race(stage)
    targets = ["finish_position", "quali_delta", "race_time"] + (["qualifying"] if stage != "post_quali" else [])
    for target in targets:
        col = CANONICAL_PRED_COLS[target]
        for i in range(len(p)):
            exp = shown_explanation(target, stage, p.iloc[[i]], p, top_n=200)
            total = exp["base_value"] + sum(c["shap"] for c in exp["top_contributions"])
            assert abs(total - p.iloc[i][col]) < 2e-3, f"{target} {stage} {p.iloc[i]['driver']}: {total} vs {p.iloc[i][col]}"


def test_the_teammate_split_keeps_each_teams_average():
    if not MATRIX.exists():
        pytest.skip("build the feature matrix first")
    raw = _race("post_practice")
    from src.models.predict import predict_all as unblended

    no_split = unblended(mask_for_stage(pd.read_parquet(MATRIX).pipe(
        lambda m: m[(m["season"] == raw["season"].iloc[0]) & (m["round"] == raw["round"].iloc[0])]).reset_index(drop=True), "post_practice"),
        blend=False)
    col = CANONICAL_PRED_COLS["qualifying"]
    both = raw[["driver", "team", col]].merge(no_split[["driver", col]], on="driver", suffixes=("", "_model"))
    teams = both.groupby("team").filter(lambda g: len(g) == 2).groupby("team")
    assert (teams[col].mean() - teams[f"{col}_model"].mean()).abs().max() < 1e-3, "the split moves teammates apart, not the team"
