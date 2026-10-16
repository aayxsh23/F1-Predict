"""The contract every model family behind src/models/families.py keeps: the
contributions add up exactly to the prediction (the Why page relies on it),
a saved model predicts the same after loading, rows with not-yet-known
columns blanked are accepted, and fitting twice gives the same model.
Synthetic data, no repo models. Run with `python -m pytest tests/test_families.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import families  # noqa: E402

FAST = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2}
FAST_LGBM = {"n_estimators": 40, "num_leaves": 8, "learning_rate": 0.2}


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.2, "b"] = np.nan  # a stage where "b" isn't known yet
    split = (X["a"] + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    y = 2 * X["a"] - X["c"] + 5 * split + rng.normal(scale=0.3, size=n)
    return X, y, split


SPECS = {
    "xgb": {"family": "xgb", "params": FAST},
    "lgbm": {"family": "lgbm", "params": FAST_LGBM},
    "ensemble": {"family": "ensemble", "weights": [0.4, 0.6],
                 "members": [{"family": "xgb", "suffix": "xgb", "params": FAST}, {"family": "lgbm", "suffix": "lgbm", "params": FAST_LGBM}]},
    "two_part_pick": {"family": "two_part", "mode": "pick", "split_column": "s", "threshold": 0.5,
                      "classifier": {"family": "xgb", "classifier": True, "suffix": "xgb_clf", "params": FAST},
                      "first": {"family": "xgb", "suffix": "xgb_a", "params": FAST}, "second": {"family": "lgbm", "suffix": "lgbm_b", "params": FAST_LGBM}},
}
SPECS["two_part_mix"] = {**SPECS["two_part_pick"], "mode": "mix"}


def _fit(spec):
    X, y, split = _data()
    m = families.build(spec)
    return (m.fit(X, y, y_split=split) if isinstance(m, families.TwoPart) else m.fit(X, y)), X


@pytest.mark.parametrize("name", list(SPECS))
def test_contributions_add_up_to_the_prediction(name):
    m, X = _fit(SPECS[name])
    c, b = m.contributions(X)
    assert np.allclose(c.sum(axis=1) + b, m.predict(X), atol=1e-3), "contributions + base must equal the prediction"
    assert list(c.columns) == list(X.columns)


@pytest.mark.parametrize("name", list(SPECS))
def test_saved_model_predicts_the_same_after_loading(name, tmp_path):
    m, X = _fit(SPECS[name])
    spec = families.save(m, tmp_path, "target")
    loaded = families.load_target(tmp_path, "target")
    assert spec["family"] == SPECS[name]["family"]
    assert np.allclose(loaded.predict(X), m.predict(X), atol=1e-5)


@pytest.mark.parametrize("name", list(SPECS))
def test_blank_columns_are_accepted_and_fitting_is_repeatable(name):
    m1, X = _fit(SPECS[name])
    m2, _ = _fit(SPECS[name])
    blank = X.assign(b=np.nan, c=np.nan)
    assert np.isfinite(m1.predict(blank)).all()
    assert np.allclose(m1.predict(X), m2.predict(X)), "same data, same seed: same model"


def test_a_target_without_a_manifest_is_the_old_xgboost_file(tmp_path):
    m, X = _fit(SPECS["xgb"])
    m.model.save_model(str(tmp_path / "old_xgb.json"))  # the pre-2026-10-11 layout
    assert np.allclose(families.load_target(tmp_path, "old").predict(X), m.predict(X))
