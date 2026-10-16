"""Model families behind one interface, so any kind of model can enter the
same walk-forward contest, blends, gates and explanations (scope change
2026-10-11: any number of predictors, any family).

Every family has:
  fit(X, y) -> self         X: the prepared feature frame, y: the target
  predict(X) -> ndarray
  contributions(X) -> (DataFrame, Series)
                            per-input contributions and base value that add
                            up EXACTLY to predict(X): the Why page relies on it
  save(stem) / load(stem)   stem: saved/<target>; files are <stem>_<member>.*

Families here:
  xgb        XGBoost regression (the original four models); TreeSHAP from XGBoost
  lgbm       LightGBM regression; TreeSHAP from LightGBM
  ensemble   a weighted average of members; contributions are the weighted sum
  two_part   a classifier picks one of two regressions per row (e.g. lapped
             cars vs lead-lap cars for the gap to the winner); the shown value
             is that regression's, so its contributions are exact

A model is described by a spec, e.g. {"family": "xgb", "params": {...}} or
{"family": "ensemble", "members": [spec, spec], "weights": [0.6, 0.4]}, stored
in saved/<target>_model.json. A target without that file is a plain XGBoost
model in saved/<target>_xgb.json (the pre-2026-10-11 layout).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

XGB_BASE = {"objective": "reg:absoluteerror", "enable_categorical": True, "random_state": 42, "n_jobs": -1}
LGBM_BASE = {"objective": "l1", "random_state": 42, "n_jobs": -1, "verbose": -1}


class XGB:
    family = "xgb"

    def __init__(self, params: dict | None = None, classifier: bool = False, suffix: str = "xgb"):
        self.params, self.classifier, self.suffix, self.model = dict(params or {}), classifier, suffix, None

    def _estimator(self):
        import xgboost as xgb

        if self.classifier:
            base = {**XGB_BASE, "objective": "binary:logistic"}
            return xgb.XGBClassifier(**base, **self.params)
        return xgb.XGBRegressor(**XGB_BASE, **self.params)

    def fit(self, X, y):
        self.model = self._estimator().fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] if self.classifier else self.model.predict(X)

    def contributions(self, X) -> tuple[pd.DataFrame, pd.Series]:
        import xgboost as xgb

        raw = self.model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
        return pd.DataFrame(raw[:, :-1], columns=X.columns, index=X.index), pd.Series(raw[:, -1], index=X.index)

    def save(self, stem: Path) -> dict:
        self.model.save_model(f"{stem}_{self.suffix}.json")
        return {"family": "xgb", "params": self.params, "classifier": self.classifier, "suffix": self.suffix}

    @classmethod
    def load(cls, stem: Path, spec: dict) -> "XGB":
        m = cls(spec.get("params"), spec.get("classifier", False), spec.get("suffix", "xgb"))
        m.model = m._estimator()
        m.model.load_model(f"{stem}_{m.suffix}.json")
        return m


class LGBM:
    family = "lgbm"

    def __init__(self, params: dict | None = None, suffix: str = "lgbm"):
        self.params, self.suffix, self.booster = dict(params or {}), suffix, None

    def fit(self, X, y):
        import lightgbm as lgb

        self.booster = lgb.LGBMRegressor(**LGBM_BASE, **self.params).fit(X, y).booster_
        return self

    def predict(self, X) -> np.ndarray:
        return self.booster.predict(X)

    def contributions(self, X) -> tuple[pd.DataFrame, pd.Series]:
        raw = self.booster.predict(X, pred_contrib=True)
        return pd.DataFrame(raw[:, :-1], columns=X.columns, index=X.index), pd.Series(raw[:, -1], index=X.index)

    def save(self, stem: Path) -> dict:
        self.booster.save_model(f"{stem}_{self.suffix}.txt")
        return {"family": "lgbm", "params": self.params, "suffix": self.suffix}

    @classmethod
    def load(cls, stem: Path, spec: dict) -> "LGBM":
        import lightgbm as lgb

        m = cls(spec.get("params"), spec.get("suffix", "lgbm"))
        m.booster = lgb.Booster(model_file=f"{stem}_{m.suffix}.txt")
        return m


class Ensemble:
    family = "ensemble"

    def __init__(self, members: list, weights: list[float]):
        assert len(members) == len(weights) and abs(sum(weights) - 1) < 1e-9, "weights must sum to 1"
        self.members, self.weights = members, list(weights)

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return sum(w * m.predict(X) for m, w in zip(self.members, self.weights))

    def contributions(self, X) -> tuple[pd.DataFrame, pd.Series]:
        parts = [(w, *m.contributions(X)) for m, w in zip(self.members, self.weights)]
        return sum(w * c for w, c, _ in parts), sum(w * b for w, _, b in parts)

    def save(self, stem: Path) -> dict:
        return {"family": "ensemble", "weights": self.weights, "members": [m.save(stem) for m in self.members]}

    @classmethod
    def load(cls, stem: Path, spec: dict) -> "Ensemble":
        return cls([load(stem, s) for s in spec["members"]], spec["weights"])


class TwoPart:
    """Two regressions and a classifier saying which one a row belongs to
    (y_split given to fit: 1 for rows of the second). mode "pick": the
    classifier's choice answers (threshold on its probability); mode "mix":
    the two answers weighted by that probability. Either way the shown value
    is a per-row weighted sum of two exactly explained models, so the
    contributions (weighted the same way) still add up exactly."""
    family = "two_part"

    def __init__(self, classifier: XGB, first, second, threshold: float = 0.5, mode: str = "pick"):
        self.classifier, self.first, self.second, self.threshold, self.mode = classifier, first, second, threshold, mode

    def fit(self, X, y, y_split=None):
        y_split = np.asarray(y_split, dtype=bool)
        self.classifier.fit(X, y_split.astype(int))
        self.first.fit(X[~y_split], np.asarray(y)[~y_split])
        self.second.fit(X[y_split], np.asarray(y)[y_split])
        return self

    def chance_second(self, X) -> np.ndarray:
        return self.classifier.predict(X)

    def weight_second(self, X) -> np.ndarray:
        p = self.chance_second(X)
        return p if self.mode == "mix" else (p > self.threshold).astype(float)

    def predict(self, X) -> np.ndarray:
        w = self.weight_second(X)
        return (1 - w) * self.first.predict(X) + w * self.second.predict(X)

    def contributions(self, X) -> tuple[pd.DataFrame, pd.Series]:
        w = self.weight_second(X)
        c1, b1 = self.first.contributions(X)
        c2, b2 = self.second.contributions(X)
        return c1.mul(1 - w, axis=0) + c2.mul(w, axis=0), b1 * (1 - w) + b2 * w

    def save(self, stem: Path) -> dict:
        return {"family": "two_part", "threshold": self.threshold, "mode": self.mode, "classifier": self.classifier.save(stem),
                "first": self.first.save(stem), "second": self.second.save(stem)}

    @classmethod
    def load(cls, stem: Path, spec: dict) -> "TwoPart":
        return cls(load(stem, spec["classifier"]), load(stem, spec["first"]), load(stem, spec["second"]), spec["threshold"],
                   spec.get("mode", "pick"))


FAMILIES = {"xgb": XGB, "lgbm": LGBM, "ensemble": Ensemble, "two_part": TwoPart}


def build(spec: dict):
    """An unfitted model from a spec."""
    f = spec["family"]
    if f == "xgb":
        return XGB(spec.get("params"), spec.get("classifier", False), spec.get("suffix", "xgb"))
    if f == "lgbm":
        return LGBM(spec.get("params"), spec.get("suffix", "lgbm"))
    if f == "ensemble":
        return Ensemble([build(s) for s in spec["members"]], spec["weights"])
    if f == "two_part":
        return TwoPart(build(spec["classifier"]), build(spec["first"]), build(spec["second"]), spec.get("threshold", 0.5),
                       spec.get("mode", "pick"))
    raise ValueError(f"unknown model family {f!r}")


def load(stem: Path, spec: dict):
    return FAMILIES[spec["family"]].load(stem, spec)


def save(model, model_dir: Path, target: str) -> dict:
    spec = model.save(model_dir / target)
    (model_dir / f"{target}_model.json").write_text(json.dumps(spec, indent=2))
    return spec


def load_target(model_dir: Path, target: str):
    """The saved model for a target, whatever its family."""
    manifest = model_dir / f"{target}_model.json"
    spec = json.loads(manifest.read_text()) if manifest.exists() else {"family": "xgb"}
    return load(model_dir / target, spec)
