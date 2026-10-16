"""Gap to the winner: one regression for every finisher, or a two-part model
(a classifier: lapped or not, then a regression per part)?

Why: held-out after qualifying, lapped cars are 26% of finishers but 45% of
the absolute error, and the single model is biased +0.26% on lead-lap cars
and -1.06% on lapped ones: it averages two populations.
"""
import sys

from src.models import compare

TARGET = "race_time"


def main(fresh: bool = False) -> dict:
    base = compare.incumbent_params(TARGET)
    xgb_params = base.get("params", base) if "family" in base else base
    incumbent = compare.run(TARGET, "xgb", {"family": "xgb", "params": xgb_params}, use_cache=not fresh)
    spec = {"family": "two_part", "split_column": "lapped", "threshold": 0.5,
            "classifier": {"family": "xgb", "classifier": True, "suffix": "xgb_lapped_clf",
                           "params": {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05, "subsample": 0.8}},
            "first": {"family": "xgb", "suffix": "xgb_lead_lap", "params": xgb_params},
            "second": {"family": "xgb", "suffix": "xgb_lapped", "params": xgb_params}}
    results = {}
    for mode in ("pick", "mix"):
        s = {**spec, "mode": mode}
        cand = compare.run(TARGET, f"two_part_{mode}", s, use_cache=not fresh)
        results[mode] = compare.score(TARGET, cand, incumbent)
        compare.record(TARGET, f"two_part_{mode}", s, results[mode], note=__doc__.strip().splitlines()[0])
        print(f"two-part ({mode}) vs single regression (gap to the winner, % of race time):")
        print(compare.summary(TARGET, results[mode], ["pre_weekend", "post_practice", "post_quali", "race_day"]))
    return results


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
