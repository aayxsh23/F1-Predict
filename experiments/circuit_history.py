"""Circuit facts measured from earlier races (src/features/circuit_history.py)
instead of the hand-typed table: better or worse for each model?

The typed safety-car frequency, retirement rate and overtaking difficulty
disagreed with the data (correlations 0.35, 0.41 and 0.24) and were fixed
across years. Same XGBoost settings, same walk-forward; the only change is
the three columns' values.
"""
import sys

from src.features import circuit_history
from src.features.build_dataset import build
from src.models import compare

TARGETS = ("finish_position", "quali_delta", "qualifying", "race_time")
STAGES = {"qualifying": ["pre_weekend", "post_practice"], "quali_delta": ["post_quali", "race_day"]}


def main(fresh: bool = False) -> dict:
    # run before the per-model column lists existed: "measured" means the three
    # typed columns' values replaced by the measured ones, for every model
    typed = build()
    measured = typed.copy()
    for col, rep in zip(circuit_history.COLUMNS, circuit_history.REPLACES):
        measured[rep] = measured[col]
    out = {}
    for target in TARGETS:
        base = compare.incumbent_params(target)
        params = base["params"] if "family" in base else base
        if "family" in base and base["family"] != "xgb":  # composite incumbents: compare with their XGBoost settings
            params = compare.incumbent_params(target).get("first", {}).get("params") or params
        spec = {"family": "xgb", "params": params}
        before = compare.run(target, "xgb_typed_circuits", spec, df=typed, use_cache=not fresh)
        after = compare.run(target, "xgb_measured_circuits", spec, df=measured, use_cache=not fresh)
        result = compare.score(target, after, before)
        compare.record(target, "xgb_measured_circuits", {**spec, "features": "measured circuit facts"}, result,
                       note="incumbent here = the same XGBoost on the hand-typed circuit facts")
        print(f"\n{target}: measured vs typed circuit facts")
        print(compare.summary(target, result, STAGES.get(target, ["pre_weekend", "post_practice", "post_quali", "race_day"])))
        out[target] = result
    return out


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)

# Outcome (2026-10-11): finishing position and places gained better on the
# tuning races (-0.005 to -0.012 and -0.03 to -0.04), held-out level; qualifying
# and gap to the winner worse on both (gap to the winner before practice: held-out
# interval above zero). So models/features.py gives the measured facts to the
# first two only.
