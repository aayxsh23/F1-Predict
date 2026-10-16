"""Does averaging XGBoost with LightGBM beat XGBoost alone?

Different tree libraries grow different trees (level-wise vs leaf-wise), so
their errors are not identical, and an average of two decent models often
beats either. LightGBM's settings are searched on the tuning window like
XGBoost's; the mixing weight is chosen on the tuning window; the held-out
races only check it. Gap to the winner is left out (its two-part model is
the separate experiment).
"""
import sys

from src.models import compare
from src.models.train import tune

TARGETS = ("finish_position", "quali_delta", "qualifying")
STAGES = {"qualifying": ["pre_weekend", "post_practice"], "finish_position": ["pre_weekend", "post_practice", "post_quali", "race_day"],
          "quali_delta": ["post_quali", "race_day"]}


def main(fresh: bool = False, n_iter: int = 30) -> dict:
    out = {}
    for target in TARGETS:
        xgb_params = compare.incumbent_params(target)
        incumbent = compare.run(target, "xgb", {"family": "xgb", "params": xgb_params}, use_cache=not fresh)
        rows, cut = compare.target_rows(target)
        tune_rows = rows[compare._race_keys(rows).isin(compare._race_keys(compare.race_order(rows))[:cut])]
        lgbm_params = tune(target, tune_rows, n_iter, family="lgbm")
        lgbm = compare.run(target, "lgbm", {"family": "lgbm", "params": lgbm_params}, use_cache=not fresh)
        w = compare.best_weight(target, incumbent, lgbm)
        spec = {"family": "ensemble", "weights": [round(1 - w, 2), w],
                "members": [{"family": "xgb", "suffix": "xgb", "params": xgb_params},
                            {"family": "lgbm", "suffix": "lgbm", "params": lgbm_params}]}
        alone = compare.score(target, lgbm, incumbent)
        mixed = compare.score(target, compare.combine(incumbent, lgbm, w), incumbent)
        compare.record(target, "lgbm", {"family": "lgbm", "params": lgbm_params}, alone, note="LightGBM alone")
        compare.record(target, "ensemble_xgb_lgbm", spec, mixed, note=f"weight on LightGBM chosen on the tuning window: {w}")
        print(f"\n{target}: LightGBM alone vs XGBoost")
        print(compare.summary(target, alone, STAGES[target]))
        print(f"{target}: ensemble (LightGBM weight {w}) vs XGBoost")
        print(compare.summary(target, mixed, STAGES[target]))
        out[target] = {"weight": w, "spec": spec, "alone": alone, "ensemble": mixed}
    return out


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
