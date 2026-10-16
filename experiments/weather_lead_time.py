"""Forecasts matched to the moment: before practice the model gets the race
forecast as it stood 3 days out, on Saturday 1 day out (where Open-Meteo
archived them: rain and wind from March 2024, temperature from 2022), instead
of the nearly same-day archive forecast for every stage.

This is a train/serve fix (a live Thursday forecast IS 3 days out), so the
question is only "does it hurt?". The tuning window has almost no 2024+ races,
so it can't choose here; the held-out races (mostly 2024+) are the check.
Same XGBoost settings; gap to the winner, finishing position and places gained
(qualifying reads no weather).
"""
import sys

from src.features.build_dataset import WEATHER_LEAD_COLS, build
from src.models import compare

TARGETS = ("finish_position", "quali_delta", "race_time")


def main(fresh: bool = False) -> dict:
    with_leads = build()
    without = with_leads.drop(columns=WEATHER_LEAD_COLS)
    out = {}
    for target in TARGETS:
        base = compare.incumbent_params(target)
        spec = {"family": "xgb", "params": base["params"] if "family" in base else base}
        before = compare.run(target, "xgb_archive_weather", spec, df=without, use_cache=not fresh)
        after = compare.run(target, "xgb_lead_weather", spec, df=with_leads, use_cache=not fresh)
        result = compare.score(target, after, before)
        compare.record(target, "xgb_lead_weather", {**spec, "features": "stage-matched forecast lead times"}, result,
                       note="incumbent here = the same XGBoost with the archive forecast at every stage")
        print(f"\n{target}: lead-matched vs archive forecasts")
        print(compare.summary(target, result, ["pre_weekend", "post_practice", "post_quali", "race_day"]
                              if target != "quali_delta" else ["post_quali", "race_day"]))
        out[target] = result
    return out


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
