"""What-if tables for the app (phase 10 item 6.4): each driver's predicted
finish from every grid slot. The serverless API can't run the models, so the
refresh job computes these with them and ships them in the forecast; the API
turns a choice into odds with the race sampler (GET /whatif).

Before qualifying the grid isn't known, so a grid what-if gives the other
qualifying inputs the qualifying model's own predictions (who qualifies
where, gaps to pole and to the teammate). Only the chosen driver's slot moves;
the rest of the field keeps its forecast.

No rain what-if: the models learned almost nothing from the rain forecast
(11 races since 2018 forecast more than 1 mm). At Austin 2026 no driver's
prediction moved until 20 mm, and then every car's by the same 0.06 places, so
a "make it rain" switch would only suggest that rain doesn't matter.
"""
import numpy as np
import pandas as pd

from src.models.predict import predict_all


def _with_predicted_qualifying(rows: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if out["quali_gap_pct"].notna().any():
        return out
    gap = out["driver"].map(pred.set_index("driver")["predicted_qualifying_gap_pct"]).astype(float)
    out["quali_gap_pct"] = gap - gap.min()
    out["quali_position"] = gap.rank(method="first")
    mate = out.groupby("team")["quali_gap_pct"].transform("sum") - out["quali_gap_pct"]  # two-car teams
    out["teammate_quali_gap"] = out["quali_gap_pct"] - mate
    return out


def grid_table(rows: pd.DataFrame, pred: pd.DataFrame) -> dict[str, list[float]]:
    """rows: the race's feature rows at their stage; pred: predict_all(rows).
    {driver: [predicted finish starting P1, P2, ... P<field size>]}."""
    base = _with_predicted_qualifying(rows, pred)
    n = len(base)
    moved = base.loc[base.index.repeat(n)].reset_index(drop=True)
    moved["grid_position"] = moved["quali_position"] = np.tile(np.arange(1, n + 1), n).astype(float)
    moved["grid_vs_expected_position"] = moved["grid_position"] - moved["driver_recent_form"]
    moved["grid_official"] = 0.0
    finish = predict_all(moved)["predicted_finish_position"].to_numpy().reshape(n, n)
    return {d: [round(float(f), 3) for f in finish[i]] for i, d in enumerate(base["driver"])}
