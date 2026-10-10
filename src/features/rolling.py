"""Leakage-safe recency-weighted rolling stats: every value is computed only
from rows with a strictly earlier date in the same group ("prior" is a date
comparison, not sort position, so teammates in the same race never count as
each other's history)."""
import numpy as np
import pandas as pd

# First season of each technical-regulation era. Form from a previous era
# says little about a new car: counting those races at a tenth of their
# weight cut held-out finish error after qualifying 3.287 -> 3.263 places and
# pre-practice qualifying error 0.735 -> 0.715% (weights 0.05 to 0.6 were
# tested; 0.05-0.15 were equally good).
ERA_STARTS = (2017, 2022, 2026)  # wider cars (2017), ground effect (2022), 2026 rules
CROSS_ERA_WEIGHT = 0.1


def era(season) -> np.ndarray:
    s = np.asarray(season)
    return np.select([s >= y for y in ERA_STARTS[::-1]], ERA_STARTS[::-1], default=ERA_STARTS[0])


def recent_form(df: pd.DataFrame, group_col: str, date_col: str, value_col: str, decay: float = 0.85) -> pd.Series:
    """Exponentially weighted mean of value_col over strictly prior rows in
    each group: the latest prior row weighs decay**0, the one before decay**1,
    and rows from an earlier regulation era a further CROSS_ERA_WEIGHT."""
    # ponytail: O(n^2) per group; fine at a few hundred rows/group, move to pandas ewm if seasons scale way up
    df = df.sort_values(date_col, kind="stable")
    eras = pd.Series(era(df["season"]) if "season" in df else 0, index=df.index)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group_col, sort=False):
        vals = g[value_col].to_numpy(dtype=float)
        dates = g[date_col].to_numpy()
        e = eras[g.index].to_numpy()
        idx = g.index.to_numpy()
        for i in range(1, len(vals)):
            prior_mask = dates[:i] < dates[i]
            prior = vals[:i][prior_mask][::-1]
            weights = decay ** np.arange(len(prior)) * np.where(e[:i][prior_mask][::-1] == e[i], 1.0, CROSS_ERA_WEIGHT)
            valid = ~np.isnan(prior)
            if valid.any():
                out.loc[idx[i]] = np.average(prior[valid], weights=weights[valid])
    return out


def track_form(df: pd.DataFrame, driver_col: str, location_col: str, season_col: str,
               date_col: str, value_col: str, decay: float = 0.7) -> pd.Series:
    """A driver's history at one circuit across seasons, weighted by year gap
    (last year ~0.7, two years back ~0.49)."""
    df = df.sort_values(date_col, kind="stable")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby([driver_col, location_col], sort=False):
        seasons = g[season_col].to_numpy(dtype=float)
        vals = g[value_col].to_numpy(dtype=float)
        dates = g[date_col].to_numpy()
        idx = g.index.to_numpy()
        for i in range(1, len(vals)):
            prior_mask = dates[:i] < dates[i]
            prior_vals = vals[:i][prior_mask]
            gap = seasons[i] - seasons[:i][prior_mask]
            valid = ~np.isnan(prior_vals)
            if valid.any():
                out.loc[idx[i]] = np.average(prior_vals[valid], weights=(decay ** gap)[valid])
    return out
