"""Leakage-safe recency-weighted rolling stats: every value here is computed
only from rows strictly before the current one within its group."""
import numpy as np
import pandas as pd


def recent_form(df: pd.DataFrame, group_col: str, date_col: str, value_col: str, decay: float = 0.85) -> pd.Series:
    """Exponential-decay weighted mean of value_col over strictly prior rows
    (date_col strictly less than the current row's, not just earlier in sort
    order) in each group_col group — the most recent prior row gets weight
    decay**0, the one before decay**1, etc. Rows tie on date_col constantly
    (every driver in a race shares one race_date), and sort_values() isn't
    stable, so filtering by position (vals[:i]) let a tied same-race row look
    "prior" depending on how the sort happened to break the tie. Comparing
    date values instead of array position makes "prior" mean prior."""
    # ponytail: O(n^2) per group; fine at a few hundred rows/group, move to pandas ewm if seasons scale way up
    df = df.sort_values(date_col, kind="stable")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group_col, sort=False):
        vals = g[value_col].to_numpy(dtype=float)
        dates = g[date_col].to_numpy()
        idx = g.index.to_numpy()
        for i in range(1, len(vals)):
            prior = vals[:i][dates[:i] < dates[i]][::-1]
            weights = decay ** np.arange(len(prior))
            valid = ~np.isnan(prior)
            if valid.any():
                out.loc[idx[i]] = np.average(prior[valid], weights=weights[valid])
    return out


def track_form(df: pd.DataFrame, driver_col: str, location_col: str, season_col: str,
                date_col: str, value_col: str, decay: float = 0.7) -> pd.Series:
    """Same as recent_form but for a driver's history at one circuit across
    seasons, weighted by year gap (last year ~0.7, two years back ~0.49)."""
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
