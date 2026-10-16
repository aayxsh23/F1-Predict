"""Circuit facts measured from the races already run there, instead of typed
by hand.

The hand-typed table (circuit_reference.csv) disagreed with the data: its
safety-car frequency correlated 0.35 with the share of races that actually had
a safety car or VSC (Spielberg typed 0.3, actually 0.82), its retirement rate
0.41, and its overtaking difficulty 0.24 with how closely results follow the
grid. It was also fixed across years, so backtests saw values typed with
knowledge of later races. The finishing model gave those columns no weight.

For each race, from STRICTLY EARLIER races at the same circuit only:
  circuit_sc_share       share of races with a safety car or VSC
  circuit_dnf_share      share of starters who retired
  circuit_grid_order     mean rank correlation of grid and finish among
                         finishers (1 = the race finishes in grid order)
each shrunk toward the same figure over all earlier races everywhere, with the
weight of PRIOR_RACES pseudo-races, so a circuit with one or two visits isn't
judged on them alone and a brand-new venue gets the field average.
"""
import numpy as np
import pandas as pd

PRIOR_RACES = 2.0
COLUMNS = ("circuit_sc_share", "circuit_dnf_share", "circuit_grid_order")
REPLACES = ("safety_car_frequency", "dnf_rate", "overtaking_difficulty")  # the typed columns they stand in for


def _per_race(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per race: date, location and the three measured facts (NaN
    where the race isn't decided yet, e.g. the live race being forecast)."""
    rows = []
    for (season, rnd), g in raw.groupby(["season", "round"], sort=False):
        fin = g[~g["dnf"].astype(bool) & g["finish_position"].notna() & g["grid_position"].notna()]
        decided = g["finish_position"].notna().any()
        rho = fin["grid_position"].corr(fin["finish_position"], method="spearman") if len(fin) > 3 else np.nan
        sc = g["sc_laps"].iloc[0] if "sc_laps" in g else np.nan
        rows.append({"season": season, "round": rnd, "race_date": g["race_date"].iloc[0], "location": g["location"].iloc[0],
                     "sc": float(sc > 0) if decided and pd.notna(sc) else np.nan,
                     "dnf": g["dnf"].astype(float).mean() if decided else np.nan,
                     "rho": rho if decided else np.nan})
    return pd.DataFrame(rows).sort_values("race_date", kind="stable").reset_index(drop=True)


def build(raw: pd.DataFrame) -> pd.DataFrame:
    """(season, round) -> the three circuit facts, from earlier races only.
    `raw` needs canonical locations (build_dataset applies LOCATION_ALIASES)."""
    races = _per_race(raw)
    out = []
    for i, r in races.iterrows():
        before = races.iloc[:i]
        before = before[before["race_date"] < r["race_date"]]
        here = before[before["location"] == r["location"]]
        row = {"season": r["season"], "round": r["round"]}
        for col, src in zip(COLUMNS, ("sc", "dnf", "rho")):
            everywhere = before[src].mean()
            own = here[src].dropna()
            if np.isnan(everywhere):
                row[col] = np.nan
            else:
                row[col] = (own.sum() + PRIOR_RACES * everywhere) / (len(own) + PRIOR_RACES)
        out.append(row)
    return pd.DataFrame(out)
