"""Driver qualifying ratings from teammate comparisons: how fast each driver
is relative to the cars around him, independent of the car.

Teammates share a car, so the gap between them is the cleanest measure of
the drivers. Every past teammate gap is fitted at once as
rating[A] - rating[B] (ridge regression, recent races weighted more), so a
driver's record carries over when he changes team: if A has beaten B and B
has a long record against others, A's rating says something about A against
a driver he has never been paired with.

The gap is measured in the last qualifying session both drivers ran (Q3 if
both reached it, else Q2, else Q1), as % of the faster lap: comparing one
driver's Q3 lap with the other's Q2 lap mixes in track evolution (0.88% vs
0.57% standard deviation of teammate gaps). Gaps over SPLIT_MAX are a crash
or a failure, not pace, and are left out.

Each race's rating uses strictly earlier races only, so it is a fair input
before qualifying. 2026-10-11 research: with the qualifying model's own team
average kept, swapping its within-team split for the ratings' raised pole
calls from 29% to 42% before practice (src/models/blend.py, "quali_split").
"""
import numpy as np
import pandas as pd

DECAY = 0.97  # weight per race of age (~a season to halve)
RIDGE = 3.0  # pull toward zero: a driver with few comparisons stays near average
SPLIT_MAX = 2.0  # % gap inside one session above which it's not pace


def session_splits(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per teammate pair per race: (race_date, a, b, split = a's lap
    over b's in the last session both ran, %). raw needs lineage team names."""
    rows = []
    cols = ["q3_time", "q2_time", "q1_time"]
    q = raw.dropna(subset=["q1_time"], how="all") if "q1_time" in raw else raw.iloc[0:0]
    for (_, _, team), g in q.groupby(["season", "round", "team"], sort=False):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        for c in cols:
            if pd.notna(a.get(c)) and pd.notna(b.get(c)):
                split = (a[c] - b[c]) / min(a[c], b[c]) * 100
                if abs(split) < SPLIT_MAX:
                    rows.append({"race_date": a["race_date"], "a": a["driver"], "b": b["driver"], "split": split})
                break
    return pd.DataFrame(rows, columns=["race_date", "a", "b", "split"])


def build(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, round, driver): driver_quali_rating (% of a lap, negative
    = faster than average) and its teammate gap, from earlier races only."""
    splits = session_splits(raw)
    drivers = sorted(set(raw["driver"]))
    idx = {d: i for i, d in enumerate(drivers)}
    dates = sorted(raw["race_date"].unique())
    race_no = {d: i for i, d in enumerate(dates)}
    s_race = splits["race_date"].map(race_no).to_numpy()
    X = np.zeros((len(splits), len(drivers)))
    if len(splits):
        X[np.arange(len(splits)), splits["a"].map(idx).to_numpy()] = 1
        X[np.arange(len(splits)), splits["b"].map(idx).to_numpy()] = -1
    y = splits["split"].to_numpy()

    ratings = {}
    for date in dates:
        r = race_no[date]
        past = s_race < r
        if not past.any():
            ratings[date] = np.zeros(len(drivers))
            continue
        w = DECAY ** (r - s_race[past])
        Xp = X[past]
        A = Xp.T @ (Xp * w[:, None]) + RIDGE * np.eye(len(drivers))
        ratings[date] = np.linalg.solve(A, Xp.T @ (w * y[past]))

    out = raw[["season", "round", "driver", "team", "race_date"]].copy()
    out["driver_quali_rating"] = [ratings[d][idx[drv]] for d, drv in zip(out["race_date"], out["driver"])]
    team_mean = out.groupby(["season", "round", "team"])["driver_quali_rating"].transform("mean")
    n_team = out.groupby(["season", "round", "team"])["driver"].transform("count")
    # gap to the teammate: own rating minus the teammate's (2 cars: twice the gap to the pair's mean)
    out["teammate_quali_rating_gap"] = np.where(n_team > 1, (out["driver_quali_rating"] - team_mean) * n_team / (n_team - 1), np.nan)
    return out[["season", "round", "driver", "driver_quali_rating", "teammate_quali_rating_gap"]]
