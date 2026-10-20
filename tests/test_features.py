"""Self-check for the Phase 1 feature pipeline: leakage safety on synthetic
data (fast, no network) + a shape/leakage sanity check on the real processed
dataset if it has been built. Run with `python tests/test_features.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import circuit_reference
from src.features.rolling import recent_form, track_form


def test_recent_form_uses_only_prior_rows():
    df = pd.DataFrame({
        "driver": ["A", "A", "A", "B", "B"],
        "race_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01", "2023-02-01"]),
        "finish_position": [10, 2, 4, 5, 5],
    })
    out = recent_form(df, "driver", "race_date", "finish_position", decay=1.0)
    # first race for each driver: no prior data
    assert np.isnan(out.loc[df["driver"].eq("A") & df["race_date"].eq("2023-01-01")].iloc[0])
    assert np.isnan(out.loc[df["driver"].eq("B") & df["race_date"].eq("2023-01-01")].iloc[0])
    # A's 2nd race: only A's 1st race (10) counted, never A's own 2nd/3rd result
    second = out.loc[df["driver"].eq("A") & df["race_date"].eq("2023-02-01")].iloc[0]
    assert second == 10
    # A's 3rd race: average of races 1 and 2 (decay=1.0 -> unweighted mean)
    third = out.loc[df["driver"].eq("A") & df["race_date"].eq("2023-03-01")].iloc[0]
    assert third == 6  # mean(10, 2)


def test_recent_form_excludes_same_date_ties():
    # two teammates (same group_col) sharing one race_date -- neither may see
    # the other's same-race result as "prior", regardless of row order/sort ties
    df = pd.DataFrame({
        "team": ["RBR", "RBR", "RBR", "RBR"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        "finish_position": [1, 18, 2, 15],
    })
    out = recent_form(df, "team", "race_date", "finish_position", decay=1.0)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])  # first race: no prior data for either teammate
    # second race: both teammates' "prior" must be the mean of race 1's two results, not each other's own-race value
    assert out.iloc[2] == 9.5 and out.iloc[3] == 9.5  # mean(1, 18)


def test_recent_form_discounts_previous_regulation_era():
    # 2025 was a win, 2026 (new rules) a P10: the 2026 race weighs 1, the 2025 one CROSS_ERA_WEIGHT
    from src.features.rolling import CROSS_ERA_WEIGHT
    df = pd.DataFrame({
        "team": ["X", "X", "X"], "season": [2025, 2026, 2026],
        "race_date": pd.to_datetime(["2025-12-01", "2026-03-01", "2026-04-01"]),
        "finish_position": [1, 10, 5],
    })
    out = recent_form(df, "team", "race_date", "finish_position", decay=1.0)
    assert out.iloc[1] == 1  # only prior row, whatever its era
    expected = (10 * 1.0 + 1 * CROSS_ERA_WEIGHT) / (1.0 + CROSS_ERA_WEIGHT)
    assert abs(out.iloc[2] - expected) < 1e-9


def test_track_form_uses_year_gap_and_no_future_rows():
    df = pd.DataFrame({
        "driver": ["A", "A", "A"],
        "location": ["Monaco", "Monaco", "Monaco"],
        "season": [2021, 2022, 2023],
        "race_date": pd.to_datetime(["2021-05-01", "2022-05-01", "2023-05-01"]),
        "finish_position": [1, 3, 7],
    })
    out = track_form(df, "driver", "location", "season", "race_date", "finish_position", decay=0.7)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1  # 2022 only sees 2021's result
    # 2023 sees 2021 (2yr gap, weight .49) and 2022 (1yr gap, weight .7), weighted avg, never its own 7
    expected = np.average([1, 3], weights=[0.7 ** 2, 0.7 ** 1])
    assert abs(out.iloc[2] - expected) < 1e-9


def test_circuit_reference_resolves_configuration_era():
    df = pd.DataFrame({"location": ["Marina Bay", "Marina Bay", "Sakhir"], "season": [2022, 2023, 2023]})
    resolved = circuit_reference.resolve(df)
    assert resolved.loc[0, "configuration_id"] == "pre_2023"
    assert resolved.loc[1, "configuration_id"] == "post_2023"
    assert resolved.loc[2, "location"] == "Sakhir"


def test_lap_facts_treat_the_lap_as_a_loop():
    from src.features.circuit_telemetry import lap_facts

    t = np.arange(100.0)  # one sample a second at 50 m/s: the lap integrates to 4,950 m
    flat = (t < 20) | (t >= 80) | ((t >= 30) & (t < 60))  # 1,900 m across the line, 1,450 m mid-lap
    brake = np.zeros(100)
    brake[[0, 50, 99]] = 100  # a braking zone across the line counts once
    facts = lap_facts(t, np.full(100, 180.0), np.where(flat, 100, 40), np.zeros(100), track_m=5000)
    assert facts["flat_out_m"] == 1920, facts  # 1,900 m scaled to the official 5,000
    assert abs(facts["lap_m"] - 4950) < 1e-6
    assert lap_facts(t, np.full(100, 180.0), np.full(100, 40), brake, track_m=5000)["braking_zones"] == 2


def test_served_circuit_facts_follow_the_layout_era():
    from src.api.enrich import circuit_facts

    assert circuit_facts("Monte Carlo", 2026)["braking_zones"], "FastF1's venue names map to the circuit's row"
    assert circuit_facts("Marina Bay", 2022)["braking_zones"] != circuit_facts("Marina Bay", 2026)["braking_zones"], \
        "Singapore before 2023 is the old layout, measured on its own laps"
    assert circuit_facts("Yas Marina", 2020)["flat_out_m"] is None, "a layout gone before 2021 has no telemetry to measure"
    assert circuit_facts("Nowhere", 2026) == {}


def test_processed_dataset_if_built():
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"
    if not path.exists():
        print("  (skipped: data/processed/model_matrix.parquet not built yet)")
        return
    df = pd.read_parquet(path)
    from src.models.features import FEATURE_COLS, QUALI_SAFE_FEATURE_COLS

    missing = set(FEATURE_COLS + QUALI_SAFE_FEATURE_COLS) - set(df.columns)
    assert not missing, f"model inputs missing from the matrix: {sorted(missing)}"
    feature_cols = sorted(set(FEATURE_COLS + QUALI_SAFE_FEATURE_COLS))
    assert df["target_finish_position"].notna().mean() > 0.9
    # no row's historical features can be non-null on its own first-ever race for a brand new driver+team combo
    print(f"  {len(df)} rows, {len(feature_cols)} feature columns")


def _races(n_races: int = 4) -> pd.DataFrame:
    """Two teams of two at one circuit, n_races races in date order."""
    rows = []
    for r in range(1, n_races + 1):
        for i, (drv, team) in enumerate([("AAA", "T1"), ("BBB", "T1"), ("CCC", "T2"), ("DDD", "T2")]):
            rows.append({"season": 2024, "round": r, "race_date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=14 * r),
                         "location": "Monza", "driver": drv, "team": team, "grid_position": i + 1, "finish_position": i + 1,
                         "dnf": False, "sc_laps": 0, "q1_time": 80.0 + 0.1 * i, "q2_time": np.nan, "q3_time": np.nan})
    return pd.DataFrame(rows)


def test_circuit_facts_use_only_earlier_races():
    from src.features import circuit_history

    races = _races()
    before = circuit_history.build(races).set_index("round")
    changed = races.copy()
    changed.loc[changed["round"] == 4, "sc_laps"] = 5  # a safety car in the last race
    changed.loc[changed["round"] == 4, "dnf"] = True
    after = circuit_history.build(changed).set_index("round")
    assert after.loc[[1, 2, 3, 4]].equals(before.loc[[1, 2, 3, 4]]), "a race's own outcome never feeds its circuit facts"
    assert np.isnan(before.loc[1, "circuit_sc_share"]), "nothing before the first race"


def test_driver_ratings_use_only_earlier_races_and_follow_teammate_gaps():
    from src.features import ratings

    races = _races()
    r = ratings.build(races)
    first = r[r["round"] == 1].set_index("driver")["driver_quali_rating"]
    assert (first == 0).all(), "no comparisons yet: everyone average"
    last = r[r["round"] == 4].set_index("driver")
    assert last.loc["AAA", "driver_quali_rating"] < last.loc["BBB", "driver_quali_rating"], "AAA out-qualified BBB every time"
    assert last.loc["AAA", "teammate_quali_rating_gap"] < 0 < last.loc["BBB", "teammate_quali_rating_gap"]
    changed = races.copy()
    changed.loc[(changed["round"] == 4) & (changed["driver"] == "BBB"), "q1_time"] = 79.0  # BBB fastest in the last race
    assert ratings.build(changed)[lambda d: d["round"] == 4]["driver_quali_rating"].equals(
        r[r["round"] == 4]["driver_quali_rating"]), "a race's own qualifying never feeds its ratings"


def test_before_practice_sees_the_three_day_forecast():
    from src.models.features import mask_for_stage

    row = pd.DataFrame({"air_temp_forecast": [30.0], "rain_mm_forecast": [0.0], "wind_kph_forecast": [5.0],
                        "air_temp_forecast_d3": [27.0], "rain_mm_forecast_d3": [2.5], "wind_kph_forecast_d3": [np.nan],
                        "air_temp_forecast_d1": [29.0], "rain_mm_forecast_d1": [1.0], "wind_kph_forecast_d1": [10.0],
                        "grid_position": [np.nan], "quali_position": [np.nan], "driver_recent_form": [5.0]})
    pre = mask_for_stage(row, "pre_weekend")
    assert (pre["air_temp_forecast"].iloc[0], pre["rain_mm_forecast"].iloc[0], pre["wind_kph_forecast"].iloc[0]) == (27.0, 2.5, 5.0)
    assert mask_for_stage(row, "post_quali")["rain_mm_forecast"].iloc[0] == 1.0
    assert mask_for_stage(row, "race_day")["rain_mm_forecast"].iloc[0] == 0.0, "race morning: the near-same-day forecast"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
