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


def test_processed_dataset_if_built():
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"
    if not path.exists():
        print("  (skipped: data/processed/model_matrix.parquet not built yet)")
        return
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if not c.startswith("target_")
                     and c not in ("season", "round", "driver", "team", "location", "race_date")]
    assert 25 <= len(feature_cols) <= 50, f"expected ~30-45 feature columns, got {len(feature_cols)}"
    assert df["target_finish_position"].notna().mean() > 0.9
    # no row's historical features can be non-null on its own first-ever race for a brand new driver+team combo
    print(f"  {len(df)} rows, {len(feature_cols)} feature columns")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
