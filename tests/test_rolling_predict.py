"""The saved models answer every point of a race weekend: the same model, fed
progressively more real session data (features.mask_for_stage), should change
its answer and stay plausible. The qualifying model must never see anything
decided in qualifying or on race day. Run with `python tests/test_rolling_predict.py`."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import blend
from src.models.features import QUALI_COLS, QUALI_SAFE_FEATURE_COLS, RACE_STAGES, SPRINT_STAGES, mask_for_stage, stage_of
from src.models.predict import CANONICAL_PRED_COLS, load_model, predict
from src.models.refresh_job import overdue_sessions, session_label
from src.models.train import _train_rows, augment

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"

TARGETS = [  # (target, column, plausible range)
    ("finish_position", "target_finish_position", (-5, 25)),
    ("quali_delta", "target_quali_to_race_delta", (-25, 25)),
    ("race_time", "target_race_gap_pct", (-1, 30)),
]


def _row(col, sprint=False):
    df = pd.read_parquet(DATA_PATH)
    ok = df[col].notna() & df["practice_pace"].notna() & df["grid_position"].notna() & df["quali_position"].notna()
    if sprint:
        ok &= df["sprint_quali_gap_pct"].notna() & df["sprint_finish_position"].notna()
    return df[ok].sample(1, random_state=7)


def test_predictions_respond_to_each_stage():
    for target, col, (lo, hi) in TARGETS:
        row, model = _row(col, sprint=True), load_model(target)
        preds = {s: float(predict(model, mask_for_stage(row, s), target=target).iloc[0]) for s in RACE_STAGES}
        print(f"  [{target}] {preds}")
        assert len({round(v, 4) for v in preds.values()}) > 1, f"[{target}] never changed across stages"
        assert all(lo <= v <= hi for v in preds.values()), f"[{target}] implausible: {preds}"


def test_stage_is_read_off_what_is_filled_in():
    row = _row("target_finish_position", sprint=True)
    assert [stage_of(mask_for_stage(row, s)) for s in RACE_STAGES] == RACE_STAGES
    # an ordinary weekend never claims a sprint stage
    row = _row("target_finish_position").assign(is_sprint_weekend=0.0, sprint_quali_gap_pct=None, sprint_finish_position=None,
                                              sprint_race_pace_pct=None)
    assert {stage_of(mask_for_stage(row, s)) for s in SPRINT_STAGES} == {"post_practice"}


def test_grid_is_the_qualifying_order_until_race_day():
    df = pd.read_parquet(DATA_PATH)
    row = df[df["grid_position"] != df["quali_position"]].dropna(subset=["quali_position", "driver_recent_form"]).iloc[[0]]
    sat, sun = mask_for_stage(row, "post_quali"), mask_for_stage(row, "race_day")
    assert sat["grid_position"].iloc[0] == row["quali_position"].iloc[0] and sat["grid_official"].iloc[0] == 0
    assert sun["grid_position"].iloc[0] == row["grid_position"].iloc[0] and sun["grid_official"].iloc[0] == 1
    assert sat["grid_vs_expected_position"].iloc[0] == row["quali_position"].iloc[0] - row["driver_recent_form"].iloc[0]


def test_position_models_train_on_finishers_only():
    df = pd.read_parquet(DATA_PATH)
    assert df["dnf"].any() and not _train_rows("finish_position", df)["dnf"].any()
    assert _train_rows("race_time", df)["dnf"].any()  # race_time's target is blank for retirements anyway


def test_shipped_blend_weights_are_committed():
    # blend.json was once gitignored, so serving silently ran without blends
    assert set(blend.load_weights()) == set(blend.BLENDS)


def test_blends_are_weighted_averages_and_explanations_stay_exact():
    fin, delta = CANONICAL_PRED_COLS["finish_position"], CANONICAL_PRED_COLS["quali_delta"]
    rows = pd.DataFrame({fin: [4.0, 9.0], delta: [1.0, -2.0], "grid_position": [3.0, None],
                         CANONICAL_PRED_COLS["qualifying"]: [0.3, 0.5]})
    out = blend.apply_rows(rows, "post_quali", {"finish_grid": 0.5})
    assert out[fin].tolist() == [0.5 * 4 + 0.5 * (3 - 1), 9.0]  # no grid: unchanged
    assert blend.apply_rows(rows, "post_practice", {"finish_grid": 0.5})[fin].tolist() == [4.0, 9.0]

    df = pd.read_parquet(DATA_PATH)
    race = df[(df["season"] == 2025) & (df["round"] == 10)]
    race = mask_for_stage(race, "post_quali").reset_index(drop=True)
    w = {"finish_grid": 0.6}
    raw = race.assign(**{c: predict(load_model(t), race, target=t).to_numpy() for t, c in CANONICAL_PRED_COLS.items()})
    shown = blend.apply_rows(raw, "post_quali", w)
    c, base = blend.contributions("finish_position", "post_quali", race.iloc[[0]], race, w)
    assert abs((c.sum() + base) - shown[fin].iloc[0]) < 1e-3


def test_sprint_stages_train_on_sprint_weekends_only():
    df = pd.read_parquet(DATA_PATH)
    aug = augment(df, RACE_STAGES)
    for stage in SPRINT_STAGES:
        assert aug.loc[aug["stage"] == stage, "is_sprint_weekend"].eq(1).all()
    assert (aug["stage"] == "post_sprint").sum() == df["is_sprint_weekend"].eq(1).sum() > 0


def test_a_session_that_ran_but_did_not_load_fails_the_refresh():
    event = pd.Series({"Session1": "Practice 1", "Session1DateUtc": pd.Timestamp("2026-10-09 08:30"),
                       "Session2": "Sprint Qualifying", "Session2DateUtc": pd.Timestamp("2026-10-09 12:30"),
                       "Session3": "Sprint", "Session3DateUtc": pd.Timestamp("2026-10-10 09:00"),
                       "Session4": "Qualifying", "Session4DateUtc": pd.Timestamp("2026-10-10 13:00"),
                       "Session5": "Race", "Session5DateUtc": pd.Timestamp("2026-10-11 12:00")})
    saturday_morning = pd.Timestamp("2026-10-10 10:30")
    assert overdue_sessions(event, [], saturday_morning) == ["FP1", "SQ"]  # the Sprint (09:00) is due from 11:00
    assert overdue_sessions(event, ["FP1", "SQ"], saturday_morning) == []
    assert overdue_sessions(event, ["FP1", "SQ"], pd.Timestamp("2026-10-10 16:00")) == ["S", "Q"]
    assert session_label(["FP1", "SQ", "S"], "post_sprint") == "After the Sprint"
    assert session_label(["FP1", "SQ", "S", "Q"], "post_quali") == "After qualifying"


def test_qualifying_model_is_leakage_safe_and_plausible():
    assert not set(QUALI_COLS) & set(QUALI_SAFE_FEATURE_COLS)
    assert not {"air_temp_forecast", "rain_mm_forecast", "wind_kph_forecast"} & set(QUALI_SAFE_FEATURE_COLS), \
        "Sunday's forecast isn't Saturday's weather"
    pred = float(predict(load_model("qualifying"), _row("target_qualifying_gap_pct"), target="qualifying").iloc[0])
    assert -0.5 <= pred <= 8, f"gap to pole {pred}% of a lap is implausible"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
