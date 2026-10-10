"""Self-check for models/probabilities.py and api/enrich.py -- pure numpy plus
the committed pl_calibration.json, no trained models or FastF1 needed.

Run with `python tests/test_probabilities.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.enrich import head_to_head, with_probabilities
from src.models.probabilities import (field_retirement_rates, load_calibration, q1_cut, quali_probabilities, race_probabilities,
                                      sample_positions)


def test_sampled_orders_are_permutations():
    pos = sample_positions(np.array([3.0, 2.0, 1.0, 0.0]), tau=1.0, p_dnf=np.zeros(4), n=500)
    assert all(sorted(row) == [1, 2, 3, 4] for row in pos), "every draw must be a valid 1..N order, no ties"


def test_win_and_podium_mass_is_conserved():
    pr = race_probabilities([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(pr["win"].sum() - 1.0) < 1e-9
    assert abs(pr["podium"].sum() - 3.0) < 1e-9


def test_better_predicted_car_is_favoured():
    pr = race_probabilities([1.0, 3.0, 5.0, 7.0])
    assert list(np.argsort(-pr["win"])) == [0, 1, 2, 3]
    assert pr["band"][0][0] <= pr["band"][3][0] and pr["band"][0][1] <= pr["band"][3][1]


def test_certain_retirement_never_wins():
    pos = sample_positions(np.array([9.0, 0.0, 0.0]), tau=1.0, p_dnf=np.array([1.0, 0.0, 0.0]), n=500)
    assert (pos[:, 0] == 3).all(), "a car that retires in every draw always classifies last"


def test_every_car_gets_the_field_retirement_rate():
    pr = race_probabilities([1.0, 2.0, 3.0])
    assert (pr["retire"] == load_calibration()["dnf_prior"]).all(), "per-car rates scored worse than one field rate"
    assert (race_probabilities([1.0, 2.0], p_dnf=0.2)["retire"] == 0.2).all()


def test_field_retirement_rate_only_looks_back():
    import pandas as pd

    raw = pd.DataFrame({"season": 2025, "round": [1, 1, 2, 2, 3, 3], "race_date": pd.to_datetime(["2025-03-01"] * 2 + ["2025-03-15"] * 2 + ["2025-03-29"] * 2),
                        "dnf": [True, True, False, False, False, True]})
    r = field_retirement_rates(raw).set_index("round")["p_dnf"]
    assert r[2] == 1.0 and r[3] == 0.5, "each race sees only the races before it"


def test_enrichment_adds_probabilities_and_survives_missing_predictions():
    def car(code, p, team="T1"):
        return {"driver": code, "team": team, "predicted_finish_position": p, "predicted_qualifying_gap_pct": None if p is None else p / 10,
                "feature_row": {"driver_dnf_rate": None, "team_reliability": None}}

    full = {"season": 2026, "round": 1, "generated_at": "t", "known_sessions": {"qualifying": False},
            "drivers": [car("AAA", 1.0), car("BBB", 2.0), car("CCC", 3.0, team="T2")]}
    out = with_probabilities(full)
    assert all("probabilities" in d and "position_band" in d and "quali_odds" in d for d in out["drivers"])
    assert abs(sum(d["probabilities"]["win"] for d in out["drivers"]) - 1.0) < 0.01
    aaa, bbb, ccc = out["drivers"]
    assert abs(aaa["beats_teammate"] + bbb["beats_teammate"] - 1.0) < 0.01 and aaa["beats_teammate"] > 0.5
    assert ccc["beats_teammate"] is None, "no teammate, no fabricated head-to-head"
    assert abs(head_to_head(full, "AAA", "CCC") + head_to_head(full, "CCC", "AAA") - 1.0) < 0.01

    after_quali = {**full, "round": 3, "known_sessions": {"qualifying": True}}
    assert all("quali_odds" not in d for d in with_probabilities(after_quali)["drivers"]), "no qualifying odds once it has happened"

    partial = {**full, "round": 2, "drivers": [car("AAA", 1.0), car("BBB", None), car("CCC", 3.0)]}
    assert all("probabilities" not in d for d in with_probabilities(partial)["drivers"]), "no fabricated odds when a score is missing"


def test_quali_odds_follow_the_knockout_format():
    q = quali_probabilities(np.linspace(0, 2, 22))
    assert abs(q["pole"].sum() - 1) < 1e-9 and abs(q["q3"].sum() - 10) < 1e-9
    assert abs(q["q1_out"].sum() - q1_cut(22)) < 1e-9 and q1_cut(22) == 6 and q1_cut(20) == 5
    assert q["pole"][0] > q["pole"][-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
