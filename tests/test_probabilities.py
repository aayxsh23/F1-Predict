"""Self-check for models/probabilities.py and api/enrich.py -- pure numpy plus
the committed pl_calibration.json, no trained models or FastF1 needed.

Run with `python tests/test_probabilities.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.enrich import head_to_head, with_probabilities
from src.models.probabilities import dnf_probability, fit_tau, q1_cut, quali_probabilities, race_probabilities, sample_positions


def test_sampled_orders_are_permutations():
    pos = sample_positions(np.array([3.0, 2.0, 1.0, 0.0]), tau=1.0, p_dnf=np.zeros(4), n=500)
    assert all(sorted(row) == [1, 2, 3, 4] for row in pos), "every draw must be a valid 1..N order, no ties"


def test_win_and_podium_mass_is_conserved():
    pr = race_probabilities([1.0, 2.0, 3.0, 4.0, 5.0], [0.1] * 5, [0.9] * 5)
    assert abs(pr["win"].sum() - 1.0) < 1e-9
    assert abs(pr["podium"].sum() - 3.0) < 1e-9


def test_better_predicted_car_is_favoured():
    pr = race_probabilities([1.0, 3.0, 5.0, 7.0], [0.05] * 4, [0.95] * 4)
    assert list(np.argsort(-pr["win"])) == [0, 1, 2, 3]
    assert pr["band"][0][0] <= pr["band"][3][0] and pr["band"][0][1] <= pr["band"][3][1]


def test_certain_retirement_never_wins():
    pos = sample_positions(np.array([9.0, 0.0, 0.0]), tau=1.0, p_dnf=np.array([1.0, 0.0, 0.0]), n=500)
    assert (pos[:, 0] == 3).all(), "a car that retires in every draw always classifies last"


def test_dnf_probability_fallbacks():
    assert dnf_probability([np.nan], [np.nan], prior=0.15)[0] == 0.15, "no rates at all falls back to the prior"
    assert abs(dnf_probability([0.1], [np.nan], prior=0.15)[0] - 0.1) < 1e-9, "one missing rate uses the other alone"
    assert dnf_probability([0.9], [0.0], prior=0.15)[0] == 0.5, "clipped so no car is written off"


def test_fit_tau_recovers_the_noise_it_was_generated_with():
    rng = np.random.default_rng(1)
    true_tau, races = 2.0, []
    for _ in range(300):
        score = rng.normal(size=12) * 3
        order = np.argsort(-(score / true_tau + rng.gumbel(size=12)))  # a Plackett-Luce draw
        races.append(score[order])
    assert abs(fit_tau(races) - true_tau) < 0.4


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
