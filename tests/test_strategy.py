"""Strategy simulator sanity: legal plans only, costs behave like tyres do, and
a safety car makes pitting under it attractive. Uses the committed
src/strategy/params.json. Run with `python tests/test_strategy.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.model import DRY, MIN_STINT, _cost, _plans, simulate


def test_every_plan_is_legal():
    for seq, pits in _plans(50, {c: 50 for c in DRY}):
        bounds = (0, *pits, 50)
        assert len(set(seq)) > 1, "a dry race must use two compounds"
        assert all(b - a >= MIN_STINT for a, b in zip(bounds, bounds[1:]))


def test_more_wear_means_more_stops():
    gentle = {c: {"offset_s": 0.0, "deg_s_per_lap": 0.02} for c in DRY}
    worn = {c: {"offset_s": 0.0, "deg_s_per_lap": 0.3} for c in DRY}
    plans = _plans(50, {c: 50 for c in DRY})

    def best(comp):
        return min(plans, key=lambda p: _cost(comp, 22.0, 50, *p))

    assert len(best(gentle)[1]) == 1 and len(best(worn)[1]) == 2


def test_safety_car_pulls_the_stop_onto_it():
    r = simulate("Baku", 51, sc_lap=12)
    assert r["scenario"]["pits_under_safety_car"] and r["scenario"]["gain_vs_sticking_to_plan_s"] >= 0
    assert abs(sum(s["chance_fastest"] for s in simulate("Baku", 51, top=10)["strategies"]) - 1) < 0.01


def test_unknown_venue_falls_back_to_the_all_circuit_model():
    r = simulate("Nowhere", 56)  # a venue with no stint data
    assert r["strategies"] and r["races_of_data"] == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
