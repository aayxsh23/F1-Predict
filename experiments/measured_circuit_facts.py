"""Circuit facts measured from F1 telemetry and timing (src/features/
circuit_telemetry.py) instead of the hand-typed table: better or worse for
each model?

Swapped together, as one decision: the longest straight (typed, roughly
corner to corner) for the longest flat-out run, the typed braking-zone count
for the brake applications counted on the fastest qualifying laps, and the
typed pit-lane loss for MultiViewer's. Layouts gone before 2021 have no
measurement (FastF1's mirror has nothing older), so their rows read missing;
only qualifying trains on seasons that early. Each target's shipped model
(family and settings), same walk-forward; the only change is the three
columns' values. Run after
`python -m src.features.circuit_telemetry`.
"""
import sys

from src.features import circuit_reference
from src.features.build_dataset import build
from src.models import compare

TARGETS = ("finish_position", "quali_delta", "qualifying", "race_time")
STAGES = {"qualifying": ["pre_weekend", "post_practice"], "quali_delta": ["post_quali", "race_day"]}
SWAP = {"longest_straight_m": "measured_flat_out_m", "braking_zone_count": "measured_braking_zones",
        "pit_lane_loss_time": "measured_pit_loss_s"}


def measured_matrix(typed):
    """The matrix with the three typed columns' values replaced by the measured ones."""
    circuits = circuit_reference.load().drop(columns=list(SWAP)).rename(columns={v: k for k, v in SWAP.items()})
    m = circuit_reference.resolve(typed[["location", "season"]], circuits=circuits)
    out = typed.copy()
    for col in SWAP:
        out[col] = m[col].astype(float).to_numpy()
    return out


def main(fresh: bool = False) -> dict:
    typed = build()
    measured = measured_matrix(typed)
    print("rows without a measurement:", {c: int(measured[c].isna().sum()) for c in SWAP}, "of", len(measured))
    out = {}
    for target in TARGETS:
        spec = compare.incumbent_params(target)  # the shipped family and settings (ensemble, two-part...)
        before = compare.run(target, "shipped_typed_physical", spec, df=typed, use_cache=not fresh)
        after = compare.run(target, "shipped_measured_physical", spec, df=measured, use_cache=not fresh)
        result = compare.score(target, after, before)
        compare.record(target, "shipped_measured_physical", {"spec": spec, "features": "measured flat-out, braking zones, pit loss"},
                       result, note="incumbent here = the same shipped model on the hand-typed physical circuit facts")
        print(f"\n{target}: measured vs typed physical circuit facts")
        print(compare.summary(target, result, STAGES.get(target, ["pre_weekend", "post_practice", "post_quali", "race_day"])))
        out[target] = result
    return out


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)

# Outcome (2026-10-20): worse on the tuning races for every model and stage
# (finishing before practice 2.620 vs 2.609, places gained after qualifying
# 2.515 vs 2.509, qualifying after practice 0.737 vs 0.733, gap to the winner
# after qualifying 0.521 vs 0.517). Held-out: qualifying slightly better
# (0.717 vs 0.722, interval -0.011 to +0.001), the rest slightly worse, no
# interval clear of zero. Not shipped: the models keep the typed columns, and
# the measured ones are for the race page only. Likely why: with about 30
# circuits these columns mostly tell the trees which circuit it is, and the
# typed values group circuits in ways that happen to suit the targets; the
# measured ones also describe 2025-26 cars, not the 2018-2023 cars the tuning
# races were run with, and are missing for 340 early rows.
