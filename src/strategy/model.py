"""Race strategy: which tyres, how many stops, when to pit, and what a safety
car on a given lap changes.

Fit (`python -m src.strategy.model`), per circuit, from real race laps
(data/raw/laps/), written to src/strategy/params.json:
  - tyre model: green-flag lap time = driver's race baseline + fuel/track
    trend x lap + compound offset + compound degradation x tyre age,
    least squares with a separate baseline per driver per race, so a slow
    car on hards doesn't make hards look slow
  - pit loss: in-lap + out-lap minus two normal laps, median over real stops
  - longest realistic stint per compound: 90th percentile of real stints
  - safety-car chance: share of past races here with a safety car or VSC
Each circuit's tyre numbers are shrunk toward the all-circuit fit in
proportion to how few laps it has on that compound, so one race at a new
venue (or a compound barely used somewhere) can't produce a soft tyre that is
slower than the medium.

Simulate: every legal 1- and 2-stop plan (a dry race must use two different
compounds) is costed in closed form. The fuel/track trend cancels out (every
plan drives the same laps), so a plan's cost is sum over stints of
(laps x compound offset + degradation x laps(laps+1)/2) + stops x pit loss.
Then 2,000 races with random safety cars: a plan whose pit window contains
the safety car pits under it at half the usual loss. "Chance fastest" is how
often each plan wins those races.

Known limits: degradation is linear (no cliff), tyres start new, traffic and
track position are ignored. This compares plans; it doesn't predict a gap.
"""
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.models.catalog import LOCATION_ALIASES

ROOT = Path(__file__).resolve().parents[2]
LAPS_DIR = ROOT / "data" / "raw" / "laps"
PARAMS_PATH = Path(__file__).with_name("params.json")
DRY = ["SOFT", "MEDIUM", "HARD"]
SC_PIT_FACTOR = 0.5  # pitting under a safety car costs about half the usual time (field slowed)
SHRINK_LAPS = 400  # a compound needs about this many laps here before its own fit outweighs the all-circuit one
MIN_STINT = 5
N_SIMS = 2000
MIN_SEASON = 2022  # 18-inch tyres and the current compound names; older laps are a different tyre
WINDOW_S = 2.0  # a pit lap is "in the window" if it costs at most this much more than the best lap


def _canon(location: str) -> str:
    return LOCATION_ALIASES.get(location, location)


def _load_laps():
    import pandas as pd

    frames = []
    for f in sorted(LAPS_DIR.glob("*.parquet")):
        season, rnd, location = f.stem.split("_", 2)
        if int(season) < MIN_SEASON:
            continue
        frames.append(pd.read_parquet(f).assign(race=f"{season}_{rnd}", location=_canon(location)))
    return pd.concat(frames, ignore_index=True)


def _green(laps):
    ok = laps[(laps["track_status"] == "1") & ~laps["pit_in"] & ~laps["pit_out"] & (laps["lap"] > 1)
              & laps["lap_time_s"].notna() & laps["compound"].isin(DRY) & laps["tyre_life"].notna()]
    med = ok.groupby(["race", "driver"])["lap_time_s"].transform("median")
    return ok[ok["lap_time_s"] < med * 1.07]


def _tyre_fit(g) -> dict:
    """Least squares on laps demeaned within each driver's race."""
    import pandas as pd

    X = pd.DataFrame({"lap": g["lap"]}, index=g.index)
    for c in DRY:
        on = (g["compound"] == c).astype(float)
        X[f"age_{c}"] = g["tyre_life"] * on
        if c != "MEDIUM":  # medium is the reference compound
            X[f"is_{c}"] = on
    y = g["lap_time_s"]
    grp = [g["race"], g["driver"]]
    Xd, yd = X - X.groupby(grp).transform("mean"), y - y.groupby(grp).transform("mean")
    beta = dict(zip(X.columns, np.linalg.lstsq(Xd.to_numpy(), yd.to_numpy(), rcond=None)[0]))
    return {c: {"offset_s": float(beta.get(f"is_{c}", 0.0)),
                "deg_s_per_lap": float(max(beta[f"age_{c}"], 0.005)),  # tyres don't get faster with age
                "laps": int((g["compound"] == c).sum())}
            for c in DRY}


def _shrink(own: dict, glob: dict) -> dict:
    out = {}
    for c in DRY:
        w = own[c]["laps"] / (own[c]["laps"] + SHRINK_LAPS)
        out[c] = {k: w * own[c][k] + (1 - w) * glob[c][k] for k in ("offset_s", "deg_s_per_lap")} | {"laps": own[c]["laps"]}
    return out


def _max_stints(laps) -> dict:
    stints = laps[laps["compound"].isin(DRY)].groupby(["race", "driver", "stint", "compound"]).size().reset_index(name="n")
    return {c: int(stints.loc[stints["compound"] == c, "n"].quantile(0.9)) for c in DRY if (stints["compound"] == c).any()}


def _pit_loss(laps) -> float:
    base = _green(laps).groupby(["race", "driver"])["lap_time_s"].median()
    nxt = laps.assign(lap=laps["lap"] - 1)[["race", "driver", "lap", "lap_time_s", "track_status", "pit_out"]]
    stops = laps[laps["pit_in"] & (laps["track_status"] == "1")].merge(nxt, on=["race", "driver", "lap"], suffixes=("", "_out"))
    stops = stops[stops["pit_out_out"] & (stops["track_status_out"] == "1")]
    loss = stops["lap_time_s"] + stops["lap_time_s_out"] - 2 * stops.set_index(["race", "driver"]).index.map(base)
    loss = loss[(loss > 5) & (loss < 60)]  # drop drive-throughs, damage, penalties
    return float(loss.median()) if len(loss) else np.nan


def fit_params() -> dict:
    from src.features.build_dataset import load_raw
    from src.features.circuit_reference import load as load_circuits
    from src.models.estimates import safety_car_probability

    laps = _load_laps()
    raw = load_raw()
    circuits = load_circuits().drop_duplicates("location").set_index("location")
    green = _green(laps)
    params = {"global": {"compounds": _tyre_fit(green), "max_stint": _max_stints(laps), "pit_loss_s": _pit_loss(laps),
                         "sc_probability": safety_car_probability(raw, "")}}  # no circuit given -> all-circuit rate
    for loc, g in laps.groupby("location"):
        own = green[green["location"] == loc]
        fallback_loss = float(circuits.loc[loc, "pit_lane_loss_time"]) if loc in circuits.index else params["global"]["pit_loss_s"]
        pit = _pit_loss(g)
        races = int(g["race"].nunique())
        params[loc] = {
            "compounds": _shrink(_tyre_fit(own), params["global"]["compounds"]) if len(own) else params["global"]["compounds"],
            "max_stint": {**params["global"]["max_stint"], **(_max_stints(g) if races >= 2 else {})},
            "pit_loss_s": pit if not np.isnan(pit) else fallback_loss,
            "sc_probability": safety_car_probability(raw, loc),
            "races": races, "green_laps": int(len(own)),
        }
    for loc, c in circuits.iterrows():  # venues with no lap history yet (a new circuit): all-circuit tyres, own pit lane
        params.setdefault(loc, {**params["global"], "pit_loss_s": float(c["pit_lane_loss_time"]),
                                "sc_probability": safety_car_probability(raw, loc), "races": 0, "green_laps": 0})
    PARAMS_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    load_params.cache_clear()
    return params


@lru_cache(maxsize=1)
def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text(encoding="utf-8"))


def _stint_cost(comp: dict, compound: str, n) -> np.ndarray:
    c = comp[compound]
    n = np.asarray(n, dtype=float)
    return n * c["offset_s"] + c["deg_s_per_lap"] * n * (n + 1) / 2


def _plans(laps: int, max_stint: dict) -> list[tuple[tuple, tuple]]:
    """(compound sequence, pit laps) for every legal 1- and 2-stop plan, every stint MIN_STINT+ laps."""
    out = []
    for seq in itertools.product(DRY, repeat=2):
        if seq[0] != seq[1]:
            out += [(seq, (p,)) for p in range(MIN_STINT, laps - MIN_STINT + 1)
                    if p <= max_stint[seq[0]] and laps - p <= max_stint[seq[1]]]
    for seq in itertools.product(DRY, repeat=3):
        if len(set(seq)) > 1:
            for p1 in range(MIN_STINT, laps - 2 * MIN_STINT + 1):
                if p1 > max_stint[seq[0]]:
                    break
                for p2 in range(p1 + MIN_STINT, laps - MIN_STINT + 1):
                    if p2 - p1 <= max_stint[seq[1]] and laps - p2 <= max_stint[seq[2]]:
                        out.append((seq, (p1, p2)))
    return out


def _cost(comp, pit_loss, laps, seq, pits, sc_lap=None) -> float:
    bounds = (0, *pits, laps)
    t = sum(float(_stint_cost(comp, c, b - a)) for c, a, b in zip(seq, bounds, bounds[1:]))
    return t + sum(pit_loss * (SC_PIT_FACTOR if p == sc_lap else 1.0) for p in pits)


def _label(seq) -> str:
    return f"{len(seq) - 1}-stop " + " → ".join(c.title() for c in seq)


def simulate(location: str, laps: int, sc_lap: int | None = None, top: int = 4, seed: int = 0) -> dict:
    """Best plans for a race at `location` over `laps` laps. `sc_lap`: a safety
    car on that lap (the scenario the app lets you inject)."""
    all_params = load_params()
    p = all_params.get(_canon(location)) or all_params["global"]
    comp, pit_loss = p["compounds"], p["pit_loss_s"]
    max_stint = {c: min(p["max_stint"].get(c, laps), laps) for c in DRY}
    plans = _plans(laps, max_stint)
    if not plans:  # stint caps too tight for this distance: relax them rather than return nothing
        plans = _plans(laps, {c: laps for c in DRY})

    # best pit laps per set of compounds. Order doesn't change the cost here
    # (no fuel-tyre interaction), so Soft→Medium and Medium→Soft are one plan,
    # shown softer-first as teams usually run it.
    best: dict[tuple, tuple[float, tuple, tuple]] = {}
    for seq, pits in plans:
        cost = _cost(comp, pit_loss, laps, seq, pits)
        key = tuple(sorted(seq, key=DRY.index))
        if key not in best or cost < best[key][0] - 1e-6:
            best[key] = (cost, seq, pits)
    ranked = [(seq, (cost, pits)) for cost, seq, pits in sorted(best.values(), key=lambda v: v[0])[: max(top, 6)]]

    # pit windows: laps where moving that one stop costs at most WINDOW_S more
    def windows(seq, pits):
        base = _cost(comp, pit_loss, laps, seq, pits)
        out = []
        for i in range(len(pits)):
            ok = [lap for lap in range(1, laps) if (lap > (pits[i - 1] if i else 0)) and (lap < (pits[i + 1] if i + 1 < len(pits) else laps))
                  and _cost(comp, pit_loss, laps, seq, pits[:i] + (lap,) + pits[i + 1:]) <= base + WINDOW_S]
            out.append([min(ok), max(ok)] if ok else [pits[i], pits[i]])
        return out

    wins = {seq: windows(seq, pits) for seq, (_, pits) in ranked}

    # random safety cars: a plan pits under one if it falls inside a pit window
    rng = np.random.default_rng(seed)
    hazard = 1 - (1 - p.get("sc_probability", 0.5)) ** (1 / laps)
    sc_laps = np.where(rng.random((N_SIMS, laps)) < hazard, np.arange(1, laps + 1), laps + 99).min(axis=1)
    times = np.zeros((N_SIMS, len(ranked)))
    for j, (seq, (cost, pits)) in enumerate(ranked):
        for s in np.unique(sc_laps):
            hit = sc_laps == s
            moved = tuple(s if lo <= s <= hi else pit for pit, (lo, hi) in zip(pits, wins[seq]))
            times[hit, j] = _cost(comp, pit_loss, laps, seq, moved, sc_lap=s) if moved != pits else cost
    p_best = np.bincount(times.argmin(axis=1), minlength=len(ranked)) / N_SIMS

    fastest = ranked[0][1][0]
    strategies = []
    for j, (seq, (cost, pits)) in enumerate(ranked[:top]):
        bounds = (0, *pits, laps)
        strategies.append({
            "name": _label(seq), "stops": len(pits), "pit_laps": list(pits), "pit_windows": wins[seq],
            "stints": [{"compound": c, "from_lap": a + 1, "to_lap": b, "laps": b - a} for c, a, b in zip(seq, bounds, bounds[1:])],
            "time_vs_best_s": round(cost - fastest, 1), "chance_fastest": round(float(p_best[j]), 3),
        })

    scenario = None
    if sc_lap is not None and 1 <= sc_lap < laps:
        options = [(_cost(comp, pit_loss, laps, seq, pits, sc_lap=sc_lap), seq, pits) for seq, pits in plans]
        cost, seq, pits = min(options, key=lambda o: o[0])
        planned = strategies[0]
        planned_cost = _cost(comp, pit_loss, laps, ranked[0][0], tuple(planned["pit_laps"]), sc_lap=sc_lap)
        bounds = (0, *pits, laps)
        scenario = {
            "sc_lap": sc_lap, "best": _label(seq), "pit_laps": list(pits),
            "stints": [{"compound": c, "from_lap": a + 1, "to_lap": b, "laps": b - a} for c, a, b in zip(seq, bounds, bounds[1:])],
            "pits_under_safety_car": sc_lap in pits,
            "gain_vs_sticking_to_plan_s": round(planned_cost - cost, 1),
        }

    return {
        "circuit": _canon(location), "laps": laps, "pit_loss_s": round(pit_loss, 1),
        "safety_car_probability": round(p.get("sc_probability", np.nan), 3),
        "races_of_data": p.get("races", 0),
        "compounds": {c: {"pace_vs_medium_s": round(comp[c]["offset_s"], 3), "deg_s_per_lap": round(comp[c]["deg_s_per_lap"], 3),
                          "max_stint_laps": max_stint[c]} for c in DRY},
        "strategies": strategies, "scenario": scenario,
    }


if __name__ == "__main__":
    params = fit_params()
    print(f"fitted {len(params) - 1} circuits -> {PARAMS_PATH}")
    for loc, laps in (("Monza", 53), ("Monaco", 78), ("Sakhir", 57), ("Baku", 51)):
        r = simulate(loc, laps)
        print(f"\n{loc}: pit loss {r['pit_loss_s']}s, SC {r['safety_car_probability']:.0%}")
        for s in r["strategies"]:
            print(f"  {s['name']:28s} pits {s['pit_laps']} window {s['pit_windows']}  +{s['time_vs_best_s']}s  fastest {s['chance_fastest']:.0%}")
