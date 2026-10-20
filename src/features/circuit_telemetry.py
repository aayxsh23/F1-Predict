"""Circuit facts measured from F1's own data, for the race page's circuit
panel (2026-10-19).

The hand-typed columns of circuit_reference.csv were rough. Austin's
"longest straight 1,200 m" is the corner-to-corner length of the back
straight, and its "6 braking zones" are 9 on the pole lap. They stay as they
are anyway: the models were trained on them, and changing a model input
needs a retrain and a held-out check. The columns written here are only read
by people (`src/api/enrich.py` serves them), never by a model.

For every layout era raced since 2021 (OpenF1 has car telemetry from 2023,
FastF1's mirror has 2021 and 2022), from the best laps of the five fastest drivers in the latest dry qualifying
in that era (the median, because one lap misleads: Silverstone 2025's pole
lap had a lift that cut its longest flat-out run from about 1,600 m to 980):
  measured_flat_out_m      longest stretch at full throttle with the brake off
  measured_braking_zones   separate brake applications on the lap
  measured_from            the session, e.g. "2025 Qualifying"
and MultiViewer's pit loss for that layout (built from F1 timing data, and
blank for venues it has no map of yet):
  measured_pit_loss_s      time a green-flag stop costs against staying out

The latest session wins, so the figures describe the cars racing now: 2026
cars lift to recharge before some braking zones, which shortens flat-out runs
a little (Monza 1,330 m in 2025, 1,220 m in 2026). Run again after each
venue's qualifying, a layout change or a new venue:
    python -m src.features.circuit_telemetry
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.openf1 import API as OPENF1
from src.data.openf1 import fetch as _get
from src.features.circuit_reference import _CSV, load
from src.models.catalog import LOCATION_ALIASES

MULTIVIEWER = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
FIRST_TELEMETRY_YEAR = 2023
FULL_THROTTLE = 98  # pedal %; OpenF1 reads 99-100 along a straight
# a lap's integrated distance against the official length: sampling costs
# about 1%, a different layout (or a broken lap) far more
LAYOUT_TOLERANCE = 0.04
TOP_LAPS, MIN_LAPS = 5, 3
COLUMNS = ["measured_flat_out_m", "measured_braking_zones", "measured_pit_loss_s", "measured_from"]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def lap_facts(t_s, speed_kph, throttle, brake, track_m: float) -> dict:
    """Longest flat-out stretch and braking zones on one lap of car data.

    OpenF1 samples about four times a second, so distance comes from
    integrating speed; that reads about 1% short, so it is scaled to the
    official length. A lap is a loop: a stretch that crosses the line counts
    once."""
    t, v = np.asarray(t_s, float), np.asarray(speed_kph, float) / 3.6
    d = np.concatenate([[0.0], np.cumsum(np.diff(t) * (v[1:] + v[:-1]) / 2)])
    lap_m = float(d[-1])
    d = d * track_m / lap_m

    flat = (np.asarray(throttle) >= FULL_THROTTLE) & (np.asarray(brake) == 0)
    runs = _runs(flat)
    lengths = [d[e - 1] - d[s] for s, e in runs]
    if len(runs) > 1 and flat[0] and flat[-1]:
        lengths[0] += lengths.pop()

    braking = np.asarray(brake) > 0
    zones = len(_runs(braking)) - int(len(_runs(braking)) > 1 and braking[0] and braking[-1])
    return {"flat_out_m": int(round(max(lengths, default=0), -1)), "braking_zones": zones, "lap_m": lap_m}


def _one_lap(session_key: int, lap: pd.Series, track_m: float) -> dict | None:
    t0 = pd.Timestamp(lap["date_start"])
    t1 = t0 + pd.Timedelta(seconds=float(lap["lap_duration"]))
    car = pd.DataFrame(_get(OPENF1 + "car_data", session_key=session_key, driver_number=int(lap["driver_number"]),
                            **{"date>": t0.isoformat(), "date<": t1.isoformat()}) or [])
    if len(car) < 100:
        return None
    car["t"] = (pd.to_datetime(car["date"], format="ISO8601") - t0).dt.total_seconds()
    car = car.sort_values("t")
    facts = lap_facts(car["t"], car["speed"], car["throttle"], car["brake"], track_m)
    return facts if abs(facts["lap_m"] / track_m - 1) <= LAYOUT_TOLERANCE else None


def _measure(session: pd.Series, track_m: float) -> dict | None:
    """Median facts over the five fastest drivers' best laps; None if the
    session was wet or too few laps measure as this layout."""
    key = session["session_key"]
    laps = pd.DataFrame(_get(OPENF1 + "laps", session_key=key) or [])
    if laps.empty:
        return None
    best = laps.dropna(subset=["lap_duration", "date_start"]).sort_values("lap_duration").drop_duplicates("driver_number").head(TOP_LAPS)
    stints = pd.DataFrame(_get(OPENF1 + "stints", session_key=key) or [])
    if len(stints):
        on = stints.merge(best[["driver_number", "lap_number"]], on="driver_number")
        on = on[(on["lap_start"] <= on["lap_number"]) & (on["lap_end"].fillna(999) >= on["lap_number"])]
        if on["compound"].isin(["INTERMEDIATE", "WET"]).any():
            return None
    return _median([f for f in (_one_lap(key, lap, track_m) for _, lap in best.iterrows()) if f], f"{session['year']} {session['location']}")


def _median(facts: list[dict], what: str) -> dict | None:
    if len(facts) < MIN_LAPS:
        print(f"  {what}: only {len(facts)} laps measure as this layout, skipped")
        return None
    return {"flat_out_m": int(round(np.median([f["flat_out_m"] for f in facts]), -1)),
            "braking_zones": int(round(np.median([f["braking_zones"] for f in facts])))}


def _fastf1_measure(season: int, round_number: int, track_m: float) -> tuple[dict | None, int | None]:
    """The same from FastF1, for layouts last raced before OpenF1 began. F1's
    archive refuses requests now, but FastF1's mirror still has 2021 and 2022
    (nothing older), so the layouts gone by 2021 stay unmeasured."""
    import fastf1

    from src.data.fastf1_client import enable_cache

    enable_cache()
    fastf1.set_log_level("ERROR")
    try:
        s = fastf1.get_session(season, round_number, "Q")
        s.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = s.laps.pick_quicklaps(1.07)
    except Exception:
        return None, None
    best = laps.sort_values("LapTime").drop_duplicates("Driver").head(TOP_LAPS)
    if best["Compound"].isin(["INTERMEDIATE", "WET"]).any():
        return None, None
    facts = []
    for _, lap in best.iterlaps():
        car = lap.get_car_data()
        f = lap_facts(car["Time"].dt.total_seconds(), car["Speed"], car["Throttle"], car["Brake"].astype(int), track_m)
        if abs(f["lap_m"] / track_m - 1) <= LAYOUT_TOLERANCE:
            facts.append(f)
    return _median(facts, f"{season} round {round_number}"), int(s.session_info["Meeting"]["Circuit"]["Key"])


def _pit_loss(circuit_key: int, year: int, valid_from: int, valid_to: float) -> float:
    """MultiViewer's green-flag pit loss, if its map of the circuit is of this
    layout era (it can answer a 2022 request with its 2023 map)."""
    mv = _get(MULTIVIEWER.format(key=circuit_key, year=year))
    if not mv or not valid_from <= mv.get("year", 0) <= valid_to or not mv.get("pitLoss"):
        return np.nan
    return float(mv["pitLoss"]["normal"])


def build(now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    sessions = pd.DataFrame([s for y in range(FIRST_TELEMETRY_YEAR, now.year + 1)
                             for s in _get(OPENF1 + "sessions", year=y, session_name="Qualifying") or []])
    ended = pd.to_datetime(sessions["date_end"], utc=True) < now
    sessions = sessions[ended & ~sessions["is_cancelled"].fillna(False).astype(bool)]
    sessions = sessions.assign(location=sessions["location"].replace(LOCATION_ALIASES)).sort_values("date_end", ascending=False)

    from src.features.build_dataset import load_raw  # load_raw knows the venues that share a name (Sakhir 2020)

    raw = load_raw()[["season", "round", "location"]].drop_duplicates()
    raw = raw.assign(location=raw["location"].replace(LOCATION_ALIASES)).sort_values(["season", "round"], ascending=False)

    circuits = load()
    for col in COLUMNS:
        circuits[col] = pd.Series(dtype="object" if col == "measured_from" else "float")
    for i, era in circuits.iterrows():
        hi = era["valid_to_year"] if pd.notna(era["valid_to_year"]) else 9999
        track_m = era["track_length_km"] * 1000
        found = None
        here = sessions[(sessions["location"] == era["location"]) & sessions["year"].between(era["valid_from_year"], hi)]
        for _, s in here.iterrows():
            if facts := _measure(s, track_m):
                found = facts, s["circuit_key"], s["year"]
                break
        if found is None:  # a layout gone before OpenF1: FastF1's mirror
            older = raw[(raw["location"] == era["location"]) & raw["season"].between(max(era["valid_from_year"], 2021), min(hi, 2022))]
            for _, r in older.iterrows():
                facts, key = _fastf1_measure(int(r["season"]), int(r["round"]), track_m)
                if facts:
                    found = facts, key, int(r["season"])
                    break
        if found:
            facts, key, year = found
            circuits.loc[i, COLUMNS] = [facts["flat_out_m"], facts["braking_zones"],
                                        _pit_loss(key, year, era["valid_from_year"], hi), f"{year} Qualifying"]
            print(f"{era['location']:18s} {era['configuration_id']:9s} {circuits.loc[i, COLUMNS].tolist()}")
    for col in ("measured_flat_out_m", "measured_braking_zones"):
        circuits[col] = circuits[col].astype("Int64")
    circuits.to_csv(_CSV, index=False, lineterminator="\n")  # pandas would write CRLF on Windows
    return circuits


if __name__ == "__main__":
    build()
