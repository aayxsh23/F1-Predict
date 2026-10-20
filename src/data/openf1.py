"""F1 session data from OpenF1 (api.openf1.org), shaped like the FastF1
sessions src/data/ingest.py reads, for when FastF1 can't load one.

Since 2026-10-19 F1's live-timing archive (livetiming.formula1.com/static,
FastF1's source) answers 401 to every request, and FastF1's mirror 404s
anything it hadn't already copied, so no new session loads through FastF1.
OpenF1 republishes the same F1 timing feed, free once a session has ended.
`fastf1_client.load_session` falls back to `load_session` here.

Only what ingest.py reads is built:
  laps      Driver, Team, LapNumber, LapTime, Time (lap end, from the first
            lap's start), PitInTime/PitOutTime (set on in- and out-laps; only
            their presence is read), Stint, Compound, TyreLife, TrackStatus,
            Deleted, Position (races: order at each lap's end)
  results   classification, grid with penalties, points, gap to the winner,
            Q1-Q3 (Sprint Qualifying too)
  weather   AirTemp, TrackTemp, Rainfall, WindSpeed, Humidity
TrackStatus is rebuilt from race-control messages with FastF1's codes (1
green, 2 yellow, 4 safety car, 5 red, 6 VSC, 7 VSC ending). Ingest only asks
whether a lap was all green, had a safety car or VSC, or a red flag.

Checked by ingesting four races both ways (2025 Austin, 2026 Baku, Kuala
Lumpur, Singapore): grid, classification, points, qualifying, practice and
long-run pace match; safety-car lap counts are within one lap; up to 3% of
laps differ on "all green" (yellow-flag timing). Known differences: OpenF1
times lap 1 about 0.4 s longer, starts a new stint when tyres are changed
under a red flag, and doesn't mark a pit-lane starter's first lap as an
out-lap.
"""
import re
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

API = "https://api.openf1.org/v1/"
# OpenF1 has no round numbers: a session is found by its name, near the race date
NAMES = {"FP1": ("Practice 1",), "FP2": ("Practice 2",), "FP3": ("Practice 3",), "Q": ("Qualifying",),
         "SQ": ("Sprint Qualifying", "Sprint Shootout"), "S": ("Sprint",), "R": ("Race",)}
GRID_FROM = {"R": "Q", "S": "SQ"}  # OpenF1 files the starting grid under the session that set it
LAP_COLS = ["Driver", "Team", "LapNumber", "LapTime", "Time", "PitInTime", "PitOutTime", "Stint", "Compound",
            "TyreLife", "TrackStatus", "Deleted", "Position"]
RESULT_COLS = ["Abbreviation", "DriverNumber", "TeamName", "Position", "GridPosition", "Points", "Status", "Laps",
               "Time", "Q1", "Q2", "Q3"]
# "CAR 81 (PIA) TIME 1:53.171 DELETED - DOUBLE YELLOW AT TURN 5 LAP 3 20:34:06". Matched on the
# time: the message's lap number can be one more than OpenF1's (HAD, 2026 Singapore Sprint Qualifying)
DELETED = re.compile(r"CAR (\d+) \S+ TIME (\d+):(\d+\.\d+) DELETED")


def fetch(url: str, **params):
    """JSON from a GET; None when OpenF1 says "No results found" (404)."""
    for _ in range(6):
        # MultiViewer (circuit_telemetry.py) refuses requests' default user agent
        r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "f1-race-predictor"})
        if r.status_code != 429:  # the free tier allows a few requests a second
            break
        time.sleep(10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def get(endpoint: str, **params) -> pd.DataFrame:
    return pd.DataFrame(fetch(API + endpoint, **params) or [])


@lru_cache(maxsize=4)
def _sessions(season: int) -> pd.DataFrame:
    s = get("sessions", year=season)
    return s.assign(start=pd.to_datetime(s["date_start"], utc=True).dt.tz_localize(None))


def session_key(season: int, round_number: int, code: str) -> int | None:
    """The session of that name in the four days up to FastF1's race date."""
    from src.data.fastf1_client import event_schedule

    sched = event_schedule(season)
    race_day = pd.Timestamp(sched.loc[sched["RoundNumber"] == round_number, "EventDate"].iloc[0])
    s = _sessions(season)
    if s.empty:  # OpenF1 starts in 2023
        return None
    hit = s[s["session_name"].isin(NAMES[code]) & s["start"].between(race_day - pd.Timedelta(days=4), race_day + pd.Timedelta(days=1))]
    return int(hit.iloc[0]["session_key"]) if len(hit) else None


def track_status(rc: pd.DataFrame) -> pd.DataFrame:
    """(date, status) at each change of FastF1's TrackStatus, rebuilt from race control."""
    events, yellow, neutral = [], set(), None
    for _, m in rc.sort_values("date").iterrows():
        msg, flag, scope = str(m["message"]), m.get("flag"), m.get("scope")
        if m["category"] == "SafetyCar":
            if "ENDING" in msg and ("VSC" in msg or "VIRTUAL" in msg):
                neutral = "7"
            elif "DEPLOYED" in msg:
                neutral = "6" if ("VSC" in msg or "VIRTUAL" in msg) else "4"
            else:
                continue  # "SAFETY CAR IN THIS LAP": still out until the track goes green
        elif flag == "RED":
            neutral = "5"
        elif flag in ("GREEN", "CLEAR") and scope == "Track":
            neutral, yellow = None, set()
        elif flag in ("YELLOW", "DOUBLE YELLOW") and scope == "Sector":
            yellow.add(m["sector"])
        elif flag == "CLEAR" and scope == "Sector":
            yellow.discard(m["sector"])
        else:
            continue
        events.append((m["date"], neutral or ("2" if yellow else "1")))
    return pd.DataFrame(events, columns=["date", "status"])


def lap_status(starts: pd.Series, ends: pd.Series, events: pd.DataFrame) -> list[str]:
    """Each lap's statuses in order, as FastF1 writes them ("1", "16", "671")."""
    t = pd.to_datetime(events["date"], format="ISO8601", utc=True).to_numpy() if len(events) else np.array([], "datetime64[ns]")
    st = events["status"].to_numpy() if len(events) else np.array([])
    out = []
    for a, b in zip(starts.to_numpy(), ends.to_numpy()):
        if pd.isna(a) or pd.isna(b):
            out.append("1")
            continue
        i, j = np.searchsorted(t, a, "right"), np.searchsorted(t, b, "left")
        seq = ([st[i - 1]] if i else ["1"]) + list(st[i:j])
        out.append("".join(s for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]))
    return out


def _laps(key: int, drivers: pd.DataFrame, rc: pd.DataFrame, race: bool) -> pd.DataFrame:
    raw = get("laps", session_key=key)
    if raw.empty:
        return pd.DataFrame(columns=LAP_COLS)
    raw = raw.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    start = pd.to_datetime(raw["date_start"], format="ISO8601", utc=True)
    duration = pd.to_timedelta(raw["lap_duration"], unit="s")
    t0 = start.min()
    # a lap with no time (a car stopping on track) ends where its next lap starts, or a typical lap later
    end = (start + duration).fillna(start.groupby(raw["driver_number"]).shift(-1)).fillna(start + duration.median())
    no_time = pd.Series(pd.NaT, index=raw.index, dtype="timedelta64[ns]")
    pit_out = raw["is_pit_out_lap"].fillna(False).astype(bool)
    # the lap before an out-lap ended in the pits. In a race OpenF1's pit list also names the in-lap
    # (a car retiring in the pits has no out-lap); in practice and qualifying it names the out-lap
    pit_in = pit_out.groupby(raw["driver_number"]).shift(-1, fill_value=False).astype(bool)
    pits = get("pit", session_key=key) if race else pd.DataFrame()
    if len(pits):
        pit_in |= pd.MultiIndex.from_frame(raw[["driver_number", "lap_number"]]).isin(list(zip(pits["driver_number"], pits["lap_number"])))
    deleted = {(int(n), round(int(m) * 60 + float(s), 3))
               for n, m, s in rc["message"].astype(str).str.extract(DELETED).dropna().to_numpy()} if len(rc) else set()
    laps = pd.DataFrame({
        "Driver": raw["driver_number"].map(drivers["name_acronym"]), "Team": raw["driver_number"].map(drivers["team_name"]),
        "LapNumber": raw["lap_number"].astype(float), "LapTime": duration, "Time": start + duration - t0,
        "PitInTime": no_time.mask(pit_in, pd.Timedelta(0)),
        "PitOutTime": no_time.mask(pit_out, pd.Timedelta(0)),
        "TrackStatus": lap_status(start, end, track_status(rc)) if len(rc) else "1",
        "Deleted": [(n, round(t, 3)) in deleted for n, t in zip(raw["driver_number"], raw["lap_duration"].fillna(0))],
    })
    stints = get("stints", session_key=key)
    if len(stints):
        on = raw[["driver_number", "lap_number"]].reset_index().merge(stints, on="driver_number")
        on = on[(on["lap_number"] >= on["lap_start"]) & (on["lap_number"] <= on["lap_end"].fillna(10_000))]
        on = on.drop_duplicates("index").set_index("index")
        laps["Stint"] = on["stint_number"].astype(float)
        laps["Compound"] = on["compound"]
        laps["TyreLife"] = (on["tyre_age_at_start"] + on["lap_number"] - on["lap_start"] + 1).astype(float)
    if race:  # running order at each lap's end: whoever completed that lap sooner is ahead
        laps["Position"] = laps.groupby("LapNumber")["Time"].rank(method="first")
    return laps.reindex(columns=LAP_COLS)


def _results(season: int, round_number: int, code: str, key: int, drivers: pd.DataFrame) -> pd.DataFrame:
    res = get("session_result", session_key=key)
    if res.empty:
        return pd.DataFrame(columns=RESULT_COLS)
    num, laps = res["driver_number"], res["number_of_laps"].astype(float)
    out = pd.DataFrame({"Abbreviation": num.map(drivers["name_acronym"]), "DriverNumber": num.astype(str),
                        "TeamName": num.map(drivers["team_name"]),
                        "Position": pd.to_numeric(res["position"], errors="coerce")})  # "RT" for a car with no time
    # cars not classified come after the ones that are, most laps first, as FastF1 orders them
    unplaced = laps[out["Position"].isna()].sort_values(ascending=False, kind="stable").index
    out.loc[unplaced, "Position"] = out["Position"].max() + 1 + np.arange(len(unplaced))
    if code in ("Q", "SQ"):  # duration is [Q1, Q2, Q3], shorter for cars knocked out
        segments = res["duration"].apply(lambda d: (list(d) if isinstance(d, list) else []) + [None] * 3)
        for i, col in enumerate(("Q1", "Q2", "Q3")):
            out[col] = pd.to_timedelta(pd.to_numeric(segments.str[i]), unit="s")
        return out.reindex(columns=RESULT_COLS)
    if code not in GRID_FROM:  # practice: ingest reads only its laps
        return out.reindex(columns=RESULT_COLS)

    out["Laps"], out["Points"] = laps, res.get("points")
    out["Status"] = np.select([res["dsq"].astype(bool), res["dns"].astype(bool), res["dnf"].astype(bool), laps < laps.max()],
                              ["Disqualified", "Did not start", "Retired", "Lapped"], "Finished")
    gap = pd.to_numeric(res["gap_to_leader"], errors="coerce")  # "+1 LAP" for lapped cars
    out["Time"] = pd.to_timedelta(gap.where(out["Position"] != 1, res["duration"]), unit="s")
    grid_key = session_key(season, round_number, GRID_FROM[code])
    grid = get("starting_grid", session_key=grid_key) if grid_key else pd.DataFrame()
    out["GridPosition"] = num.map(grid.set_index("driver_number")["position"]).astype(float) if len(grid) else np.nan
    return out.reindex(columns=RESULT_COLS)


def load_session(season: int, round_number: int, code: str) -> SimpleNamespace | None:
    """A FastF1-like session (laps, results, weather_data), empty if it hasn't
    run yet; None if OpenF1 doesn't list it at all."""
    key = session_key(season, round_number, code)
    if key is None:
        return None
    drivers = get("drivers", session_key=key)
    if drivers.empty:
        return SimpleNamespace(laps=pd.DataFrame(columns=LAP_COLS), results=pd.DataFrame(columns=RESULT_COLS),
                               weather_data=pd.DataFrame())
    drivers = drivers.drop_duplicates("driver_number").set_index("driver_number")
    rc = get("race_control", session_key=key)
    w = get("weather", session_key=key)
    weather = pd.DataFrame({"AirTemp": w["air_temperature"], "TrackTemp": w["track_temperature"],
                            "Rainfall": w["rainfall"].astype(bool), "WindSpeed": w["wind_speed"],
                            "Humidity": w["humidity"]}) if len(w) else pd.DataFrame()
    return SimpleNamespace(laps=_laps(key, drivers, rc, race=code in GRID_FROM), results=_results(season, round_number, code, key, drivers),
                           weather_data=weather)
