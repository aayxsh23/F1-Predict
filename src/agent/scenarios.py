"""Championship maths: what a driver needs to win the title, and the chance
they do.

`title_scenario` is exact arithmetic (points pool, elimination, points needed
to draw level if the rival scores nothing more). `title_odds` simulates the
rest of the season many times: each remaining race drawn from the race
sampler with each driver's strength at that venue (or one strength for every
round), optionally with each team's form drifting from round to round, and
ties on points settled by FIA countback (most Grand Prix wins, then seconds,
down to fifths; a tie beyond that splits the title).
"""
import numpy as np


RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # P1..P10
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]  # P1..P8


def remaining_rounds(schedule: list[dict], completed_through_round: int) -> list[dict]:
    return [r for r in schedule if r["round"] > completed_through_round]


def max_points_available(remaining: list[dict]) -> int:
    """Most points one driver can still score: win every race and sprint."""
    return len(remaining) * RACE_POINTS[0] + sum(r["is_sprint"] for r in remaining) * SPRINT_POINTS[0]


def closest_rival(standings: list[dict], driver_code: str) -> dict:
    """The highest-scoring other driver: the leader if `driver` trails, second place if `driver` leads."""
    return max((d for d in standings if d["code"] != driver_code), key=lambda d: d["points"])


def title_scenario(driver_points: float, rival_points: float, remaining: list[dict]) -> dict:
    """Can `driver` still finish ahead of `rival`, and how many points would
    they need to draw level if the rival scores nothing more? (A trailing
    driver can never *guarantee* the title against a rival who can match
    them, so "points to draw level" is the useful number.)"""
    pool = max_points_available(remaining)
    needed = max(0, rival_points - driver_points + 1)
    return {
        "driver_points": driver_points,
        "rival_points": rival_points,
        "points_behind": rival_points - driver_points,
        "remaining_races": len(remaining),
        "remaining_sprints": sum(r["is_sprint"] for r in remaining),
        "max_points_available": pool,
        "still_mathematically_in_contention": driver_points + pool > rival_points,
        "points_needed_if_rival_scores_zero": needed if needed <= pool else None,
    }


def _points(positions: np.ndarray, table: list[int]) -> np.ndarray:
    lookup = np.zeros(max(positions.max(), len(table)) + 1)
    lookup[1: len(table) + 1] = table
    return lookup[positions]


COUNTBACK = 5  # finishing places compared on a tie: wins, then seconds, ... fifths


def _sample(score: np.ndarray, tau: float, p_dnf, rng) -> np.ndarray:
    """Finishing positions (n, cars) for per-sample scores (n, cars), higher = better."""
    perf = score / tau + rng.gumbel(size=score.shape)
    retired = rng.random(score.shape) < np.broadcast_to(np.asarray(p_dnf, dtype=float), score.shape)
    perf = np.where(retired, -1e9 + rng.random(score.shape), perf)
    return (-perf).argsort(axis=1).argsort(axis=1) + 1


def _winner_share(total: np.ndarray, places: np.ndarray) -> np.ndarray:
    """Each column's share of titles over the samples: most points, then
    countback on Grand Prix places (FIA), any remaining tie split evenly."""
    key = total.copy()
    for k in range(places.shape[2]):
        key += places[:, :, k] * 1e-3 ** (k + 1)  # counts <= 30 never outweigh a point, or the place above
    best = key.max(axis=1, keepdims=True)
    return ((key == best) / (key == best).sum(axis=1, keepdims=True)).mean(axis=0)


def title_odds(points, predicted_finish, remaining: list[dict], tau: float, p_dnf=0.1,
               groups=None, n: int = 5000, seed: int = 0, drift: float = 0.0, places=None) -> dict:
    """points: current points per driver. predicted_finish: forecast finishing
    position per driver (lower = stronger), either one strength for every
    remaining round or one row per remaining round (rounds, drivers): the
    driver's strength at that venue. `drift`: the per-round standard
    deviation of a random walk in each team's strength (finishing places), so
    later rounds are less certain than the next one. `places`: Grand Prix
    finishes so far per driver (drivers, k), column j = number of (j+1)th
    places, for countback (default: none). `groups`: optional team per driver,
    to also get constructors' odds. Returns per-driver arrays `champion` and
    `expected_points`, plus `team_champion` {team: p}."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=float)
    cars = len(points)
    strength = np.asarray(predicted_finish, dtype=float)
    if strength.ndim == 1:
        strength = np.tile(strength, (max(len(remaining), 1), 1))
    teams = sorted(set(groups)) if groups is not None else list(range(cars))
    team_of = np.array([teams.index(g) for g in groups]) if groups is not None else np.arange(cars)
    form = np.zeros((n, len(teams)))
    total = np.tile(points, (n, 1))
    won = np.zeros((n, cars, COUNTBACK))
    if places is not None:
        won += np.asarray(places, dtype=float)[None, :, :COUNTBACK]
    for i, r in enumerate(remaining):
        if drift:
            form += rng.normal(0.0, drift, size=form.shape)
        score = -strength[i][None, :] + form[:, team_of]
        pos = _sample(score, tau, p_dnf, rng)
        total += _points(pos, RACE_POINTS)
        for k in range(COUNTBACK):
            won[:, :, k] += pos == k + 1
        if r["is_sprint"]:
            total += _points(_sample(score, tau, p_dnf, rng), SPRINT_POINTS)
    out = {"champion": _winner_share(total, won), "expected_points": total.mean(axis=0)}
    if groups is not None:
        team_total, team_won = np.zeros((n, len(teams))), np.zeros((n, len(teams), COUNTBACK))
        np.add.at(team_total, (slice(None), team_of), total)
        np.add.at(team_won, (slice(None), team_of), won)
        out["team_champion"] = dict(zip(teams, _winner_share(team_total, team_won)))
    return out
