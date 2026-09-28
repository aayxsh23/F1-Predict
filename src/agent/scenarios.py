"""Championship maths: what a driver needs to win the title, and the chance
they do.

`title_scenario` is exact arithmetic (points pool, elimination, points needed
to draw level if the rival scores nothing more). `title_odds` simulates the
rest of the season many times, each remaining race drawn from the race
sampler with every driver's current forecast strength.

Simplification: ties on points are not broken by FIA countback (most wins,
then most seconds...); a tie splits the title chance evenly.
"""
import numpy as np

from src.models.probabilities import sample_positions

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


def title_odds(points, predicted_finish, remaining: list[dict], tau: float, p_dnf=0.1,
               groups=None, n: int = 5000, seed: int = 0) -> dict:
    """points / predicted_finish: current points and forecast finishing
    position per driver (lower = stronger). `groups`: optional team per
    driver, to also get constructors' odds. Returns per-driver arrays
    `champion` and `expected_points`, plus `team_champion` {team: p}."""
    points = np.asarray(points, dtype=float)
    score = -np.asarray(predicted_finish, dtype=float)
    total = np.tile(points, (n, 1))
    for i, r in enumerate(remaining):
        total += _points(sample_positions(score, tau, p_dnf, n=n, seed=seed + 2 * i), RACE_POINTS)
        if r["is_sprint"]:
            total += _points(sample_positions(score, tau, p_dnf, n=n, seed=seed + 2 * i + 1), SPRINT_POINTS)
    best = total.max(axis=1, keepdims=True)
    share = (total == best) / (total == best).sum(axis=1, keepdims=True)  # ties split the title
    out = {"champion": share.mean(axis=0), "expected_points": total.mean(axis=0)}
    if groups is not None:
        teams = sorted(set(groups))
        idx = np.array([teams.index(g) for g in groups])
        team_total = np.zeros((n, len(teams)))
        np.add.at(team_total, (slice(None), idx), total)
        tbest = team_total.max(axis=1, keepdims=True)
        tshare = (team_total == tbest) / (team_total == tbest).sum(axis=1, keepdims=True)
        out["team_champion"] = dict(zip(teams, tshare.mean(axis=0)))
    return out
