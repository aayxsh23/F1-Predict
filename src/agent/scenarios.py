"""F1 championship scenario math -- the multi-step reasoning behind questions
like "what does Norris need to win the title this weekend" (the project
plan's own example). Deterministic Python, not LLM-driven: pulling points,
computing the remaining-race points pool, and comparing ceilings/floors is
exact arithmetic, not something to hand to a 1B model and hope it gets right.

Known, deliberate simplification: this does NOT implement full FIA countback
tie-break rules (equal points on points is broken by most wins, then most
2nd places, etc.). It answers "how many points would guarantee it," not "who
technically wins a countback at equal points" -- correct for the vast
majority of real scenarios, and the honest thing for a "points needed to
guarantee" answer to state is the points threshold, not adjudicate a tie.
"""
RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # P1..P10, points-paying positions
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]  # P1..P8


def remaining_rounds(schedule: list[dict], completed_through_round: int) -> list[dict]:
    return [r for r in schedule if r["round"] > completed_through_round]


def max_points_available(remaining: list[dict]) -> int:
    """Max points a single driver could still score: win every remaining
    race and every remaining sprint outright."""
    races = len(remaining)
    sprints = sum(r["is_sprint"] for r in remaining)
    return races * max(RACE_POINTS) + sprints * max(SPRINT_POINTS)


def closest_rival(standings: list[dict], driver_code: str) -> dict:
    """The driver most relevant to a title question when none is named: the
    highest-points driver other than this one. If `driver` isn't leading,
    that's simply the actual points leader (who they'd have to overhaul); if
    `driver` IS leading, the highest-points *other* driver is automatically
    whoever is 2nd -- the real threat to their lead. One expression handles
    both cases; no need to branch on whether `driver` happens to be P1."""
    others = [d for d in standings if d["code"] != driver_code]
    return max(others, key=lambda d: d["points"])


def title_scenario(driver_points: float, rival_points: float, remaining: list[dict]) -> dict:
    """Is `driver` still mathematically able to finish ahead of `rival`, and
    what they'd need to score across the remaining races to draw level
    assuming the rival scores nothing more.

    Earlier version of this function computed "points needed to guarantee
    finishing ahead of the rival regardless of what the rival does" -- that
    number turns out to only ever be achievable (non-None) when `driver` is
    ALREADY ahead of `rival`, because the rival's own ceiling grows by the
    identical remaining-points pool: a driver who trails can never force a
    guarantee against a rival who could simply match their result for the
    rest of the season. Mathematically correct, but useless for exactly the
    question this function exists to answer (a driver who's behind, like the
    plan's own "what does Norris need" example) -- caught by
    tests/test_scenarios.py asserting a concrete expected value and getting
    None back. "Needed if the rival scores zero more" is the standard framing
    real title-race analysis uses instead: always a well-defined number,
    and it's exactly the gap the trailing driver has to make up."""
    pool = max_points_available(remaining)
    driver_ceiling = driver_points + pool

    still_in_contention = driver_ceiling > rival_points
    points_needed_if_rival_scores_zero = max(0, rival_points - driver_points + 1)

    return {
        "driver_points": driver_points,
        "rival_points": rival_points,
        "points_behind": rival_points - driver_points,
        "remaining_races": len(remaining),
        "remaining_sprints": sum(r["is_sprint"] for r in remaining),
        "max_points_available": pool,
        "still_mathematically_in_contention": still_in_contention,
        "points_needed_if_rival_scores_zero": (
            points_needed_if_rival_scores_zero if points_needed_if_rival_scores_zero <= pool else None
        ),
    }
