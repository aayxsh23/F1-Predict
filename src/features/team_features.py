import pandas as pd

from src.features.driver_features import QUALI_MAX_GAP
from src.features.rolling import recent_form

UPGRADE_WINDOW = 3  # races of upgrades a team's form may not have caught up with yet


def _quali_pace(df: pd.DataFrame, testing: pd.DataFrame | None) -> pd.Series:
    """Recency-weighted qualifying gap over representative laps. A season's
    pre-season test counts as one more, latest-before-round-1 session, so at
    the start of a season (and above all of a new rules era, when the old
    car's races count a tenth) the test says most and real qualifying soon
    takes over."""
    q = df[["team", "race_date", "season"]].assign(_q=df["quali_gap_pct"].where(df["quali_gap_pct"] <= QUALI_MAX_GAP))
    if testing is None or testing.empty:
        return recent_form(q, "team", "race_date", "_q")
    pseudo = pd.DataFrame({"team": testing["team"].to_numpy(), "race_date": pd.to_datetime(testing["date"]).to_numpy(),
                           "season": testing["season"].to_numpy(), "_q": testing["testing_gap_pct"].to_numpy()})
    pseudo.index = pd.RangeIndex(len(pseudo)) + (int(df.index.max()) + 1 if len(df) else 0)
    return recent_form(pd.concat([q, pseudo]), "team", "race_date", "_q").loc[df.index]


def _upgrades_recent(df: pd.DataFrame) -> pd.Series:
    """Performance upgrades a team brought over its previous UPGRADE_WINDOW
    races (this race excluded). NaN before the FIA published them (2024)."""
    if "performance_upgrades" not in df:
        return pd.Series(float("nan"), index=df.index)
    races = df[["team", "season", "round", "race_date", "performance_upgrades"]].drop_duplicates(["team", "season", "round"])
    races = races.sort_values("race_date")
    races["team_upgrades_recent"] = races.groupby("team")["performance_upgrades"].transform(
        lambda s: s.shift(1).rolling(UPGRADE_WINDOW, min_periods=1).sum())
    return df.merge(races[["team", "season", "round", "team_upgrades_recent"]], on=["team", "season", "round"],
                    how="left")["team_upgrades_recent"].set_axis(df.index)


def build(df: pd.DataFrame, testing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-constructor-per-race form, all recency-weighted over strictly
    prior races. df must already have `track_type` merged in (circuit layer)."""
    df = df.copy()
    df["team_recent_form"] = recent_form(df, "team", "race_date", "finish_position")
    df["team_quali_pace"] = _quali_pace(df, testing)
    df["team_race_pace"] = recent_form(df, "team", "race_date", "race_pace_pct")
    df["_dnf_f"] = df["dnf"].astype(float)
    df["team_reliability"] = 1 - recent_form(df, "team", "race_date", "_dnf_f", decay=0.9)

    df["_team_type_key"] = df["team"] + "|" + df["track_type"]
    df["team_track_type_form"] = recent_form(df, "_team_type_key", "race_date", "finish_position")

    # car upgrades (FIA car presentations): this weekend's, published on the
    # Friday, and the recent ones form hasn't caught up with yet
    df["team_performance_upgrades"] = df.reindex(columns=["performance_upgrades"])["performance_upgrades"]
    df["team_upgrades_recent"] = _upgrades_recent(df)

    return df[[
        "season", "round", "driver", "team_recent_form", "team_quali_pace",
        "team_race_pace", "team_reliability", "team_track_type_form",
        "team_performance_upgrades", "team_upgrades_recent",
    ]]
