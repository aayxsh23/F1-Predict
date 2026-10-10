"""The history tools (src/agent/history.py) against facts about F1 that are
settled and well known. If one of these fails, the tool computed the wrong
answer, which is exactly what a chat model must never be handed. Needs
data/history/ (python -m src.data.history); skipped where it isn't built.
Run with `python tests/test_history.py` or pytest."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import history as h

pytestmark = pytest.mark.skipif(not (h.HISTORY_DIR / "races.json").exists(), reason="data/history not built")


def _count(leaders: list[dict], name: str, key: str) -> int:
    return next(x[key] for x in leaders if x["name"] == name)


def test_data_covers_every_season_since_1950():
    d = h._data()
    seasons = {r["season"] for r in d["races"]}
    assert min(seasons) == 1950 and set(range(1950, max(seasons) + 1)) <= seasons
    assert len(d["races"]) > 1100


def test_fangio_career_matches_the_record_book():
    # also guards the shared-drive fix: Fangio appears twice in 7 races but started 51
    c = h.driver_career("Fangio")["career"]
    assert (c["races"], c["wins"], c["poles"], c["podiums"], c["world_championships"]) == (51, 24, 29, 35, 5)
    assert c["title_years"] == [1951, 1954, 1955, 1956, 1957]


@pytest.mark.parametrize("name,wins,poles,titles", [("Ayrton Senna", 41, 65, 3), ("Michael Schumacher", 91, 68, 7),
                                                     ("Alain Prost", 51, 32, 4), ("Sebastian Vettel", 53, 57, 4), ("Jim Clark", 25, 33, 2)])
def test_champions_career_totals(name, wins, poles, titles):
    # Prost: the record book says 33 poles; the Jolpica grid data has 32 (one pre-1994 grid slot differs). Pinned as the
    # data's answer so a silent change in how poles are counted is noticed; history.NOTES tells the model about the caveat.
    # Schumacher 68 needs the grid-slot rule: he qualified P1 at Monaco 2012 but started P6.
    c = h.driver_career(name)["career"]
    assert (c["wins"], c["poles"], c["world_championships"]) == (wins, poles, titles)


def test_a_surname_shared_by_several_drivers_asks_instead_of_guessing():
    r = h.driver_career("Schumacher")
    assert "error" in r and {c["name"] for c in r["candidates"]} >= {"Michael Schumacher", "Mick Schumacher", "Ralf Schumacher"}
    assert h.find_driver("HAM")[0] == "hamilton" and h.find_driver("Lewis Hamilton")[0] == "hamilton"
    assert "error" in h.driver_career("Zzzz Nobody")


def test_records_are_computed_not_recalled():
    wins = h.records("wins", by="driver", n=3)["leaders"]
    assert wins[0]["name"] == "Lewis Hamilton" and wins[0]["wins"] > _count(wins, "Michael Schumacher", "wins") == 91
    monaco = h.records("wins", circuit="Monaco", n=1)["leaders"][0]
    assert (monaco["name"], monaco["wins"]) == ("Ayrton Senna", 6)
    assert h.records("titles", by="team", n=1)["leaders"][0] == {"name": "Ferrari", "titles": 16}
    by_circuit = h.records("wins", by="circuit", driver="Ayrton Senna", n=1)["leaders"][0]
    assert by_circuit["wins"] == 6
    assert "error" in h.records("wins", by="nonsense")


def test_season_and_race_lookups():
    s = h.season_summary(1988)
    assert (s["drivers_champion"], s["constructors_champion"], s["races_held"], s["complete"]) == ("Ayrton Senna", "McLaren", 16, True)
    assert s["wins_by_driver"]["Alain Prost"] == 7 and s["wins_by_driver"]["Ayrton Senna"] == 8
    monaco = h.history_results(1988, 3)
    assert monaco["race"] == "Monaco Grand Prix" and monaco["classification"][0]["driver"] == "Alain Prost"
    assert len(h.history_results(grand_prix="Monaco")["winners"]) >= 70
    assert "error" in h.history_results(1949, 1) and "error" in h.history_results()


def test_the_season_in_progress_is_not_called_complete():
    latest = h._data()["latest_season"]
    st = h._data()["standings"][str(latest)]
    if st["rounds"] < st["races_in_season"]:
        assert h.season_summary(latest)["complete"] is False
        assert all(latest not in ys for ys in h._data()["titles"]["driver"].values())


def test_team_history():
    f = h.team_history("Ferrari")
    assert f["constructors_titles"][:3] == [1961, 1964, 1975] and f["wins"] > 240
    assert "error" in h.team_history("Zzzz Racing")
    # "Lotus" is several teams (Team Lotus 1958-94, Lotus F1 2011-15...): ask which, never silently pick the wrong one
    lotus = h.team_history("Lotus")
    assert "error" in lotus and {"Team Lotus", "Lotus F1"} <= {c["name"] for c in lotus["candidates"]}
    assert h.team_history("Team Lotus")["wins"] == 79 and h.team_history("Team Lotus")["constructors_titles"] == [1963, 1965, 1968, 1970, 1972, 1973, 1978]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and not hasattr(v, "pytestmark")]
    print("run with pytest for the parametrized tests")
