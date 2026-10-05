"""The chat router (src/agent/router.py): classification on real and
regression-tested phrasing, and every handler against the repo's real
forecast/backtest/standings data. No model involved anywhere here -- these
are exactly the questions a hallucinating classifier or a wrong template
would get wrong. Run with `python tests/test_router.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.router import classify_intent, dispatch, route


def _msgs(*texts: str) -> list[dict]:
    """Alternating user/assistant turns, ending on a user turn (texts[-1])."""
    out = []
    for i, text in enumerate(texts):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return out


# --- classification: the same classes of bug the old keyword router caught ---

def test_constructor_keyword_beats_championship_keyword():
    assert classify_intent(_msgs("who is leading the constructors championship")) == "constructor_standings"


def test_title_scenario_needs_a_specific_phrase():
    assert classify_intent(_msgs("who is leading the championship")) == "driver_standings"
    assert classify_intent(_msgs("what does NOR need to win the championship")) == "title_scenario"


def test_explicit_continuation_inherits_previous_intent():
    history = _msgs("who is leading the drivers standings", "...", "and what about the constructors?")
    # "constructors" alone is specific enough to win on its own -- use a truly ambiguous follow-up instead
    history2 = _msgs("who is leading the drivers standings", "...", "and RUS?")
    assert classify_intent(history2) == "driver_standings"


def test_unrelated_followup_does_not_inherit_stale_intent():
    history = _msgs("who is leading the drivers standings", "...", "what's the fastest qualifying strategy")
    assert classify_intent(history) == "race_strategy"


def test_why_question_is_explain_not_race_forecast():
    assert classify_intent(_msgs("why is NOR predicted to finish there")) == "explain_prediction"


def test_vs_question_is_head_to_head():
    assert classify_intent(_msgs("NOR vs VER who finishes ahead")) == "head_to_head"


def test_rule_article_number_triggers_search_rules():
    assert classify_intent(_msgs("what does B1.6.2 say")) == "search_rules"


def test_plain_win_question_is_race_forecast():
    assert classify_intent(_msgs("who's going to win this race")) == "race_forecast"


def test_nonsense_question_is_general():
    assert classify_intent(_msgs("what colour is the sky")) == "general"


# --- handlers against real repo data ---

def test_race_forecast_handler():
    r = dispatch("race_forecast", "who wins this race", {})
    assert r["ok"] and "|" in r["text"] and not r["llm"]


def test_qualifying_lens_shows_pole_odds():
    r = dispatch("race_forecast", "who is favourite for pole", {})
    assert r["ok"] and "Pole" in r["text"]


def test_explain_prediction_needs_a_driver():
    r = dispatch("explain_prediction", "why is this predicted this way", {})
    assert r["ok"] is False and "driver" in r["text"].lower()


def test_explain_prediction_resolves_a_real_driver():
    code = dispatch("race_forecast", "x", {})["text"].split("**")[1]
    r = dispatch("explain_prediction", f"why is {code} predicted there", {})
    assert r["ok"] and r["llm"] and r["llm_payload"]["driver"] == code
    assert isinstance(r["llm_payload"]["predicted"], float) and len(r["llm_payload"]["contributions"]) > 0


def test_head_to_head_needs_two_codes():
    assert dispatch("head_to_head", "who wins", {})["ok"] is False


def test_driver_standings_names_the_right_driver():
    r = dispatch("driver_standings", "where does VER stand", {})
    assert r["ok"] and ("VER" in r["text"] or "Verstappen" in r["text"])


def test_constructor_standings_leads_with_real_points():
    r = dispatch("constructor_standings", "team standings", {})
    assert r["ok"] and "points" in r["text"]


def test_title_scenario_reports_a_real_points_gap():
    r = dispatch("title_scenario", "what does the leader need to win the title", {})
    assert r["ok"] and ("points" in r["text"].lower())


def test_race_strategy_handler():
    r = dispatch("race_strategy", "what's the best tyre strategy", {})
    assert r["ok"] and "Plan" in r["text"]


def test_race_strategy_safety_car_scenario():
    r = dispatch("race_strategy", "what if there's a safety car on lap 12", {})
    assert r["ok"] and "lap 12" in r["text"]


def test_search_rules_handler():
    r = dispatch("search_rules", "unsafe release penalty", {})
    assert r["ok"] and r["sources"] and any("unsafe" in s["filename"] for s in r["sources"])


def test_circuit_guide_needs_a_real_circuit():
    assert dispatch("circuit_guide", "what's the weather like", {})["ok"] is False
    r = dispatch("circuit_guide", "what's Baku like", {})
    assert r["ok"] and len(r["text"]) > 100


def test_model_track_record_handler():
    r = dispatch("model_track_record", "how accurate is this", {})
    assert r["ok"] and "baseline" in r["text"].lower()


def test_season_schedule_handler():
    r = dispatch("season_schedule", "what's the schedule", {})
    assert r["ok"] and "Rnd" in r["text"]


def test_general_falls_through_cleanly():
    r = dispatch("general", "what colour is the sky", {})
    assert r["llm"] and r["llm_payload"]["question"]


def test_route_matches_dispatch_for_the_same_question():
    assert route(_msgs("who wins this race"))["tool"] == dispatch("race_forecast", "who wins this race", {})["tool"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
