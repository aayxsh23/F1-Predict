"""Turn a single prediction's SHAP feature attributions into a natural-language
retrieval query -- this is the bridge the project plan describes: SHAP output
IS the query into the RAG layer, not a separate hand-written question."""
import pandas as pd
import shap
import xgboost as xgb

from src.models.features import prepare_features

# feature name -> plain-English topic, used to phrase the top SHAP features as
# a query rather than dumping raw column names at the retriever/LLM
FEATURE_PHRASES = {
    "overtaking_difficulty": "how difficult overtaking is at this circuit",
    "is_street_circuit": "this being a street circuit",
    "pit_lane_loss_time": "pit lane time loss at this circuit",
    "safety_car_frequency": "how often safety cars occur at this circuit",
    "dnf_rate": "this circuit's retirement/DNF rate",
    "longest_straight_m": "the length of the longest straight here",
    "braking_zone_count": "the number of overtaking/braking zones on this track",
    "tyre_degradation_level": "tyre degradation at this circuit",
    "rain_race_frequency": "how often this circuit sees rain",
    "track_length_km": "the track length",
    "air_temp": "air temperature",
    "track_temp": "track temperature",
    "rain_probability": "the chance of rain this weekend",
    "wind_speed": "wind conditions",
    "wet_track_probability": "the chance of a wet track",
    "grid_position": "starting grid position",
    "quali_gap_to_pole": "the qualifying gap to pole position",
    "practice_pace": "practice session pace",
    "driver_recent_form": "the driver's recent form",
    "driver_track_form": "the driver's history at this specific circuit",
    "driver_positions_gained_form": "the driver's typical grid-to-finish gain",
    "driver_dnf_rate": "the driver's retirement rate",
    "team_recent_form": "the team's recent form",
    "team_quali_pace": "the team's qualifying pace",
    "team_race_pace": "the team's race pace",
    "team_reliability": "the team's reliability",
    "team_track_type_form": "the team's form on this type of circuit",
    "teammate_quali_gap": "the qualifying gap to their teammate",
    "teammate_race_pace_gap": "the race pace gap to their teammate",
    "grid_vs_expected_position": "how far grid position deviated from what recent form predicted",
    "starting_tire_compound": "the starting tyre compound",
    "expected_stops": "the expected number of pit stops",
    "historical_compound_performance": "historical performance on this tyre compound",
    "grid_x_overtaking_difficulty": "the combined effect of grid position and how hard this circuit is to overtake on",
}


def top_shap_features(model: xgb.XGBRegressor, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """row: a single-row DataFrame with the raw Phase 1 feature-table columns
    (same shape predict.predict() expects). Returns the top_k features driving
    THIS prediction, ranked by |SHAP value|, each with its signed contribution."""
    X = prepare_features(row)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    contributions = pd.Series(shap_values, index=X.columns).sort_values(key=abs, ascending=False)
    return [
        {"feature": feat, "shap_value": float(val), "phrase": FEATURE_PHRASES.get(feat, feat)}
        for feat, val in contributions.head(top_k).items()
    ]


def build_retrieval_query(circuit_name: str, prediction: float, target: str, top_features: list[dict]) -> str:
    """One natural-language paragraph summarizing what mattered most for this
    prediction, used as the semantic search query against the corpus.

    Sign convention differs by target: for finish_position, a HIGHER number is
    a WORSE finish (P1 is best); for quali_delta (= grid - finish), a HIGHER
    number means MORE positions gained, which is BETTER. A positive SHAP
    contribution therefore reads as "worse" for one target and "better" for
    the other -- getting this backwards would make every delta-model
    explanation say the opposite of what actually happened."""
    target_label = "finishing position" if target == "finish_position" else "quali-to-race position change"
    parts = [f"Explain the predicted {target_label} of {prediction:.1f} at {circuit_name}."]
    worse_word, better_word = ("higher (worse finish)", "lower (better finish)") if target == "finish_position" \
        else ("higher (more positions gained)", "lower (fewer positions gained)")
    for f in top_features:
        direction = f"pushed the prediction {worse_word}" if f["shap_value"] > 0 else f"pushed the prediction {better_word}"
        parts.append(f"{f['phrase'].capitalize()} {direction}.")
    return " ".join(parts)
