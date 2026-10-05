"""Turn a single prediction's SHAP feature attributions into a natural-language
retrieval query -- this is the bridge the project plan describes: SHAP output
IS the query into the RAG layer, not a separate hand-written question.

Only top_shap_features() needs pandas/XGBoost (it runs a model), so that
import is lazy, inside the function -- everything else here is plain string
logic the chat router can use on an already-computed SHAP breakdown (read
straight from a stored forecast) without pulling in either dependency."""

# the one prompt both real inference (chat.py, explain.py) and the Phase 5
# fine-tuning dataset (build_finetune_dataset.py) use, so the fine-tuned
# adapter is always prompted the same way it was trained
EXPLAIN_SYSTEM_PROMPT = (
    "You are an F1 race-prediction analyst. Explain a model's prediction in "
    "plain English, in 3-5 sentences, using ONLY the retrieved context provided "
    "below plus the listed feature contributions. Do not invent facts, "
    "regulation article numbers, or incidents that are not in the context. "
    "If the context doesn't cover something, don't mention it."
)

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
    "quali_gap_pct": "the qualifying gap to pole position",
    "practice_long_run_pace": "long-run pace in practice",
    "air_temp_forecast": "the forecast air temperature",
    "rain_mm_forecast": "the rain forecast",
    "wind_kph_forecast": "the wind forecast",
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


def top_shap_features(model, row, target: str = "finish_position", top_k: int = 5) -> list[dict]:
    """The top_k features driving THIS prediction, by |SHAP value|, each with its signed contribution.
    model: xgb.XGBRegressor, row: pd.DataFrame -- typed loosely so this file doesn't need xgboost/pandas
    at import time; only a caller that actually calls this function does."""
    from src.models.features import PREPARE_FN
    from src.models.predict import contributions

    contrib, _ = contributions(model, PREPARE_FN[target](row))
    c = contrib.iloc[0]
    c = c.reindex(c.abs().sort_values(ascending=False).index)
    return [
        {"feature": feat, "shap_value": float(val), "phrase": FEATURE_PHRASES.get(feat, feat)}
        for feat, val in c.head(top_k).items()
    ]


# (label, higher-means, lower-means) per target -- sign convention genuinely
# differs across all four: for finish_position/qualifying/race_time a HIGHER
# number is WORSE (further from P1/pole/the winner), but for quali_delta
# (= grid - finish) a HIGHER number means MORE positions gained, i.e. BETTER.
# Getting this backwards would make every delta-model explanation claim the
# opposite of what actually happened.
TARGET_INFO = {
    "finish_position": ("finishing position", "higher (worse finish)", "lower (better finish)"),
    "quali_delta": ("quali-to-race position change", "higher (more positions gained)", "lower (fewer positions gained)"),
    "qualifying": ("qualifying gap to pole (% of the pole lap)", "higher (further from pole)", "lower (closer to pole)"),
    "race_time": ("race time gap to the winner (% of race time)", "higher (further behind the winner)", "lower (closer to the winner)"),
}


def format_context(hits: list[dict], max_chars: int = 600) -> str:
    """The retrieved-context block of the user prompt, shared by chat.py,
    explain.py and build_finetune_dataset.py so training and inference never
    see a different shape. Each hit is capped at max_chars -- a shorter,
    more focused context is faster to train and run and still carries the
    citation plus the grounding fact that actually matters; the uncapped
    version was the main driver of a 2555-token worst case that made
    retraining too slow for a 4GB GPU to finish in one sitting."""
    return "\n\n".join(f"[{h['source']}{' art. ' + h['article'] if h['article'] else ''}]\n{h['text'][:max_chars]}" for h in hits)


def build_retrieval_query(circuit_name: str, prediction: float, target: str, top_features: list[dict]) -> str:
    """One natural-language paragraph summarizing what mattered most for this
    prediction, used as the semantic search query against the corpus."""
    target_label, worse_word, better_word = TARGET_INFO[target]
    parts = [f"Explain the predicted {target_label} of {prediction:.1f} at {circuit_name}."]
    for f in top_features:
        direction = f"pushed the prediction {worse_word}" if f["shap_value"] > 0 else f"pushed the prediction {better_word}"
        parts.append(f"{f['phrase'].capitalize()} {direction}.")
    return " ".join(parts)


def build_user_prompt(prediction: float, target: str, circuit_name: str, top_features: list[dict], context: str) -> str:
    """The exact user-prompt shape both real inference (chat.py, explain.py)
    and the Phase 5 fine-tuning dataset (build_finetune_dataset.py) must use
    -- one implementation so the two can never silently drift apart on
    prompt shape, which would reintroduce a train/inference skew.

    Each feature line states its own direction in words ("pushed the
    prediction toward a better finish"), not just the signed number. A real
    generation test caught the model inventing the wrong target's direction
    vocabulary (quali_delta's "fewer positions gained" on a finish_position
    explanation) when it had to infer which wording applied to which target
    from only ~27 fine-tuning examples per target; stating it plainly removes
    that inference step rather than hoping a small model gets it right."""
    target_label, worse_word, better_word = TARGET_INFO[target]
    feature_lines = "\n".join(
        f"- {f['phrase']}: SHAP contribution {f['shap_value']:+.2f} "
        f"(pushed the prediction {worse_word if f['shap_value'] > 0 else better_word})"
        for f in top_features
    )
    return (
        f"Prediction: {prediction:.1f} ({target_label}) at {circuit_name}.\n\n"
        f"Top feature contributions:\n{feature_lines}\n\n"
        f"Retrieved context:\n{context}"
    )
