"""Phase 5: build the LoRA fine-tuning dataset. Per the scope decision (see
PROGRESS.md), this is agent-drafted rather than hand-written or self-distilled
from the base model -- one example per (circuit, target) combination, built
from REAL predictions and REAL retrieved corpus context (the exact same
pipeline explain.py uses at inference time), with the completion text
authored directly rather than sampled from the un-fine-tuned base model.

Why not just call the base model and clean up its output? Phase 4's own test
run showed the base model's failure mode isn't grammar -- it's grounding
discipline (repetition, and at least one internally-inconsistent claim in the
Monaco test). The whole point of this dataset is to teach exactly the
discipline the base model lacks, so the completions have to actually have
that discipline, not just be "a language model's attempt at it, cleaned up."
"""
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.features.circuit_reference import LOCATION_ALIASES
from src.models.predict import load_model, predict
from src.rag.explain import SYSTEM_PROMPT
from src.rag.ingest_corpus import format_retrieved_context, load_vector_store
from src.rag.llm import BASE_MODEL
from src.rag.shap_query import TARGET_INFO, build_retrieval_query, build_user_prompt, top_shap_features

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"
OUT_PATH = Path(__file__).resolve().parent / "training_data" / "explanations.jsonl"
# comfortably above the dataset's observed max token length -- finetune.py
# imports this same constant so the two can't silently drift apart, and
# main() below asserts no example actually exceeds it before training ever sees it
MAX_LENGTH = 1536
TARGET_COLS = {
    "finish_position": "target_finish_position", "quali_delta": "target_quali_to_race_delta",
    "qualifying": "target_qualifying_gap", "race_time": "target_race_time_gap",
}

# one accurate, specific fact per circuit, drawn directly from the
# corresponding data/corpus/race_summaries/*.txt file this agent authored --
# used to ground the composed explanation in a real, verifiable circuit fact
# rather than a generic sentence that could apply to any track
CIRCUIT_FACTS = {
    "Sakhir": "one of the easiest circuits to overtake at in the calendar (~0.25 difficulty) but also one of the highest in tyre degradation",
    "Jeddah": "one of the fastest street circuits in F1, with one of the highest safety car rates in the calendar (~0.6) due to walls close to a high-speed layout",
    "Melbourne": "a reprofiled semi-street circuit with moderate overtaking difficulty (~0.5) since its 2022 changes widened several corners specifically to aid passing",
    "Imola": "a fast, old-school circuit with few genuine passing zones, rating high on overtaking difficulty (~0.75)",
    "Miami": "a temporary circuit with moderate overtaking difficulty (~0.5) and an elevated safety car rate typical of street-adjacent tracks",
    "Monaco": "the hardest circuit to overtake at in the entire calendar (~0.95), where grid position is close to destiny",
    "Baku": "a track combining a tight old-town section with the calendar's longest straight, producing the highest safety car rate of any circuit (~0.7)",
    "Barcelona": "a circuit with high tyre degradation (~4/5) but the lowest safety car rate in the calendar (~0.2), so races usually run green start to finish",
    "Montréal": "a circuit with the shortest pit lane loss time in the calendar (~15.5s) and the highest DNF rate (~0.2), with frequent safety cars",
    "Spielberg": "a short, flowing circuit with only three real braking zones per lap, keeping overtaking difficulty moderate (~0.4)",
    "Le Castellet": "a wide circuit with asphalt run-off that dropped off the calendar after 2022, with the lowest DNF rate in the reference table (~0.08)",
    "Silverstone": "one of the fastest permanent circuits, with the highest tyre degradation rating in the calendar (~4/5) and frequent unpredictable weather",
    "Budapest": "a tight, technical circuit often compared to 'Monaco without the walls', rating near the top of the calendar for overtaking difficulty (~0.8)",
    "Spa-Francorchamps": "the longest circuit on the calendar (7.0km) and, alongside Bahrain, one of the easiest to overtake at (~0.25), with the highest rain frequency in the calendar",
    "Zandvoort": "a narrow, banked circuit with the second-highest overtaking difficulty in the calendar (~0.85), just behind Monaco",
    "Monza": "the 'Temple of Speed' and the easiest circuit to overtake at in the entire calendar (~0.2), dominated by long straights",
    "Marina Bay": "a night street circuit with the highest safety car rate (~0.9) and pit lane loss time (~24s) in the calendar",
    "Suzuka": "a technical figure-eight circuit with one of the highest tyre degradation ratings in the calendar (~4/5) and frequent autumn rain",
    "Lusail": "a fast, flowing desert circuit originally built for motorcycles, with one of the highest tyre degradation ratings in the calendar (~4/5)",
    "Austin": "a purpose-built circuit that behaves like a calendar baseline -- none of its reference-table characteristics are extreme outliers",
    "Mexico City": "a circuit at roughly 2,200m altitude, which reduces both downforce and engine power and lowers tyre degradation (~2/5) relative to sea level",
    "São Paulo": "a short anti-clockwise circuit with one of the highest rain frequencies in the calendar (~0.35) and an easy-overtaking Turn 1 straight",
    "Las Vegas": "a cold-weather night street circuit with the second-longest straight in the calendar (~1,900m), keeping overtaking difficulty moderate (~0.35) despite being a street track",
    "Yas Island": "a circuit reprofiled in 2021 to improve overtaking, now rating a moderate ~0.55, with one of the lowest DNF rates in the calendar (~0.08)",
    "Shanghai": "a circuit with a long decreasing-radius opening complex and one of the longest back straights in the calendar (~1,170m), keeping overtaking moderate (~0.4)",
    "Madrid": "a new 2026 street circuit whose reference-table values are explicitly flagged as a rougher inaugural estimate rather than data-backed numbers",
}

# per-feature sentence template, parameterized by the SHAP direction phrase
# (e.g. "higher (worse finish)") -- kept accurate to what the feature actually
# measures rather than a generic "X mattered" filler sentence
FEATURE_TEMPLATES = {
    "grid_x_overtaking_difficulty": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "overtaking_difficulty": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "grid_position": "{phrase_cap} {direction} -- at {circuit}, {fact}.",
    "quali_gap_to_pole": "{phrase_cap} {direction}, a direct signal from this weekend's own qualifying session.",
    "practice_pace": "{phrase_cap} {direction}, a direct signal from this weekend's own practice sessions.",
    "driver_recent_form": "{phrase_cap} {direction}, reflecting the driver's underlying form heading into this race.",
    "driver_track_form": "{phrase_cap} {direction}, based on this driver's own history at {circuit} specifically.",
    "driver_positions_gained_form": "{phrase_cap} {direction}, based on how many places this driver typically gains or loses on race day.",
    "driver_dnf_rate": "{phrase_cap} {direction}.",
    "team_recent_form": "{phrase_cap} {direction}, pointing to the car's overall competitiveness this season.",
    "team_quali_pace": "{phrase_cap} {direction}, reflecting the car's one-lap qualifying pace this season.",
    "team_race_pace": "{phrase_cap} {direction}, reflecting the car's race-stint pace this season.",
    "team_reliability": "{phrase_cap} {direction}.",
    "team_track_type_form": "{phrase_cap} {direction}, based on how this car type has performed at similar circuits.",
    "teammate_quali_gap": "{phrase_cap} {direction} relative to their teammate, which isolates driver skill from car performance since they share equipment.",
    "teammate_race_pace_gap": "{phrase_cap} {direction} relative to their teammate's historical race pace.",
    "grid_vs_expected_position": "{phrase_cap} {direction}.",
    "starting_tire_compound": "{phrase_cap} {direction}.",
    "expected_stops": "{phrase_cap} {direction}, based on this circuit's typical strategy.",
    "historical_compound_performance": "{phrase_cap} {direction}, based on how this compound has performed at {circuit} historically.",
    "safety_car_frequency": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "tyre_degradation_level": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "dnf_rate": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "pit_lane_loss_time": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "longest_straight_m": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "braking_zone_count": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "rain_race_frequency": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "track_length_km": "{phrase_cap} {direction}, since {circuit} is {fact}.",
    "is_street_circuit": "{phrase_cap} {direction}, since {circuit} is {fact}.",
}
for _w in ("air_temp", "track_temp", "rain_probability", "wind_speed", "wet_track_probability"):
    FEATURE_TEMPLATES[_w] = "{phrase_cap} {direction} for this race weekend."


def _circuit_fact(circuit: str) -> str:
    # ingest.py's raw `location` uses whatever FastF1 called the circuit that
    # season (e.g. "Monte Carlo" from 2026 onward) -- resolve the same alias
    # map circuit_reference.py uses, instead of a second copy of it here
    canonical = LOCATION_ALIASES.get(circuit, circuit)
    if canonical not in CIRCUIT_FACTS:
        # fail loud, matching circuit_reference.resolve()'s own philosophy for
        # this exact class of lookup -- a silent generic fallback here would
        # bake "a circuit with its own distinct characteristics" into training
        # data for a real circuit with no warning, the first time a new
        # calendar entry is added to circuit_reference.csv but not here too
        raise KeyError(
            f"no CIRCUIT_FACTS entry for {circuit!r} (canonical: {canonical!r}) -- "
            f"add one before building the fine-tuning dataset for this circuit"
        )
    return CIRCUIT_FACTS[canonical]


def _feature_sentence(feature: dict, circuit: str) -> str:
    # "pushed the prediction {word}" is the verb every template below is
    # missing without this -- {phrase_cap} is a noun phrase, so concatenating
    # it directly with "higher (worse finish)" reads as a sentence fragment
    direction = f"pushed the prediction {feature['_direction']}"
    fact = _circuit_fact(circuit)
    template = FEATURE_TEMPLATES.get(feature["feature"], "{phrase_cap} {direction}.")
    return template.format(phrase_cap=feature["phrase"].capitalize(), direction=direction, circuit=circuit, fact=fact)


# features whose template already injects the circuit fact -- if one of
# these is among the top 3, the closing sentence skips restating the same
# fact a second time. Derived from FEATURE_TEMPLATES itself (which features
# use "{fact}") rather than hand-maintained separately, so the two can't
# silently drift apart when a template is added or edited later
_FACT_REFERENCING_FEATURES = {feat for feat, tmpl in FEATURE_TEMPLATES.items() if "{fact}" in tmpl}


def compose_explanation(target: str, prediction: float, circuit: str, features: list[dict]) -> str:
    target_label, worse_word, better_word = TARGET_INFO[target]
    for f in features:
        f["_direction"] = worse_word if f["shap_value"] > 0 else better_word

    top3 = features[:3]
    lead = f"The model predicts a {target_label} of {prediction:.1f} at {circuit}."
    body = " ".join(_feature_sentence(f, circuit) for f in top3)
    fact_already_used = any(f["feature"] in _FACT_REFERENCING_FEATURES for f in top3)
    if fact_already_used:
        return f"{lead} {body}"
    close = f"That's consistent with {circuit} being {_circuit_fact(circuit)}."
    return f"{lead} {body} {close}"


def build_examples(rows_per_combo: int = 1) -> list[dict]:
    df = pd.read_parquet(DATA_PATH)
    store = load_vector_store()
    examples = []

    for target, target_col in TARGET_COLS.items():
        model = load_model(target)
        valid = df[df[target_col].notna()]
        for circuit, group in valid.groupby("location"):
            sample = group.sort_values("race_date").tail(rows_per_combo)
            for _, r in sample.iterrows():
                row = pd.DataFrame([r])
                prediction = float(predict(model, row, target=target).iloc[0])
                features = top_shap_features(model, row, target=target, top_k=5)
                query = build_retrieval_query(circuit, prediction, target, features)
                retrieved = store.similarity_search(query, k=4)
                context = format_retrieved_context(retrieved)
                user_prompt = build_user_prompt(prediction, target, circuit, features, context)
                completion = compose_explanation(target, prediction, circuit, features)
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": completion},
                    ]
                })
    return examples


def _assert_within_max_length(examples: list[dict]) -> None:
    # finetune.py's truncation_mode="keep_start" silently drops the tail of
    # any example over MAX_LENGTH -- exactly the assistant completion the
    # fine-tune is trying to teach. Catch that here, at dataset-build time,
    # rather than letting it happen silently mid-training.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    too_long = []
    for ex in examples:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        n_tokens = len(tokenizer(text)["input_ids"])
        if n_tokens > MAX_LENGTH:
            too_long.append(n_tokens)
    if too_long:
        raise ValueError(
            f"{len(too_long)} example(s) exceed MAX_LENGTH={MAX_LENGTH} tokens "
            f"(max found: {max(too_long)}) -- would be silently truncated during training; "
            f"raise MAX_LENGTH in both this file and finetune.py instead of ignoring this"
        )


def main():
    examples = build_examples()
    _assert_within_max_length(examples)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
