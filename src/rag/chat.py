"""Interactive test harness: pick a race, get all 4 predictions for every
driver, then ask free-form questions about it -- a lightweight way to try the
local explainer without writing Python each time. Not Phase 6/7's chatbot:
that's the LangGraph live-standings agent (current points, title-scenario
math from live data), a different job from per-race prediction Q&A. This
reuses everything already built (predict.py, live_predict.py, the local LLM
in llm.py, the Chroma store) rather than adding new prediction/retrieval logic.

Usage: python -m src.rag.chat <season> <round>
"""
import sys

import pandas as pd

from src.features import build_dataset
from src.models.live_predict import build_live_rows
from src.models.predict import CANONICAL_PRED_COLS, predict_all
from src.rag.ingest_corpus import format_retrieved_context, load_vector_store
from src.rag.llm import generate

HISTORY_EXCHANGES = 6  # how many past Q/A exchanges to keep in context (2 lines each)

SYSTEM_PROMPT = (
    "You are an F1 race-prediction assistant. Answer the user's question about "
    "this race using ONLY the prediction table and retrieved context provided "
    "below. Do not invent drivers, numbers, or facts that aren't given to you. "
    "If something isn't covered by the table or context, say so."
)


def build_race_predictions(season: int, round_number: int) -> pd.DataFrame:
    df = pd.read_parquet(build_dataset.OUT_PATH)
    rows = df[(df["season"] == season) & (df["round"] == round_number)].copy()
    if rows.empty:
        # not a completed historical race in the table -- treat as upcoming/live
        rows = build_live_rows(season, round_number)
    return predict_all(rows)


def format_table(rows: pd.DataFrame) -> str:
    cols = ["driver", "team"] + list(CANONICAL_PRED_COLS.values())
    # .round(2) alone doesn't stop to_string() from padding back out to 6
    # decimal places -- an explicit float_format is needed to actually show 2
    return rows[cols].sort_values("predicted_finish_position").to_string(index=False, float_format=lambda v: f"{v:.2f}")


def main():
    # stdout on this machine defaults to cp1252, which can't encode real
    # circuit names like "Sao Paulo"/"Montreal" (accented characters) --
    # force UTF-8 rather than crash mid-session on whichever race is picked
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) != 3:
        print("usage: python -m src.rag.chat <season> <round>")
        return
    try:
        season, round_number = int(sys.argv[1]), int(sys.argv[2])
    except ValueError:
        print("usage: python -m src.rag.chat <season> <round>  (both must be integers)")
        return

    rows = build_race_predictions(season, round_number)
    circuit = rows["location"].iloc[0]
    table = format_table(rows)
    print(f"\n=== {circuit} (season {season}, round {round_number}) ===")
    print(table)
    print("\nAsk questions about this race (blank line to quit).\n")

    store = load_vector_store()
    history: list[str] = []
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break

        retrieved = store.similarity_search(question, k=3)
        context = format_retrieved_context(retrieved)
        recent = "\n".join(history[-(HISTORY_EXCHANGES * 2):])
        prompt = (
            f"Circuit: {circuit}\n\nPrediction table:\n{table}\n\n"
            f"Retrieved context:\n{context}\n\n"
            + (f"Earlier in this conversation:\n{recent}\n\n" if recent else "")
            + f"Question: {question}"
        )
        answer = generate(SYSTEM_PROMPT, prompt)
        print(answer, "\n")
        history.append(f"Q: {question}")
        history.append(f"A: {answer}")


if __name__ == "__main__":
    main()
