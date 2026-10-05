"""End-to-end explainer chain: prediction -> SHAP -> SHAP-driven retrieval
query -> retrieved corpus context -> local LLM generation. No external API
anywhere in this pipeline -- embeddings, retrieval, and generation are all
local. Retrieval is the article-chunked BM25 index (src/rag/corpus.py), not
a vector database: see LEARNING.md for why that reads a rulebook better than
embedding similarity does."""
import pandas as pd

from src.models.predict import load_model, predict
from src.rag.corpus import search
from src.rag.llm import generate
from src.rag.shap_query import EXPLAIN_SYSTEM_PROMPT, build_retrieval_query, build_user_prompt, format_context, top_shap_features

SYSTEM_PROMPT = EXPLAIN_SYSTEM_PROMPT  # kept as a module-level name for backward compatibility


def explain(row: pd.DataFrame, target: str = "finish_position", top_k_features: int = 5, top_k_context: int = 4) -> dict:
    """row: single-row DataFrame with the raw Phase 1 feature-table columns
    plus 'location' (circuit name) -- same shape predict.predict() takes."""
    model = load_model(target)
    prediction = float(predict(model, row, target=target).iloc[0])
    features = top_shap_features(model, row, target=target, top_k=top_k_features)
    circuit_name = row["location"].iloc[0]

    query = build_retrieval_query(circuit_name, prediction, target, features)
    hits = search(query, k=top_k_context, circuit=circuit_name)
    user_prompt = build_user_prompt(prediction, target, circuit_name, features, format_context(hits))

    return {
        "prediction": prediction,
        "target": target,
        "circuit": circuit_name,
        "top_features": features,
        "retrieval_query": query,
        "sources": [h["source"] for h in hits],
        "explanation": generate(SYSTEM_PROMPT, user_prompt),
    }
