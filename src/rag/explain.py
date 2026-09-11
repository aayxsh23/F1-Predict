"""End-to-end explainer chain: prediction -> SHAP -> SHAP-driven retrieval
query -> retrieved corpus context -> local LLM generation. No external API
anywhere in this pipeline -- embeddings, retrieval, and generation are all
local, per the project's Phase 4+5 scope decision (see PROGRESS.md)."""
import pandas as pd

from src.models.predict import load_model, predict
from src.rag.ingest_corpus import format_retrieved_context, load_vector_store
from src.rag.llm import generate
from src.rag.shap_query import build_retrieval_query, build_user_prompt, top_shap_features

SYSTEM_PROMPT = (
    "You are an F1 race-prediction analyst. Explain a model's prediction in "
    "plain English, in 3-5 sentences, using ONLY the retrieved context provided "
    "below plus the listed feature contributions. Do not invent facts, "
    "regulation article numbers, or incidents that are not in the context. "
    "If the context doesn't cover something, don't mention it."
)


def explain(row: pd.DataFrame, target: str = "finish_position", top_k_features: int = 5, top_k_context: int = 4) -> dict:
    """row: single-row DataFrame with the raw Phase 1 feature-table columns
    plus 'location' (circuit name) -- same shape predict.predict() takes."""
    model = load_model(target)
    prediction = float(predict(model, row, target=target).iloc[0])
    features = top_shap_features(model, row, target=target, top_k=top_k_features)
    circuit_name = row["location"].iloc[0]

    query = build_retrieval_query(circuit_name, prediction, target, features)
    retrieved = load_vector_store().similarity_search(query, k=top_k_context)
    context = format_retrieved_context(retrieved)
    user_prompt = build_user_prompt(prediction, target, circuit_name, features, context)

    return {
        "prediction": prediction,
        "target": target,
        "circuit": circuit_name,
        "top_features": features,
        "retrieval_query": query,
        "sources": [d.metadata["source"] for d in retrieved],
        "explanation": generate(SYSTEM_PROMPT, user_prompt),
    }
