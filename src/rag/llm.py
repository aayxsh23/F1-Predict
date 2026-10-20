"""Local generation through Ollama: the fine-tuned Qwen3.5 9B ("f1-analyst",
see training/README.md for how it is built and installed), on this machine.
No hosted model anywhere.

Ollama runs in raw mode: the prompt is rendered here by qwen_format.render,
the same function that rendered the training data, so the model sees exactly
the format it learned. Only the standard library is used, so importing this
costs nothing where Ollama isn't running (the serverless API):
`is_available()` says whether it is, and every caller degrades to a
deterministic answer when it isn't.

    OLLAMA_HOST   default http://localhost:11434
    OLLAMA_MODEL  default f1-analyst
"""
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "f1-analyst")
NUM_CTX = 12288  # tools + system prompt are ~2.5k tokens; up to four lookups of ~1k each, and the answer
OPTIONS = {"temperature": 0.2, "top_p": 0.9, "num_ctx": NUM_CTX, "stop": ["<|im_end|>"]}


def _post(path: str, body: dict, timeout: float):
    req = urllib.request.Request(f"{HOST}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def is_available() -> bool:
    """Ollama is running here and has the model."""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=2) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except (OSError, ValueError):
        return False
    return MODEL in names or f"{MODEL}:latest" in names


def generate(prompt: str, max_tokens: int = 700) -> str:
    """One model turn for an already-rendered prompt (qwen_format.render)."""
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
            "options": {**OPTIONS, "num_predict": max_tokens}}
    try:
        with _post("/api/generate", body, timeout=300) as r:
            return json.load(r)["response"]
    except urllib.error.HTTPError as exc:  # a request that arrives while Ollama is still loading the model can get a 500
        if exc.code < 500:
            raise
    with _post("/api/generate", body, timeout=300) as r:
        return json.load(r)["response"]


def generate_stream(prompt: str, max_tokens: int = 700) -> Iterator[str]:
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": True, "keep_alive": "30m",
            "options": {**OPTIONS, "num_predict": max_tokens}}
    with _post("/api/generate", body, timeout=300) as r:
        for line in r:
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                return
