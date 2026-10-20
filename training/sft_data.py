"""Turn the kept conversations into the fine-tuning file: each conversation
rendered exactly as the app will show it to the model (src/rag/qwen_format.py,
today's tool list), plus the character spans the model must learn to write.

    python -m training.sft_data --runs full topup      # -> training/data/sft/{train,val}.jsonl.gz

1% of conversations (by id) are held back to watch the validation loss; the
600-question evaluation set was never generated from, so it stays clean.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from src.agent import toolspec
from src.rag.qwen_format import segments

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "sft"


def example(messages: list[dict], tools: list[dict]) -> dict:
    text, spans = "", []
    for piece, learn in segments(messages, tools):
        if learn:
            spans.append([len(text), len(text) + len(piece)])
        text += piece
    return {"text": text, "spans": spans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    tools = toolspec.schemas()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    with gzip.open(OUT / "train.jsonl.gz", "wt", encoding="utf-8") as tr, gzip.open(OUT / "val.jsonl.gz", "wt", encoding="utf-8") as va:
        for run in args.runs:
            for line in (DATA / run / "kept.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                split = "val" if int(hashlib.md5(row["id"].encode()).hexdigest(), 16) % 100 == 0 else "train"
                (va if split == "val" else tr).write(json.dumps({"id": row["id"], **example(row["messages"], tools)}, ensure_ascii=False) + "\n")
                counts[split] += 1
    print(json.dumps({**counts, "tools": len(tools), "out": str(OUT)}))


if __name__ == "__main__":
    main()
