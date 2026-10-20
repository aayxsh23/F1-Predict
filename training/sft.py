"""Fine-tune Qwen3.5 9B on the kept conversations (LoRA, bf16), then export
the merged model as a Q4_K_M GGUF for Ollama. Runs ON the rented GPU machine
(training/gpu_setup.sh installs everything); see training/README.md.

    python sft.py train  --data sft --out run [--epochs 2] [--max-hours 20] [--max-steps 30]
    python sft.py export --out run           # adapter -> merged 16-bit -> GGUF bf16 -> Q4_K_M

The loss covers only what the model writes in its current turn (the spans
from training/sft_data.py): its tool calls and its answer, never the
prompt, the tool results or earlier turns.

Unsloth: bf16 LoRA (4-bit training is not recommended for Qwen3.5 by
Unsloth), LoRA on every attention and MLP projection, gradient checkpointing.
Training stops cleanly at --max-hours and resumes from the last checkpoint
when run again with the same --out. The validation loss is checked every 200
steps and the best checkpoint is the one kept at the end.
"""
import argparse
import gzip
import json
import random
import subprocess
import sys
import time
from pathlib import Path

BASE = "Qwen/Qwen3.5-9B"
LLAMA_CPP = Path.home() / "llama.cpp"


def load_rows(path: Path, tok, max_len: int) -> list[dict]:
    import numpy as np

    rows, dropped = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            enc = tok(ex["text"], add_special_tokens=False, return_offsets_mapping=True)
            if len(enc["input_ids"]) > max_len:
                dropped += 1
                continue
            spans = ex["spans"]
            labels = [i if any(s <= a < e for s, e in spans) else -100 for i, (a, _) in zip(enc["input_ids"], enc["offset_mapping"])]
            rows.append({"input_ids": np.array(enc["input_ids"], dtype=np.int32), "labels": np.array(labels, dtype=np.int32)})  # int32: 25k long examples fit in RAM
    print(f"{path.name}: {len(rows)} examples, {dropped} longer than {max_len} tokens dropped", flush=True)
    return rows


def collate(pad_id: int):
    import torch

    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, v: torch.tensor([list(b[key]) + [v] * (n - len(b[key])) for b in batch], dtype=torch.long)  # noqa: E731
        return {"input_ids": pad("input_ids", pad_id), "labels": pad("labels", -100),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])}
    return fn


def train(args) -> None:
    from unsloth import FastLanguageModel  # first: it patches transformers

    from transformers import Trainer, TrainerCallback, TrainingArguments

    model, _ = FastLanguageModel.from_pretrained(BASE, max_seq_length=args.max_len, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")
    data = Path(args.data)
    train_rows = load_rows(data / "train.jsonl.gz", tok, args.max_len)
    val_rows = load_rows(data / "val.jsonl.gz", tok, args.max_len)
    random.Random(3407).shuffle(train_rows)
    started = time.time()

    class StopAfter(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if time.time() - started > args.max_hours * 3600:
                print(f"time limit reached at step {state.global_step}: saving and stopping", flush=True)
                control.should_save = control.should_training_stop = True

        def on_save(self, a, state, control, **kw):  # a copy off the machine, so a lost machine can resume elsewhere
            if args.s3:
                subprocess.Popen(["aws", "s3", "sync", str(out), f"{args.s3}/{out.name}/", "--only-show-errors", "--exclude", "*/optimizer.pt"])

    out = Path(args.out)
    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps or -1, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03,  # transformers v5: a fraction below 1 is a share of all steps (warmup_ratio is gone)
        weight_decay=0.0, optim="adamw_8bit", bf16=True, logging_steps=1 if args.max_steps else 10,
        eval_strategy="steps" if not args.max_steps else "no", eval_steps=200, per_device_eval_batch_size=1,
        save_strategy="steps", save_steps=200, save_total_limit=2, load_best_model_at_end=not args.max_steps,
        metric_for_best_model="eval_loss", report_to="none", seed=3407, remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=collate(tok.pad_token_id), callbacks=[StopAfter()])
    resume = any(out.glob("checkpoint-*"))
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume or None)
    steps = trainer.state.global_step
    print(json.dumps({"steps": steps, "hours": round((time.time() - t0) / 3600, 2), "log": trainer.state.log_history[-5:]}), flush=True)
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))


def export(args) -> None:
    out = Path(args.out)
    try:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(str(out / "adapter"), max_seq_length=args.max_len, load_in_4bit=False, load_in_16bit=True)
        model.save_pretrained_merged(str(out / "merged"), tok, save_method="merged_16bit")
    except Exception as exc:  # the plain route: base weights + adapter, merged by PEFT
        print(f"unsloth merge failed ({type(exc).__name__}: {exc}); merging with PEFT", flush=True)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        base = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
        merged = PeftModel.from_pretrained(base, str(out / "adapter")).merge_and_unload()
        merged.save_pretrained(str(out / "merged"))
        AutoTokenizer.from_pretrained(BASE).save_pretrained(str(out / "merged"))
    gguf(out / "merged", out / "f1-analyst")


def gguf(model_dir: Path, stem: Path) -> None:
    """HF weights -> bf16 GGUF -> Q4_K_M (what Ollama runs on the laptop)."""
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], check=True)  # noqa: E731
    run(sys.executable, LLAMA_CPP / "convert_hf_to_gguf.py", model_dir, "--outtype", "bf16", "--outfile", f"{stem}.bf16.gguf")
    run(LLAMA_CPP / "build" / "bin" / "llama-quantize", f"{stem}.bf16.gguf", f"{stem}-q4_k_m.gguf", "Q4_K_M")
    Path(f"{stem}.bf16.gguf").unlink()
    print(f"wrote {stem}-q4_k_m.gguf", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "export", "gguf-base"])
    ap.add_argument("--data", default="sft")
    ap.add_argument("--out", default="run")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="a short run to measure speed")
    ap.add_argument("--max-hours", type=float, default=20)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--s3", help="s3://bucket/prefix to copy checkpoints to")
    a = ap.parse_args()
    if a.action == "gguf-base":  # the untuned model through the same export path: proves it before the long run
        from huggingface_hub import snapshot_download

        Path(a.out).mkdir(exist_ok=True)
        gguf(Path(snapshot_download(BASE)), Path(a.out) / "base")
    else:
        {"train": train, "export": export}[a.action](a)
