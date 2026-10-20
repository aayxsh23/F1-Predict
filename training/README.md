# The chat model: how it is built

Pit Radio answers fixed questions (forecasts, standings, strategy...) with plain
Python. Everything else goes to a local model that looks things up with the
app's own tools and writes the answer from what it found. That model is
**Qwen3.5 9B, fine-tuned on about 25,000 example conversations**, running in
[Ollama](https://ollama.com) on the owner's PC. Nothing at runtime calls a
hosted model.

This folder builds it. Everything it writes goes to `training/data/`
(gitignored).

## 1. Training conversations (done: 24,997 kept)

```
python -m training.generate --name full --n 42000 --cap 85 --balance-from pilot
python -m training.generate --name topup --n 5200 --cap 15 --balance-from full --exclude-from full
```

- `questions.py` builds the question bank: templated questions with a known
  right first lookup, plus free-form ones written by DeepSeek V3.2 per
  category and persona. Every question is checked against the held-out
  evaluation set (`eval_questions.jsonl`, about 600) and earlier runs.
- `episodes.py` plays each question for real. gpt-oss-120b (on Amazon
  Bedrock) chooses lookups, **our tools run them on the real data**, and the
  writer answers from the results. Weekend questions replay past race
  weekends (`snapshots.py`), so the model learns to read whatever forecast
  it gets.
- `quality.py` keeps a conversation only if it passes free checks in code
  (every number and name in the answer appears in a result) and a judge
  model from another family. About 55% pass.
- `bedrock.py` and `budget_guard.py` cap the spend: every run has a hard cap,
  and the guard refuses to start anything that would leave less than $15 of
  credit.

## 2. Fine-tuning (one rented GPU)

```
python -m training.sft_data --runs full topup topup2 topup3      # -> data/sft/{train,val}.jsonl.gz
python -m training.ec2 launch --hours 30 --disk 200 --types g6e.xlarge
```

- The machine is an L40S with 48 GB (EC2 g6e.xlarge, $1.861/h). It
  **terminates itself** after `--hours`, whatever happens.
- It is driven through SSM, because this network blocks outbound SSH:
  `python -c "from training import ec2; print(ec2.run('bash ~/status.sh'))"`.
  Files travel through the training bucket.
- `gpu_setup.sh` installs Unsloth, the linear-attention kernels, llama.cpp
  and Ollama 0.40.2 (the laptop's version), and downloads the model.
- `sft.py train` runs bf16 LoRA (rank 32, every attention and MLP
  projection), 2 passes, at about 20 s per step.
  - It learns only the model's own turns: its tool calls and its answers.
  - It checks the held-out loss every 200 steps and keeps the best checkpoint.
  - Checkpoints are copied to S3.
- `sft.py export` merges the adapter, converts it to GGUF and quantises it
  to Q4_K_M (about 5.5 GB). `sft.py gguf-base` runs the same path on the
  untuned model; it was proven before the long run.

**Why the prompt format is ours:** `src/rag/qwen_format.py` renders every
prompt, for the training file and for the app.
- It matches Qwen3.5's official template exactly
  (`tests/test_qwen_format.py`).
- Ollama's built-in Qwen3.5 renderer doesn't match it: it drops the empty
  think block on earlier steps and orders the tool JSON differently. So the
  app calls Ollama in raw mode with this renderer instead.

## 3. The gate

```
python -m training.eval_chat --models qwen3.5:9b-q4_K_M f1-analyst --n 200
```

- The same 200 held-out questions go through the app's real lookup loop
  (`src.agent.chat.agent_answer`). They're scored by the training-data checks
  and one judge (Kimi K2.5).
- `f1-analyst` ships only if it beats the untuned model on picking the right
  tool and on grounded answers.
- **2026-10-20: passed.** Right first lookup 97/97 against 79/97; checks and
  judge 145 against 89; traps 32/40 against 16/40 (docs/MODELS.md).
- Re-running the command re-scores saved answers if the checks changed; only
  newly passing ones go to the judge.

## 4. Install on the PC

The model's compact form is the LoRA adapter in `data/model/adapter/`
(233 MB). With the public `Qwen/Qwen3.5-9B` it rebuilds the GGUF:
`sft.py export` on a GPU machine.

```
ollama create f1-analyst -f Modelfile      # Modelfile next to the downloaded GGUF:
                                           #   FROM ./f1-analyst-q4_k_m.gguf
                                           #   TEMPLATE {{ .Prompt }}
                                           #   PARAMETER stop "<|im_end|>"
```

Run the API on the same PC (`uvicorn src.api.main:app`). `/health` then
reports `chat_model: true`, and Pit Radio offers open questions.
- `OLLAMA_HOST` and `OLLAMA_MODEL` override the defaults
  (`http://localhost:11434`, `f1-analyst`).
- The serverless deployment never has the model. There, open questions get
  a plain fallback.

## 5. Teardown (when the phase ends)

Terminate the GPU machine (`python -m training.ec2 terminate`). Then delete:
- the training bucket;
- the roles `f1-bedrock-batch-role` and `f1-chat-gpu-role`;
- the kill-switch pieces (`kill_switch.py` lists them);
- the security group and key pair.

Every dollar spent is recorded in `data/spend_ledger.json`; `python -m training.budget_guard` shows the total and what is left.
