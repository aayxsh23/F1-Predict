"""Phase 5: LoRA/QLoRA fine-tune of the local Llama 3.2 1B Instruct on the
explanation dataset built by build_finetune_dataset.py. Saves the adapter to
src/rag/llm.py's existing ADAPTER_DIR -- generate() picks it up automatically
next time it loads the model, no other code changes needed anywhere upstream
(retrieval, prompting) per the plan's "only the generation call changes" design.

4-bit QLoRA (not full fine-tuning) because the base model is already loaded
that way for inference (src/rag/llm.py) and because the 4GB-VRAM budget
(RTX 3050 Ti) has no headroom for full-precision gradients on even a 1B model.
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.rag.build_finetune_dataset import MAX_LENGTH
from src.rag.llm import ADAPTER_DIR, BASE_MODEL

DATASET_PATH = Path(__file__).resolve().parent / "training_data" / "explanations.jsonl"


def load_dataset() -> Dataset:
    with open(DATASET_PATH, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    return Dataset.from_list(examples)


def main():
    dataset = load_dataset()
    print(f"{len(dataset)} training examples")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_config = SFTConfig(
        output_dir=str(ADAPTER_DIR.parent / "explainer_lora_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=4,
        learning_rate=2e-4,
        bf16=True,
        max_length=MAX_LENGTH,
        # ponytail: full-sequence loss, not assistant-only -- Llama 3.2's
        # default chat template lacks the {% generation %} markers TRL needs
        # for assistant-only masking, and patching that template is more
        # complexity than a 108-example style/discipline fine-tune needs.
        # Upgrade path if the prompt-token gradient signal ever seems to hurt
        # quality: patch the tokenizer's chat_template with generation
        # markers and set assistant_only_loss=True.
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=BASE_MODEL, args=sft_config, train_dataset=dataset,
        quantization_config=quant_config, peft_config=lora_config,
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_DIR))
    print(f"saved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
