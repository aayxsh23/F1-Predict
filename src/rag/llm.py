"""Local generation step -- Llama 3.2 1B Instruct, 4-bit quantized, run
entirely on-device via transformers. No external API call anywhere in this
module. Phase 5 fine-tuning swaps in a LoRA adapter here (see ADAPTER_DIR)
without touching anything upstream (retrieval, prompting stay identical)."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent / "saved" / "explainer_lora"

_model = None
_tokenizer = None
_loaded_with_adapter = None


def _load(use_adapter: bool = True):
    global _model, _tokenizer, _loaded_with_adapter
    if _model is not None and _loaded_with_adapter == use_adapter:
        return _model, _tokenizer

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")

    if use_adapter and ADAPTER_DIR.exists():
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, str(ADAPTER_DIR))

    _loaded_with_adapter = use_adapter
    return _model, _tokenizer


def generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 400) -> str:
    model, tokenizer = _load()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # this transformers version returns a BatchEncoding (dict of input_ids/
    # attention_mask), not a bare tensor, despite only passing return_tensors
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
