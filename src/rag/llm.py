"""Local generation -- Llama 3.2 1B Instruct, 4-bit quantized, run entirely
on-device via transformers, with the project's own LoRA adapter (see
ADAPTER_DIR). No external API call anywhere in this module.

torch/transformers/peft are imported lazily, inside the functions that
actually need them, not at module level. That matters: this module is
imported by the chat router so the router can offer its local-LLM-backed
intents (explain_prediction, general) when the extras are installed, but
importing it must stay cheap everywhere else -- most chat intents are plain
Python/data lookups that never touch a model, and the API that serves them
must not require a GPU or a ~2GB dependency stack just to start up. Check
`is_available()` before calling `generate()`/`generate_stream()`.
"""
import importlib.util
from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent / "saved" / "explainer_lora"

_model = None
_tokenizer = None
_loaded_with_adapter = None


def is_available() -> bool:
    """Whether the local-LLM extras are installed (requirements-llm.txt).
    A cheap check -- it doesn't import torch, just asks if it's findable --
    so callers can branch before paying any import cost."""
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def _load(use_adapter: bool = True):
    global _model, _tokenizer, _loaded_with_adapter
    if _model is not None and _loaded_with_adapter == use_adapter:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def _inputs(model, tokenizer, system_prompt: str, user_prompt: str):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # this transformers version returns a BatchEncoding (dict of input_ids/
    # attention_mask), not a bare tensor, despite only passing return_tensors
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)


def generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 400) -> str:
    model, tokenizer = _load()
    inputs = _inputs(model, tokenizer, system_prompt, user_prompt)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_stream(system_prompt: str, user_prompt: str, max_new_tokens: int = 400):
    """Same as generate(), yielding text chunks as they're produced -- lets
    the chat endpoint stream a real token-by-token answer instead of waiting
    for the whole thing (a 1B model on a 4GB GPU is not instant)."""
    import threading

    from transformers import TextIteratorStreamer

    model, tokenizer = _load()
    inputs = _inputs(model, tokenizer, system_prompt, user_prompt)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()
    try:
        yield from streamer
    finally:
        thread.join(timeout=1)
