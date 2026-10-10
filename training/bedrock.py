"""Amazon Bedrock Converse wrapper for generating training data: retries on
throttling, thread-safe token and dollar accounting, and a hard spending cap
that stops a run instead of letting it overspend. Credentials come from the
AWS CLI's own files (~/.aws/credentials); nothing secret is ever read or
written here."""
import atexit
import os
import threading
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Decided 2026-10-07 after comparing five writers on the same 100 questions (see PROGRESS.md):
WRITER = "openai.gpt-oss-120b-1:0"   # writes the conversations: DeepSeek-level quality at about a third of the cost
JUDGE = "deepseek.v3.2"              # grades them (a different family from the writer)
QUESTIONER = "deepseek.v3.2"         # writes the free-form questions (diversity matters more than price here)
NEUTRAL = "moonshotai.kimi-k2.5"     # the ONE judge for comparing writers: Moonshot, a family none of the candidates belongs to
CANDIDATES = {"deepseek": "deepseek.v3.2", "qwen3-next": "qwen.qwen3-next-80b-a3b", "minimax-m2.1": "minimax.minimax-m2.1",
              "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0"}
# USD per 1M tokens (input, output), us-east-1 on-demand, from the AWS price list on 2026-10-06
PRICES = {"openai.gpt-oss-120b-1:0": (0.15, 0.60), "deepseek.v3.2": (0.62, 1.85), "qwen.qwen3-next-80b-a3b": (0.14, 1.20),
          "zai.glm-4.7": (0.60, 2.20), "minimax.minimax-m2.1": (0.30, 1.20), "us.meta.llama4-maverick-17b-instruct-v1:0": (0.24, 0.97),
          "moonshotai.kimi-k2.5": (0.60, 3.00), "openai.gpt-oss-20b-1:0": (0.07, 0.30), "zai.glm-4.7-flash": (0.07, 0.40),
          "mistral.ministral-3-14b-instruct": (0.20, 0.20), "nvidia.nemotron-super-3-120b": (0.15, 0.65)}
RETRYABLE = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException", "InternalServerException", "ModelErrorException"}


class CapExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, cap_usd: float):
        self.cap, self.spent, self.calls, self.tokens = cap_usd, 0.0, 0, {}
        self._lock = threading.Lock()

    def add(self, model: str, tokens_in: int, tokens_out: int) -> None:
        pin, pout = PRICES[model]
        with self._lock:
            self.spent += (tokens_in * pin + tokens_out * pout) / 1e6
            self.calls += 1
            t = self.tokens.setdefault(model, [0, 0])
            t[0] += tokens_in
            t[1] += tokens_out
            if self.spent > self.cap:
                raise CapExceeded(f"spent ${self.spent:.2f} > cap ${self.cap:.2f}; stopping")

    def summary(self) -> dict:
        return {"spent_usd": round(self.spent, 3), "cap_usd": self.cap, "calls": self.calls,
                "tokens": {m: {"in": i, "out": o} for m, (i, o) in self.tokens.items()}}


class Bedrock:
    def __init__(self, cap_usd: float, region: str = "us-east-1", writer: str = WRITER, judge: str = JUDGE):
        from training import budget_guard

        budget_guard.check(cap_usd, f"Bedrock run (cap ${cap_usd:.2f})")  # the cap is the most this run can spend
        self.writer, self.judge = writer, judge
        key = f"bedrock-{os.getpid()}-{time.time():.0f}"
        atexit.register(lambda: self.ledger.spent and budget_guard.record(f"Bedrock run (pid {os.getpid()})", self.ledger.spent, key))
        self.client = boto3.client("bedrock-runtime", region_name=region,
                                   config=Config(read_timeout=180, retries={"max_attempts": 2, "mode": "standard"}, max_pool_connections=32))  # more than the worker count
        self.ledger = Ledger(cap_usd)

    def converse(self, model: str, system: str, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 700, temperature: float = 0.3, force_tool: bool = False) -> dict:
        kwargs = dict(modelId=model, system=[{"text": system}], messages=messages,
                      inferenceConfig={"maxTokens": max_tokens, "temperature": temperature})
        if tools:
            kwargs["toolConfig"] = {"tools": tools, **({"toolChoice": {"any": {}}} if force_tool else {})}
        delay = 2.0
        for attempt in range(8):
            try:
                resp = self.client.converse(**kwargs)
                self.ledger.add(model, resp["usage"]["inputTokens"], resp["usage"]["outputTokens"])
                return resp
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in RETRYABLE or attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")
