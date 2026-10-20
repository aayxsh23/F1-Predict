"""The chat model's wire format (Qwen3.5), in plain Python: one function turns
a conversation into the exact text the model reads, the other turns the
model's reply back into text and tool calls.

Training (training/sft.py) and serving (src/rag/llm.py, Ollama in raw mode)
both go through `render`, so the model is served precisely the format it was
fine-tuned on. Ollama's own Qwen3.5 renderer differs from the official template
(no empty think block on earlier steps, tool JSON keys in another order); the
official template is what the base model was tuned on, so this mirrors it.
tests/test_qwen_format.py checks it against the template itself.
"""
import json
import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
NO_THINK = "<think>\n\n</think>\n\n"
_TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_TAIL = """
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


def _is_tool_response(text: str) -> bool:
    return text.startswith("<tool_response>") and text.endswith("</tool_response>")


def _arg(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)  # what Jinja's |string gives: True, 3, 1.5


def segments(messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = False) -> list[tuple[str, bool]]:
    """The rendered conversation as (text, written_by_the_model) pieces. The
    second flag marks what the model itself produced in its current turn
    (answers and tool calls after the last user question): the tokens training
    learns from. Everything else is context."""
    out: list[tuple[str, bool]] = []
    system = messages[0]["content"].strip() if messages and messages[0]["role"] == "system" else None
    if tools:
        text = IM_START + "system\n" + _TOOLS_HEAD + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + _TOOLS_TAIL
        out.append((text + ("\n\n" + system if system else "") + IM_END + "\n", False))
    elif system is not None:
        out.append((f"{IM_START}system\n{system}{IM_END}\n", False))
    last_query = max((i for i, m in enumerate(messages) if m["role"] == "user" and not _is_tool_response((m["content"] or "").strip())), default=-1)
    if last_query < 0:
        raise ValueError("no user question in the conversation")
    for i, m in enumerate(messages):
        content = (m.get("content") or "").strip()
        if m["role"] == "system":
            if i:
                raise ValueError("a system message must come first")
        elif m["role"] == "user":
            out.append((f"{IM_START}user\n{content}{IM_END}\n", False))
        elif m["role"] == "assistant":
            reasoning = ""
            if "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                content = content.split("</think>")[-1].lstrip("\n")
            current = i > last_query
            out.append((f"{IM_START}assistant\n" + (f"<think>\n{reasoning}\n</think>\n\n" if current else ""), False))
            body = content
            for j, call in enumerate(m.get("tool_calls") or []):
                f = call.get("function", call)
                body += ("\n\n" if content.strip() else "") if j == 0 else "\n"
                body += f"<tool_call>\n<function={f['name']}>\n"
                body += "".join(f"<parameter={k}>\n{_arg(v)}\n</parameter>\n" for k, v in (f.get("arguments") or {}).items())
                body += "</function>\n</tool_call>"
            out += [(body + IM_END, current), ("\n", False)]
        elif m["role"] == "tool":
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            out.append(((f"{IM_START}user" if first else "") + f"\n<tool_response>\n{content}\n</tool_response>" + (IM_END + "\n" if last else ""), False))
        else:
            raise ValueError(f"unexpected role {m['role']!r}")
    if add_generation_prompt:
        out.append((f"{IM_START}assistant\n{NO_THINK}", False))
    return out


def render(messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True) -> str:
    return "".join(text for text, _ in segments(messages, tools, add_generation_prompt))


_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict):
    kind = schema.get("type")
    try:
        if kind == "integer":
            return int(value.strip())
        if kind == "number":
            return float(value.strip())
        if kind in ("array", "object"):
            return json.loads(value)
    except ValueError:
        pass  # left as text: the tool's own argument check reports it to the model
    return value


def parse_reply(text: str, tools: list[dict] | None = None) -> tuple[str, list[dict]]:
    """(the visible text, the tool calls) from one model turn. Argument values
    are typed by the tool's schema (the format writes them all as text)."""
    text = text.split(IM_END)[0]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools or []}
    calls = [{"name": name.strip(), "arguments": {k.strip(): _coerce(v, props.get(name.strip(), {}).get(k.strip(), {}))
                                                   for k, v in _PARAM_RE.findall(body)}}
             for name, body in _CALL_RE.findall(text)]
    visible = text.split("<tool_call>")[0] if calls else text
    return visible.strip(), calls
