"""Chat with the assistant in a terminal: `python -m src.agent.cli`.
Needs GEMINI_API_KEY (and optionally GEMINI_MODEL) in the environment or .env."""
import sys

from src.agent.chat import stream_chat


def main():
    from dotenv import load_dotenv

    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    history: list[dict] = []
    print("Ask about forecasts, strategy, standings or the rules (blank line to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        history.append({"role": "user", "content": question})
        answer = []
        for e in stream_chat(history):
            if e["type"] == "token":
                answer.append(e["text"])
                print(e["text"], end="", flush=True)
            elif e["type"] == "tool_start":
                print(f"[{e['name']}]", end=" ", flush=True)
            elif e["type"] == "error":
                print(e["message"], end="")
        print("\n")
        history.append({"role": "assistant", "content": "".join(answer)})


if __name__ == "__main__":
    main()
