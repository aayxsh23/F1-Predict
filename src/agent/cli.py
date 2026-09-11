"""Interactive test harness for the Phase 6 live-standings agent, mirroring
rag/chat.py's pattern. Usage: python -m src.agent.cli
"""
import sys

from src.agent.graph import build_graph, ask


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    graph = build_graph()
    thread_id = "cli-session"
    print("F1 live-standings agent. Ask about driver/constructor standings or title scenarios (blank line to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        print(ask(graph, question, thread_id=thread_id), "\n")


if __name__ == "__main__":
    main()
