#!/usr/bin/env python
"""
Task 1 — Interactive multi-turn chatbot (CLI).

Usage (from project root, venv activated):
    python scripts/chat_cli.py

Commands:  /reset  — clear history   /history — show transcript   /quit — exit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot.multi_turn import MultiTurnChatbot


def main() -> None:
    print("Enterprise AI Knowledge Assistant — multi-turn chat")
    print("Type /quit to exit, /reset to clear history, /history to view transcript.\n")

    bot = MultiTurnChatbot()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            break
        if user_input.lower() == "/reset":
            bot.reset()
            print("[History cleared]\n")
            continue
        if user_input.lower() == "/history":
            print(bot.get_history_text() or "(empty)")
            print()
            continue

        reply = bot.chat(user_input)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
