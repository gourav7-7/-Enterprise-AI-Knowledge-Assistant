#!/usr/bin/env python
"""
Task 1 — First working OpenAI API call (direct SDK, no LangChain).

Usage (from project root, venv activated):
    python scripts/test_openai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    print(f"Model: {settings.openai_model}")
    print("Sending test prompt to OpenAI...\n")

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: OpenAI connection successful."},
        ],
        temperature=settings.openai_temperature,
        max_tokens=50,
    )

    reply = response.choices[0].message.content
    print("Response:", reply)
    print("\nOpenAI API call completed successfully.")


if __name__ == "__main__":
    main()
