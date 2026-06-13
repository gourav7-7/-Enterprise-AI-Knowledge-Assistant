"""
Multi-turn chatbot using LangChain ChatModels and message history.

Maintains conversation context across turns via LangChain message objects.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

SYSTEM_PROMPT = (
    "You are an enterprise AI knowledge assistant. "
    "Answer clearly, stay on topic, and ask for clarification when needed."
)


class MultiTurnChatbot:
    """Stateful multi-turn chat using ChatOpenAI."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=cfg.openai_temperature,
            api_key=cfg.openai_api_key,
        )
        self._history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    @property
    def turn_count(self) -> int:
        """Number of user/assistant exchange pairs (excludes system message)."""
        non_system = [m for m in self._history if not isinstance(m, SystemMessage)]
        return len(non_system) // 2

    def reset(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant reply."""
        user_message = user_message.strip()
        if not user_message:
            return "Please enter a message."

        self._history.append(HumanMessage(content=user_message))
        response = self._llm.invoke(self._history)
        text = response.content if isinstance(response.content, str) else str(response.content)
        self._history.append(AIMessage(content=text))
        return text

    def get_history_text(self) -> str:
        """Human-readable transcript for debugging or display."""
        lines: list[str] = []
        for msg in self._history:
            if isinstance(msg, SystemMessage):
                continue
            role = "You" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
