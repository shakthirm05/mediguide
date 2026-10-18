from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def _message_text(message: Any) -> str:
    """Extract readable text from a LangChain message."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def build_turn_context(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble current-turn context from LangGraph's checkpointed state."""
    history: list[dict[str, str]] = []

    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = _message_text(message)
        if content:
            history.append({"role": role, "content": content})

    return {
        "conversation_id": state.get("conversation_id"),
        "user_id": state.get("user_id"),
        "current_question": str(state.get("question", "")).strip(),
        "recent_history": history[-12:],
        "entities": dict(state.get("entities") or {}),
        "dialogue_slots": dict(state.get("dialogue_slots") or {}),
        "active_workflow": state.get("active_workflow"),
        "workflow_status": state.get("workflow_status", "in_progress"),
    }
