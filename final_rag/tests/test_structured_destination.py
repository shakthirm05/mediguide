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

import unittest


class TestBuildTurnContext(unittest.TestCase):
    def test_extracts_question_and_identity(self):
        state = {
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "question": "Where is Radiology?",
            "messages": [
                HumanMessage(content="Where is Radiology?"),
                AIMessage(content="Radiology is on the ground floor."),
            ],
            "entities": {"destination": "Radiology"},
            "dialogue_slots": {},
            "active_workflow": None,
            "workflow_status": "completed",
        }

        context = build_turn_context(state)

        self.assertEqual(context["current_question"], "Where is Radiology?")
        self.assertEqual(context["conversation_id"], "conversation-1")
        self.assertEqual(context["user_id"], "user-1")
        self.assertEqual(context["entities"]["destination"], "Radiology")
        self.assertEqual(len(context["recent_history"]), 2)

    def test_keeps_only_last_twelve_messages(self):
        messages = []
        for i in range(7):
            messages.append(HumanMessage(content=f"Question {i}"))
            messages.append(AIMessage(content=f"Answer {i}"))

        context = build_turn_context({"messages": messages})

        self.assertEqual(len(context["recent_history"]), 12)
        self.assertEqual(
            context["recent_history"][0]["content"], "Question 1"
        )
        self.assertEqual(
            context["recent_history"][-1]["content"], "Answer 6"
        )

    def test_ignores_unsupported_message_types(self):
        context = build_turn_context({
            "messages": [
                HumanMessage(content="Where is Radiology?"),
                object(),
                AIMessage(content="Ground floor."),
            ]
        })

        self.assertEqual(len(context["recent_history"]), 2)

    def test_handles_missing_state_fields(self):
        context = build_turn_context({})

        self.assertEqual(context["current_question"], "")
        self.assertEqual(context["recent_history"], [])
        self.assertEqual(context["entities"], {})
        self.assertEqual(context["dialogue_slots"], {})
        self.assertEqual(context["workflow_status"], "in_progress")


if __name__ == "__main__":
    unittest.main()
