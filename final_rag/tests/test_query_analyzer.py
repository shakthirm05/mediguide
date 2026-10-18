
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.backend.services.llm_query_analyzer import LLMQueryAnalyzer


def model_result(
    *,
    intent="navigation",
    entities=None,
    is_follow_up=False,
    resolved_query="Take me there",
):
    """Create a valid mock response matching the analyzer's JSON schema."""
    return {
        "intent": intent,
        "entities": entities or {},
        "is_follow_up": is_follow_up,
        "resolved_query": resolved_query,
        "needs_clarification": False,
        "clarification_question": None,
        "active_workflow": None,
    }


class TestQueryAnalyzerDestinationResolution(unittest.TestCase):

    def setUp(self):
        self.analyzer = LLMQueryAnalyzer()
        self.analyzer.client = Mock()

    def configure_model(self, result):
        self.analyzer.client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(result))
        )

    def test_resolves_contextual_navigation_when_model_misses_follow_up(self):
        self.configure_model(model_result(is_follow_up=False))

        result = self.analyzer.analyze(
            "Take me there",
            {
                "entities": {
                    "destination": "Radiology",
                    "location_code": "RAD-01",
                    "floor": "Ground Floor",
                },
                "recent_history": [
                    {"role": "user", "content": "Where is Radiology?"},
                    {
                        "role": "assistant",
                        "content": "Radiology is at RAD-01 on Ground Floor.",
                    },
                ],
            },
        )

        self.assertEqual(result["intent"], "navigation")
        self.assertTrue(result["is_follow_up"])
        self.assertEqual(result["entities"]["destination"], "Radiology")
        self.assertEqual(result["entities"]["location_code"], "RAD-01")
        self.assertIn("Radiology", result["resolved_query"])

    def test_new_destination_is_not_overwritten_by_old_destination(self):
        self.configure_model(
            model_result(
                entities={"destination": "Neurology"},
                is_follow_up=False,
                resolved_query="Give directions to Neurology",
            )
        )

        result = self.analyzer.analyze(
            "Guide me to Neurology",
            {
                "entities": {
                    "destination": "Radiology",
                    "location_code": "RAD-01",
                    "floor": "Ground Floor",
                },
                "recent_history": [
                    {"role": "user", "content": "Where is Radiology?"},
                ],
            },
        )

        self.assertEqual(result["intent"], "navigation")
        self.assertEqual(result["entities"]["destination"], "Neurology")
        self.assertNotEqual(result["entities"]["destination"], "Radiology")

    def test_does_not_invent_destination_when_context_is_missing(self):
        self.configure_model(model_result(is_follow_up=False))

        result = self.analyzer.analyze(
            "Take me there",
            {"entities": {}, "recent_history": []},
        )

        self.assertEqual(result["intent"], "navigation")
        self.assertNotIn("destination", result["entities"])

    def test_location_question_is_not_navigation(self):
        self.configure_model(
            model_result(
                intent="hospital_knowledge",
                entities={"destination": "Radiology"},
                is_follow_up=False,
                resolved_query="Where is Radiology?",
            )
        )

        result = self.analyzer.analyze(
            "Where is Radiology?",
            {
                "entities": {},
                "recent_history": [],
            },
        )

        self.assertEqual(result["intent"], "hospital_knowledge")

    def test_appointment_action_is_preserved(self):
        self.configure_model(
            model_result(
                intent="appointment",
                entities={
                    "appointment_action": "schedule",
                    "department_or_doctor": "Neurology",
                },
                is_follow_up=False,
                resolved_query="Book an appointment in Neurology",
            )
        )

        result = self.analyzer.analyze(
            "Schedule an appointment in Neurology",
            {"entities": {}, "recent_history": []},
        )

        self.assertEqual(result["intent"], "appointment")
        self.assertEqual(result["entities"]["appointment_action"], "book")


if __name__ == "__main__":
    unittest.main()
