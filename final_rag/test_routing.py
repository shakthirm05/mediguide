import unittest

from src.backend.services.llm_query_analyzer import (
    _deterministic_intent,
)


class DeterministicIntentTests(unittest.TestCase):

    def test_location_questions(self):
        self.assertEqual(
            _deterministic_intent("Where is Radiology?"),
            "hospital_knowledge",
        )
        self.assertEqual(
            _deterministic_intent("where is cafetria"),
            "hospital_knowledge",
        )

    def test_hospital_policy_question(self):
        self.assertEqual(
            _deterministic_intent(
                "Can you elaborate visitor policies?"
            ),
            "hospital_knowledge",
        )

    def test_navigation_requests(self):
        self.assertEqual(
            _deterministic_intent("Guide me to Radiology"),
            "navigation",
        )
        self.assertEqual(
            _deterministic_intent(
                "Give me directions to Cardiology"
            ),
            "navigation",
        )

    def test_appointment_request(self):
        self.assertEqual(
            _deterministic_intent("I have an appointment"),
            "appointment",
        )

    def test_appointment_location_question(self):
        self.assertEqual(
            _deterministic_intent(
                "I have an appointment with Dr Arun. "
                "Where should I go?"
            ),
            "appointment",
        )

    def test_ambiguous_department_name(self):
        self.assertIsNone(
            _deterministic_intent("Radiology")
        )


if __name__ == "__main__":
    unittest.main()