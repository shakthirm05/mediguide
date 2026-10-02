import json
import re


class QueryUnderstanding:

    def __init__(self, llm):

        self.llm = llm

    # ======================================================
    # UNDERSTAND USER MESSAGE
    # ======================================================

    def understand(
        self,
        user_message: str,
        dialogue_state: dict
    ):

        state_json = json.dumps(
            dialogue_state,
            indent=2
        )

        prompt = f"""
You are the query understanding component of a
hospital information assistant.

Your task is to classify the patient's latest message
and extract entities.

Return ONLY one JSON object.

Do NOT explain.
Do NOT reason.
Do NOT use markdown.
Do NOT include any text before or after the JSON.

CURRENT DIALOGUE STATE:

{state_json}

LATEST PATIENT MESSAGE:

{user_message}


AVAILABLE INTENTS:

appointment
appointment_navigation
doctor_information
doctor_department
doctor_navigation
department_navigation
registration
appointment_cancellation
appointment_rescheduling
service_information
facility_information
visitor_information
emergency_information
hospital_information
greeting
unknown


ENTITY TYPES:

doctor
department
service
destination


RULES:

1. Identify the intent from the patient's latest message.

2. If the patient refers to a doctor already present
in CURRENT DIALOGUE STATE, preserve that doctor entity.

3. If the patient refers to:
"the doctor",
"my doctor",
"my appointment doctor"

use the doctor already stored in the dialogue state.

4. If the patient explicitly provides a doctor name,
extract it.

5. If the patient asks which department a doctor
belongs to, use:

doctor_department

6. If the patient asks where the doctor or the doctor's
department is located, use:

doctor_navigation

7. If the patient asks general information about a doctor,
use:

doctor_information

8. If the patient asks where a department is located,
use:

department_navigation

9. If the patient says they have an appointment,
use:

appointment

10. If the patient asks where they should go for an
appointment, use:

appointment_navigation

11. If the patient asks about registration, use:

registration

12. If the patient asks about visitors or visiting rules,
use:

visitor_information

13. If the patient asks about emergency procedures or
emergency locations, use:

emergency_information

14. Do not invent entities.

15. Unknown entities must be null.

16. Preserve relevant entities from the current
dialogue state.

17. Return ONLY the JSON object.

Return exactly this structure:

{{
  "intent": null,
  "entities": {{
    "doctor": null,
    "department": null,
    "service": null,
    "destination": null
  }},
  "conversation_goal": null,
  "missing_information": [],
  "awaiting": null
}}
"""

        response = self.llm.generate_raw(
            prompt,
            max_new_tokens=250
        )

        print("\n" + "-" * 80)
        print("RAW LLM RESPONSE")
        print("-" * 80)
        print(response)
        print("-" * 80)

        result = self._parse_json(
            response
        )

        # --------------------------------------------------
        # Preserve doctor from dialogue state
        # --------------------------------------------------

        current_entities = dialogue_state.get(
            "entities",
            {}
        )

        if (
            result["entities"]["doctor"] is None
            and current_entities.get("doctor")
            and self._refers_to_existing_doctor(
                user_message
            )
        ):

            result["entities"]["doctor"] = (
                current_entities["doctor"]
            )

        # --------------------------------------------------
        # Preserve department from dialogue state
        # --------------------------------------------------

        if (
            result["entities"]["department"] is None
            and current_entities.get("department")
            and self._refers_to_existing_department(
                user_message
            )
        ):

            result["entities"]["department"] = (
                current_entities["department"]
            )

        print("\nPARSED RESULT:")
        print(result)

        return result

    # ======================================================
    # EXISTING DOCTOR REFERENCE
    # ======================================================

    def _refers_to_existing_doctor(
        self,
        message: str
    ):

        message = message.lower()

        references = [
            "the doctor",
            "my doctor",
            "doctor",
            "my appointment doctor"
        ]

        return any(
            phrase in message
            for phrase in references
        )

    # ======================================================
    # EXISTING DEPARTMENT REFERENCE
    # ======================================================

    def _refers_to_existing_department(
        self,
        message: str
    ):

        message = message.lower()

        references = [
            "the department",
            "my department",
            "that department"
        ]

        return any(
            phrase in message
            for phrase in references
        )

    # ======================================================
    # JSON PARSER
    # ======================================================

    def _parse_json(
        self,
        response: str
    ):

        response = response.strip()

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()

        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if not match:

            print(
                "\nWARNING: No valid JSON object found."
            )

            return self._default_result()

        json_text = match.group(0)

        try:

            data = json.loads(
                json_text
            )

        except json.JSONDecodeError as error:

            print(
                "\nWARNING: JSON parsing failed:"
            )

            print(error)

            return self._default_result()

        if not isinstance(
            data,
            dict
        ):

            return self._default_result()

        entities = data.get(
            "entities",
            {}
        )

        if not isinstance(
            entities,
            dict
        ):

            entities = {}

        normalized_entities = {

            "doctor":
                entities.get("doctor"),

            "department":
                entities.get("department"),

            "service":
                entities.get("service"),

            "destination":
                entities.get("destination")
        }

        missing_information = data.get(
            "missing_information",
            []
        )

        if not isinstance(
            missing_information,
            list
        ):

            missing_information = []

        return {

            "intent":
                data.get(
                    "intent",
                    "unknown"
                ),

            "entities":
                normalized_entities,

            "conversation_goal":
                data.get(
                    "conversation_goal"
                ),

            "missing_information":
                missing_information,

            "awaiting":
                data.get(
                    "awaiting"
                )
        }

    # ======================================================
    # DEFAULT RESULT
    # ======================================================

    def _default_result(self):

        return {

            "intent": "unknown",

            "entities": {

                "doctor": None,

                "department": None,

                "service": None,

                "destination": None
            },

            "conversation_goal": None,

            "missing_information": [],

            "awaiting": None
        }