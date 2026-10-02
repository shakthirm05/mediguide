from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class DialogueState:

    # ==================================================
    # SESSION
    # ==================================================

    session_id: str = "default"


    # ==================================================
    # CONVERSATION
    # ==================================================

    current_intent: Optional[str] = None

    conversation_goal: Optional[str] = None


    # ==================================================
    # ENTITIES
    # ==================================================

    entities: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "doctor": None,
            "department": None,
            "service": None,
            "destination": None
        }
    )


    # ==================================================
    # MISSING INFORMATION
    # ==================================================

    missing_information: List[str] = field(
        default_factory=list
    )

    awaiting: Optional[str] = None


    # ==================================================
    # LAST TURN
    # ==================================================

    last_question: Optional[str] = None

    last_answer: Optional[str] = None


    # ==================================================
    # NAVIGATION CONTEXT
    # ==================================================

    navigation_context: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "current_location": "main_entrance",
            "destination": None
        }
    )


    # ==================================================
    # UPDATE STATE
    # ==================================================

    def update(
        self,
        intent=None,
        entities=None,
        missing_information=None,
        awaiting=None,
        conversation_goal=None
    ):

        # --------------------------------------------------
        # Intent
        # --------------------------------------------------

        if intent is not None:
            self.current_intent = intent


        # --------------------------------------------------
        # Conversation goal
        # --------------------------------------------------

        if conversation_goal is not None:
            self.conversation_goal = conversation_goal


        # --------------------------------------------------
        # Entities
        # --------------------------------------------------

        if entities:

            for key, value in entities.items():

                if key not in self.entities:
                    continue

                # Never overwrite an existing entity with None
                if value is not None:
                    self.entities[key] = value


        # --------------------------------------------------
        # Missing information
        # --------------------------------------------------

        if missing_information is not None:

            self.missing_information = list(
                missing_information
            )


        # --------------------------------------------------
        # Awaiting
        # --------------------------------------------------

        if awaiting is not None:

            self.awaiting = awaiting


        # --------------------------------------------------
        # Navigation destination
        # --------------------------------------------------

        if self.entities.get("destination"):

            self.navigation_context[
                "destination"
            ] = self.entities["destination"]


    # ==================================================
    # DETERMINE REQUIRED INFORMATION
    # ==================================================

    def determine_missing_information(self):

        # --------------------------------------------------
        # Appointment
        # --------------------------------------------------

        if self.current_intent == "appointment":

            # Doctor is required for appointment assistance
            if not self.entities.get("doctor"):

                if "doctor" not in self.missing_information:

                    self.missing_information.append(
                        "doctor"
                    )

                self.awaiting = "doctor"


        # --------------------------------------------------
        # Appointment navigation
        # --------------------------------------------------

        elif self.current_intent == "appointment_navigation":

            # If doctor is already known, navigation can
            # continue using the existing appointment context.
            if self.entities.get("doctor"):

                self.missing_information = [
                    item
                    for item in self.missing_information
                    if item != "doctor"
                ]

                self.awaiting = None


    # ==================================================
    # RESOLVE CONTEXTUAL RESPONSES
    # ==================================================

    def resolve_context(
        self,
        user_message: str
    ):

        message = user_message.strip()


        # --------------------------------------------------
        # Waiting for doctor
        # --------------------------------------------------

        if self.awaiting == "doctor":

            if message:

                self.entities["doctor"] = message

                self.missing_information = [
                    item
                    for item in self.missing_information
                    if item != "doctor"
                ]

                self.awaiting = None

                # Continue appointment conversation
                self.current_intent = "appointment"


        # --------------------------------------------------
        # Waiting for department
        # --------------------------------------------------

        elif self.awaiting == "department":

            if message:

                self.entities["department"] = message

                self.missing_information = [
                    item
                    for item in self.missing_information
                    if item != "department"
                ]

                self.awaiting = None


        # --------------------------------------------------
        # Waiting for service
        # --------------------------------------------------

        elif self.awaiting == "service":

            if message:

                self.entities["service"] = message

                self.missing_information = [
                    item
                    for item in self.missing_information
                    if item != "service"
                ]

                self.awaiting = None


        # --------------------------------------------------
        # Waiting for destination
        # --------------------------------------------------

        elif self.awaiting == "destination":

            if message:

                self.entities["destination"] = message

                self.navigation_context[
                    "destination"
                ] = message

                self.missing_information = [
                    item
                    for item in self.missing_information
                    if item != "destination"
                ]

                self.awaiting = None


    # ==================================================
    # STORE CONVERSATION TURN
    # ==================================================

    def store_turn(
        self,
        question=None,
        answer=None
    ):

        if question is not None:
            self.last_question = question

        if answer is not None:
            self.last_answer = answer


    # ==================================================
    # CONVERT TO DICTIONARY
    # ==================================================

    def to_dict(self):

        return {

            "session_id":
                self.session_id,

            "current_intent":
                self.current_intent,

            "conversation_goal":
                self.conversation_goal,

            "entities":
                self.entities.copy(),

            "missing_information":
                list(
                    self.missing_information
                ),

            "awaiting":
                self.awaiting,

            "last_question":
                self.last_question,

            "last_answer":
                self.last_answer,

            "navigation_context":
                self.navigation_context.copy()

        }


    # ==================================================
    # DISPLAY STATE
    # ==================================================

    def display(self):

        print(
            "\n" + "=" * 70
        )

        print(
            "DIALOGUE STATE"
        )

        print(
            "=" * 70
        )

        print(
            f"Intent: {self.current_intent}"
        )

        print(
            f"Goal: {self.conversation_goal}"
        )

        print(
            f"Entities: {self.entities}"
        )

        print(
            f"Missing: {self.missing_information}"
        )

        print(
            f"Awaiting: {self.awaiting}"
        )

        print(
            f"Navigation: {self.navigation_context}"
        )

        print(
            "=" * 70
        )