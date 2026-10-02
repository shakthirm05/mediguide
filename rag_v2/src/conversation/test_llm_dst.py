from ..llm.llm import QwenLLM
from ..llm.query_understanding import QueryUnderstanding
from .dialogue_state import DialogueState


# ==================================================
# LOAD LLM
# ==================================================

print("\nLoading LLM...")

llm = QwenLLM()

understanding = QueryUnderstanding(
    llm
)


# ==================================================
# CREATE DIALOGUE STATE
# ==================================================

state = DialogueState(
    session_id="TEST-001"
)


# ==================================================
# TEST CONVERSATION
# ==================================================

conversation = [

    "I have an appointment.",

    "Dr. Arun.",

    "Where should I go?"

]


# ==================================================
# PROCESS EACH TURN
# ==================================================

for turn, message in enumerate(
    conversation,
    start=1
):

    print(
        "\n\n" + "#" * 80
    )

    print(
        f"TURN {turn}"
    )

    print(
        "#" * 80
    )

    print(
        f"\nPatient: {message}"
    )


    # ==================================================
    # STEP 1
    # RESOLVE RESPONSE USING PREVIOUS STATE
    # ==================================================
    #
    # IMPORTANT:
    #
    # We do this BEFORE updating the state with the
    # current message.
    #
    # Example:
    #
    # Turn 1:
    # "I have an appointment."
    #
    # Previous state:
    # awaiting = None
    #
    # Therefore nothing is resolved.
    #
    # After Turn 1, state becomes:
    # awaiting = doctor
    #
    # Turn 2:
    # "Dr. Arun."
    #
    # Previous state:
    # awaiting = doctor
    #
    # Therefore "Dr. Arun." is stored as doctor.
    #
    # ==================================================

    state.resolve_context(
        message
    )


    # ==================================================
    # STEP 2
    # LLM UNDERSTANDS CURRENT MESSAGE
    # ==================================================

    extracted = understanding.understand(

        user_message=message,

        dialogue_state=state.to_dict()

    )


    print(
        "\nLLM EXTRACTION:"
    )

    print(
        extracted
    )


    # ==================================================
    # STEP 3
    # UPDATE STATE WITH LLM EXTRACTION
    # ==================================================

    state.update(

        intent=extracted.get(
            "intent"
        ),

        entities=extracted.get(
            "entities"
        ),

        missing_information=extracted.get(
            "missing_information"
        ),

        awaiting=extracted.get(
            "awaiting"
        ),

        conversation_goal=extracted.get(
            "conversation_goal"
        )

    )


    # ==================================================
    # STEP 4
    # DETERMINE WHAT INFORMATION IS NEEDED NEXT
    # ==================================================

    state.determine_missing_information()


    # ==================================================
    # STEP 5
    # DISPLAY CURRENT STATE
    # ==================================================

    state.display()


# ==================================================
# FINAL STATE
# ==================================================

print(
    "\n\n" + "=" * 80
)

print(
    "FINAL DIALOGUE STATE"
)

print(
    "=" * 80
)

print(
    state.to_dict()
)