from ..llm.llm import QwenLLM

from ..llm.query_understanding import (
    QueryUnderstanding
)

from ..conversation.dialogue_state import (
    DialogueState
)

from .conversation_rag import (
    ConversationRAG
)


# ==================================================
# INITIALIZE SYSTEM
# ==================================================

print(
    "\nInitializing MediGuide Phase 1..."
)


# ==================================================
# LOAD QWEN ONLY ONCE
# ==================================================

print(
    "\nLoading shared Qwen LLM..."
)

llm = QwenLLM()


# ==================================================
# QUERY UNDERSTANDING
# ==================================================

understanding = QueryUnderstanding(
    llm
)


# ==================================================
# DIALOGUE STATE
# ==================================================

state = DialogueState(
    session_id="PHASE1-TEST-001"
)


# ==================================================
# CONVERSATION RAG
# ==================================================

rag = ConversationRAG(
    llm=llm
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
# PROCESS CONVERSATION
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
    # RESOLVE CONTEXT USING PREVIOUS STATE
    # ==================================================

    state.resolve_context(
        message
    )


    # ==================================================
    # STEP 2
    # QUERY UNDERSTANDING
    # ==================================================

    extracted = understanding.understand(

        user_message=message,

        dialogue_state=state.to_dict()

    )


    # ==================================================
    # STEP 3
    # UPDATE STATE
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
    # DETERMINE NEXT REQUIRED INFORMATION
    # ==================================================

    state.determine_missing_information()


    # ==================================================
    # DISPLAY STATE
    # ==================================================

    state.display()


    # ==================================================
    # IF INFORMATION IS MISSING
    # DON'T RUN RAG
    # ==================================================

    if state.missing_information:

        print(
            "\nSystem needs more information:"
        )

        print(
            state.missing_information
        )

        continue


    # ==================================================
    # STEP 5
    # RUN CONVERSATION RAG
    # ==================================================

    answer = rag.answer(

        user_message=message,

        dialogue_state=state.to_dict()

    )


    # ==================================================
    # DISPLAY FINAL ANSWER
    # ==================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL PATIENT ANSWER"
    )

    print(
        "=" * 80
    )

    print(
        answer
    )


    # ==================================================
    # STORE TURN
    # ==================================================

    state.store_turn(

        question=message,

        answer=answer

    )


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