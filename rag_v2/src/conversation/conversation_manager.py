from ..llm.llm import QwenLLM
from ..llm.query_understanding import QueryUnderstanding

from .dialogue_state import DialogueState

from ..rag.conversation_rag import ConversationRAG

from ..voice.speech_to_text import SpeechToText
from ..voice.text_to_speech import TextToSpeech


class ConversationManager:

    def __init__(self):

        print("\nInitializing MediGuide Assistant...")

        # ==================================================
        # LOAD ONE LLM INSTANCE
        # ==================================================

        # One QwenLLM instance is shared by:
        # - Query Understanding
        # - RAG answer generation
        #
        # Do NOT create another LLM instance.
        self.llm = QwenLLM()


        # ==================================================
        # DIALOGUE STATE
        # ==================================================

        self.state = DialogueState(
            session_id="PHASE1-LIVE-001"
        )


        # ==================================================
        # QUERY UNDERSTANDING
        # ==================================================

        self.understanding = QueryUnderstanding(
            self.llm
        )


        # ==================================================
        # HOSPITAL RAG
        # ==================================================

        self.rag = ConversationRAG(
            llm=self.llm
        )


        # ==================================================
        # SPEECH TO TEXT
        # ==================================================

        print("\nInitializing Speech-to-Text...")

        self.stt = SpeechToText(
            model_name="base"
        )


        # ==================================================
        # TEXT TO SPEECH
        # ==================================================

        print("\nInitializing Text-to-Speech...")

        self.tts = TextToSpeech()


        print("\nMediGuide Assistant is ready.")


    # ==================================================
    # PROCESS TEXT MESSAGE
    # ==================================================

    def process_message(
        self,
        message: str
    ):

        message = message.strip()


        # --------------------------------------------------
        # Empty message
        # --------------------------------------------------

        if not message:

            return None


        # ==================================================
        # STEP 1:
        # CHECK WHETHER WE ARE WAITING FOR INFORMATION
        # ==================================================

        previous_awaiting = (
            self.state.awaiting
        )


        if previous_awaiting:

            # The new message may be the answer
            # to the previous clarification question.

            self.state.resolve_context(
                message
            )


            # --------------------------------------------------
            # If the requested information was successfully
            # provided, don't treat it as a completely
            # new question.
            # --------------------------------------------------

            if (
                previous_awaiting
                != self.state.awaiting
                and self.state.awaiting is None
            ):

                entity_value = (
                    self.state.entities.get(
                        previous_awaiting
                    )
                )


                answer = (
                    f"Got it. "
                    f"{previous_awaiting.capitalize()}: "
                    f"{entity_value}. "
                    f"What would you like to know?"
                )


                print(
                    f"\nAI: {answer}"
                )


                self.state.store_turn(
                    question=message,
                    answer=answer
                )


                # --------------------------------------------------
                # Return structured result
                # --------------------------------------------------

                return {

                    "input_type": "text",

                    "question": message,

                    "transcription": None,

                    "answer": answer,

                    "retrieval_query": None,

                    "retrieved_documents": [],

                    "dialogue_state":
                        self.state.to_dict()
                }


        # ==================================================
        # STEP 2:
        # UNDERSTAND LATEST USER MESSAGE
        # ==================================================

        extracted = (
            self.understanding.understand(

                user_message=message,

                dialogue_state=
                    self.state.to_dict()
            )
        )


        # --------------------------------------------------
        # Display Query Understanding
        # --------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "QUERY UNDERSTANDING"
        )

        print(
            "=" * 80
        )

        print(
            extracted
        )


        # ==================================================
        # STEP 3:
        # UPDATE DIALOGUE STATE
        # ==================================================

        self.state.update(

            intent=extracted.get(
                "intent"
            ),

            entities=extracted.get(
                "entities"
            ),

            missing_information=
                extracted.get(
                    "missing_information"
                ),

            awaiting=extracted.get(
                "awaiting"
            ),

            conversation_goal=
                extracted.get(
                    "conversation_goal"
                )
        )


        # --------------------------------------------------
        # Determine whether additional information
        # is required.
        # --------------------------------------------------

        self.state.determine_missing_information()


        # ==================================================
        # STEP 4:
        # MISSING INFORMATION
        # ==================================================

        if self.state.missing_information:

            missing = (
                self.state.missing_information[0]
            )


            # --------------------------------------------------
            # Current clarification questions
            # --------------------------------------------------

            clarification_questions = {

                "doctor":
                    "Who is your appointment with?",

                "department":
                    "Which department are you looking for?",

                "service":
                    "Which service are you asking about?",

                "destination":
                    "Where would you like to go?"
            }


            question = (
                clarification_questions.get(

                    missing,

                    f"Could you provide the "
                    f"{missing}?"
                )
            )


            print(
                f"\nAI: {question}"
            )


            # --------------------------------------------------
            # Store clarification turn
            # --------------------------------------------------

            self.state.last_question = (
                question
            )


            self.state.awaiting = (
                missing
            )


            self.state.store_turn(

                question=message,

                answer=question
            )


            # --------------------------------------------------
            # Return structured result
            # --------------------------------------------------

            return {

                "input_type": "text",

                "question": message,

                "transcription": None,

                "answer": question,

                "retrieval_query": None,

                "retrieved_documents": [],

                "dialogue_state":
                    self.state.to_dict()
            }


        # ==================================================
        # STEP 5:
        # RUN HOSPITAL RAG
        # ==================================================

        rag_result = (
            self.rag.answer(

                user_message=message,

                dialogue_state=
                    self.state.to_dict(),

                top_k=5
            )
        )


        # ==================================================
        # STEP 6:
        # EXTRACT ANSWER
        # ==================================================

        answer = (
            rag_result["answer"]
        )


        # ==================================================
        # STEP 7:
        # STORE CONVERSATION TURN
        # ==================================================

        self.state.store_turn(

            question=message,

            answer=answer
        )


        # ==================================================
        # DISPLAY ANSWER
        # ==================================================

        print(
            f"\nAI: {answer}"
        )


        # ==================================================
        # STEP 8:
        # RETURN STRUCTURED RESULT
        # ==================================================

        return {

            "input_type": "text",

            "question": message,

            "transcription": None,

            "answer": answer,

            "retrieval_query":
                rag_result[
                    "retrieval_query"
                ],

            "retrieved_documents":
                rag_result[
                    "retrieved_documents"
                ],

            "dialogue_state":
                self.state.to_dict()
        }


    # ==================================================
    # PROCESS VOICE MESSAGE
    # ==================================================

    def process_voice(
        self,
        audio_file
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            "VOICE INPUT"
        )

        print(
            "=" * 80
        )


        # ==================================================
        # STEP 1:
        # SPEECH → TEXT
        # ==================================================

        print(
            "\nTranscribing audio..."
        )


        transcription = (
            self.stt.transcribe_file(
                audio_file
            )
        )


        print(
            "\nTRANSCRIPTION:"
        )

        print(
            transcription
        )


        # --------------------------------------------------
        # Safety check for empty transcription
        # --------------------------------------------------

        if not transcription.strip():

            answer = (
                "I could not understand the audio. "
                "Please try speaking again."
            )


            return {

                "input_type": "voice",

                "question": "",

                "transcription": "",

                "answer": answer,

                "retrieval_query": None,

                "retrieved_documents": [],

                "dialogue_state":
                    self.state.to_dict()
            }


        # ==================================================
        # STEP 2:
        # SEND TRANSCRIPTION THROUGH NORMAL
        # TEXT PIPELINE
        # ==================================================

        result = (
            self.process_message(
                transcription
            )
        )


        # --------------------------------------------------
        # process_message should always return a result
        # --------------------------------------------------

        if result is None:

            return {

                "input_type": "voice",

                "question":
                    transcription,

                "transcription":
                    transcription,

                "answer":
                    "I could not process that request.",

                "retrieval_query":
                    None,

                "retrieved_documents":
                    [],

                "dialogue_state":
                    self.state.to_dict()
            }


        # ==================================================
        # STEP 3:
        # ANSWER → SPEECH
        # ==================================================

        print(
            "\n" + "=" * 80
        )

        print(
            "TEXT TO SPEECH"
        )

        print(
            "=" * 80
        )


        print(
            "\nGenerating voice response..."
        )


        self.tts.speak(
            result["answer"]
        )


        # ==================================================
        # STEP 4:
        # UPDATE RESULT FOR VOICE
        # ==================================================

        result["input_type"] = (
            "voice"
        )

        result["transcription"] = (
            transcription
        )

        result["question"] = (
            transcription
        )


        # ==================================================
        # RETURN COMPLETE VOICE RESULT
        # ==================================================

        return result


    # ==================================================
    # INTERACTIVE TEXT MODE
    # ==================================================

    def run(self):

        print(
            "\n" + "=" * 80
        )

        print(
            "MEDIGUIDE GENERAL HOSPITAL"
        )

        print(
            "AI PATIENT ASSISTANT"
        )

        print(
            "=" * 80
        )


        print(
            "\nType your question."
        )

        print(
            "Type 'state' to see dialogue state."
        )

        print(
            "Type 'exit' to quit."
        )


        while True:

            try:

                message = (
                    input(
                        "\nYou: "
                    ).strip()
                )


            except (
                KeyboardInterrupt,
                EOFError
            ):

                print(
                    "\n\nExiting..."
                )

                break


            if not message:

                continue


            # --------------------------------------------------
            # Exit
            # --------------------------------------------------

            if (
                message.lower()
                == "exit"
            ):

                print(
                    "\nGoodbye."
                )

                break


            # --------------------------------------------------
            # Show state
            # --------------------------------------------------

            if (
                message.lower()
                == "state"
            ):

                self.state.display()

                continue


            # --------------------------------------------------
            # Process message
            # --------------------------------------------------

            result = (
                self.process_message(
                    message
                )
            )


            # --------------------------------------------------
            # Display structured result
            # --------------------------------------------------

            if result:

                print(
                    "\n" + "-" * 80
                )

                print(
                    "RESULT"
                )

                print(
                    "-" * 80
                )

                print(
                    result
                )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    assistant = (
        ConversationManager()
    )

    assistant.run()