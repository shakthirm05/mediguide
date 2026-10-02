from ..conversation.conversation_manager import (
    ConversationManager
)

from .microphone import (
    MicrophoneRecorder
)


def main():

    print(
        "\n" + "=" * 80
    )

    print(
        "MEDIGUIDE VOICE ASSISTANT"
    )

    print(
        "=" * 80
    )

    print(
        "\nSpeak naturally with the assistant."
    )

    print(
        "Say 'exit' to stop."
    )


    # ==================================================
    # INITIALIZE
    # ==================================================

    recorder = (
        MicrophoneRecorder()
    )

    assistant = (
        ConversationManager()
    )


    # ==================================================
    # CONVERSATION LOOP
    # ==================================================

    while True:

        # --------------------------------------------------
        # Record voice
        # --------------------------------------------------

        print(
            "\n" + "-" * 80
        )

        print(
            "🎤 YOUR TURN"
        )

        print(
            "-" * 80
        )


        audio_file = (
            recorder.record(

                duration=5,

                output_file="input.wav"
            )
        )


        # --------------------------------------------------
        # Process complete voice pipeline
        # --------------------------------------------------

        result = (
            assistant.process_voice(
                audio_file
            )
        )


        # --------------------------------------------------
        # Display transcription
        # --------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "TRANSCRIPTION"
        )

        print(
            "=" * 80
        )

        print(
            result["transcription"]
        )


        # --------------------------------------------------
        # Display answer
        # --------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "AI ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            result["answer"]
        )


        # --------------------------------------------------
        # Display retrieval information
        # --------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "BACKGROUND"
        )

        print(
            "=" * 80
        )


        print(
            "\nRetrieval Query:"
        )

        print(
            result["retrieval_query"]
        )


        print(
            "\nTop Retrieved Documents:"
        )


        for document in (
            result[
                "retrieved_documents"
            ]
        ):

            print(
                f"\n"
                f"{document['rank']}. "
                f"{document['title']}"
            )

            print(
                f"   Score: "
                f"{document['score']}"
            )

            print(
                f"   Type: "
                f"{document['document_type']}"
            )

            print(
                f"   Category: "
                f"{document['category']}"
            )


        # --------------------------------------------------
        # Display dialogue state
        # --------------------------------------------------

        print(
            "\nDialogue State:"
        )

        print(
            result[
                "dialogue_state"
            ]
        )


        # --------------------------------------------------
        # Exit detection
        # --------------------------------------------------

        if (
            result["transcription"]
            .lower()
            .strip()
            in [
                "exit",
                "quit",
                "stop"
            ]
        ):

            print(
                "\nGoodbye."
            )

            break


if __name__ == "__main__":

    main()