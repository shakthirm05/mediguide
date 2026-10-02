import streamlit as st

from src.conversation.conversation_manager import ConversationManager
from src.voice.microphone import MicrophoneRecorder


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="MediGuide General Hospital",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# LOAD BACKEND
# --------------------------------------------------

@st.cache_resource
def load_assistant():
    return ConversationManager()


@st.cache_resource
def load_recorder():
    return MicrophoneRecorder()


assistant = load_assistant()
recorder = load_recorder()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏥 MediGuide Intelligent Cognitive Robot for Personalized Healthcare Assistance")
st.subheader("AI Patient Assistant")

st.divider()


# ==================================================
# VOICE + QUESTION & ANSWER
# ==================================================

voice_col, text_col = st.columns(2)


# --------------------------------------------------
# VOICE SECTION
# --------------------------------------------------

with voice_col:

    st.header("🎤 Voice Assistant")

    st.write(
        "Speak to MediGuide using your voice."
    )

    if st.button(
        "🎙️ Record 5 Seconds",
        use_container_width=True
    ):

        with st.spinner("Listening..."):

            audio_file = recorder.record(
                duration=5,
                output_file="input.wav"
            )

        with st.spinner("MediGuide is processing..."):

            result = assistant.process_voice(
                audio_file
            )

        st.session_state["result"] = result

        st.success("Voice processed successfully.")

    # Show voice result if the latest result is voice
    result = st.session_state["result"]

    if result and result.get("input_type") == "voice":

        st.subheader("You said")

        st.info(
            result.get("transcription", "")
        )

        st.subheader("AI Answer")

        st.write(
            result.get("answer", "")
        )

        st.caption(
            "The answer was also spoken through Piper TTS."
        )


# --------------------------------------------------
# TEXT Q&A SECTION
# --------------------------------------------------

with text_col:

    st.header("💬 Question & Answer")

    question = st.text_input(
        "Ask your question",
        placeholder=(
            "Example: What do I have to carry "
            "for an appointment?"
        )
    )

    if st.button(
        "Ask",
        type="primary",
        use_container_width=True
    ):

        if question.strip():

            with st.spinner(
                "MediGuide is thinking..."
            ):

                result = assistant.process_message(
                    question
                )

            st.session_state["result"] = result

        else:

            st.warning(
                "Please enter a question."
            )

    # Show text result if latest result is text
    result = st.session_state["result"]

    if result and result.get("input_type") == "text":

        st.subheader("AI Answer")

        st.write(
            result.get("answer", "")
        )


# ==================================================
# BACKGROUND
# ==================================================

result = st.session_state["result"]

if result:

    st.divider()

    st.header("🔬 Background")


    # --------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------

    if result.get("input_type") == "voice":

        st.subheader("Voice Transcription")

        st.code(
            result.get("transcription", ""),
            language="text"
        )


    # --------------------------------------------------
    # RETRIEVAL QUERY
    # --------------------------------------------------

    st.subheader("Retrieval Query")

    retrieval_query = result.get(
        "retrieval_query"
    )

    if retrieval_query:

        st.code(
            retrieval_query,
            language="text"
        )

    else:

        st.info(
            "No retrieval was performed for this response."
        )


    # --------------------------------------------------
    # TOP 5 RETRIEVED DOCUMENTS
    # --------------------------------------------------

    st.subheader(
        "Top 5 Retrieved Documents"
    )

    documents = result.get(
        "retrieved_documents",
        []
    )

    if documents:

        for document in documents:

            title = document.get(
                "title",
                "Unknown document"
            )

            score = document.get(
                "score",
                0
            )

            with st.expander(
                f"{document['rank']}. "
                f"{title} "
                f"— Score: {score}"
            ):

                st.write(
                    f"**Document Type:** "
                    f"{document.get('document_type', '')}"
                )

                st.write(
                    f"**Category:** "
                    f"{document.get('category', '')}"
                )

                st.write(
                    f"**Chunk ID:** "
                    f"{document.get('chunk_id', '')}"
                )

                st.write(
                    f"**Similarity Score:** "
                    f"{score}"
                )

                st.write("**Content:**")

                st.write(
                    document.get("content", "")
                )

    else:

        st.info(
            "No documents were retrieved."
        )


    # --------------------------------------------------
    # DIALOGUE STATE
    # --------------------------------------------------

    st.subheader("Dialogue State")

    st.json(
        result.get(
            "dialogue_state",
            {}
        )
    )