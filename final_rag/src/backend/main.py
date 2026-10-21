from contextlib import asynccontextmanager
import logging
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field

from src.backend.agents.chat_graph import build_chat_graph
from src.backend.api.auth import router as auth_router
from src.backend.api.patient_identity import router as patient_identity_router
from src.backend.api.appointments import router as appointments_router
from src.backend.auth.dependencies import get_current_user
from src.backend.database.connection import DATABASE_URL, check_database_connection
from src.backend.services.rag_service import get_rag_service
from src.backend.api.patient_memory import router as patient_memory_router
from src.backend.api.complaints import router as complaints_router
from src.backend.api.speech import router as speech_router
import base64
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from src.backend.api.speech import (
    router as speech_router,
    transcribe_audio,
    synthesize_speech,
    SpeechSynthesisRequest,
)
from src.backend.api.conversations import router as conversations_router
from src.backend.api.developer import router as developer_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and release application resources."""
    service = get_rag_service()
    app.state.rag_service = service
    app.state.chat_graph = None

    try:
        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            checkpointer.setup()
            app.state.chat_graph = build_chat_graph(service, checkpointer)
            logger.info("MediGuide chat graph initialized.")
            yield
    finally:
        app.state.chat_graph = None
        service.close()


app = FastAPI(
    title="MediGuide API",
    description="Backend API for the MediGuide hospital assistance system.",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(patient_identity_router)
app.include_router(appointments_router)
app.include_router(patient_memory_router)
app.include_router(complaints_router)
app.include_router(speech_router)
app.include_router(conversations_router)
app.include_router(developer_router)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "mediguide-api"}


@app.get("/health/db", tags=["Health"])
def database_health():
    try:
        check_database_connection()
        return {"status": "ok", "database": "connected"}
    except Exception:
        logger.exception("Database health check failed.")
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        )


@app.post("/api/v1/chat", tags=["Chat"])
def chat(request: AskRequest, current_user: dict = Depends(get_current_user)):
    graph = getattr(app.state, "chat_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="The chat workflow is not initialized.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    conversation_id = request.conversation_id
    user_id = str(current_user["user_id"])

    # Separate checkpoint namespace for every authenticated user.
    # Keep conversation_id unchanged for ownership/database operations.
    thread_id = f"{user_id}:{conversation_id}"

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversations (conversation_id, user_id)
                    VALUES (%s, %s)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                    (conversation_id, user_id),
                )

                cursor.execute(
                    """
                    SELECT user_id
                    FROM conversations
                    WHERE conversation_id = %s
                    FOR UPDATE
                    """,
                    (conversation_id,),
                )
                owner = cursor.fetchone()

                if owner is None or str(owner[0]) != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have access to this conversation.",
                    )

                cursor.execute(
                    """
                    UPDATE conversations
                    SET updated_at = NOW()
                    WHERE conversation_id = %s
                    """,
                    (conversation_id,),
                )

        # Only the verified owner reaches graph.invoke().
        result = graph.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
                "user_id": user_id,
                "conversation_id": str(conversation_id),
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        # Diagnostic boundary: determine whether the ENT answer is already
        # present in the graph result or introduced while building the API response.
        graph_answer = result.get("answer", "")
        logger.warning(
            "MEDIGUIDE DEBUG | stage=graph_output | conversation_id=%s | "
            "intent=%r | workflow_status=%r | answer=%r | entities=%r | "
            "active_workflow=%r | dialogue_slots=%r",
            conversation_id,
            result.get("intent"),
            result.get("workflow_status"),
            graph_answer,
            result.get("entities"),
            result.get("active_workflow"),
            result.get("dialogue_slots"),
        )

        response = {
            "conversation_id": str(conversation_id),
            "answer": graph_answer,
            "intent": result.get("intent", "unknown"),
            "workflow_status": result.get("workflow_status", "in_progress"),
            "evidence": result.get("evidence", []),
        }

        logger.warning(
            "MEDIGUIDE DEBUG | stage=http_response | conversation_id=%s | answer=%r",
            conversation_id,
            response["answer"],
        )
        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat workflow failed.")
        raise HTTPException(
            status_code=503,
            detail="MediGuide could not process the request.",
        )

@app.post("/api/v1/speech/voice-chat", tags=["Speech"])
def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: UUID = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Transcribe audio, process it through MediGuide chat, and synthesize the answer."""

    # 1. Convert uploaded audio into text using the existing Whisper endpoint logic.
    transcription = transcribe_audio(
        audio=audio,
        current_user=current_user,
    )

    transcript = transcription["text"].strip()

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="No speech could be recognized.",
        )

    # 2. Reuse the existing authenticated chat workflow.
    # This preserves conversation ownership checks, LangGraph,
    # RAG, patient memory, and appointment workflows.
    chat_result = chat(
        AskRequest(
            question=transcript,
            conversation_id=conversation_id,
        ),
        current_user=current_user,
    )

    answer = str(chat_result.get("answer") or "").strip()

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="MediGuide did not generate a response.",
        )

    # 3. Convert the actual MediGuide answer into WAV audio.
    audio_response = synthesize_speech(
        SpeechSynthesisRequest(text=answer),
        current_user=current_user,
    )

    # 4. Return the transcript, answer, metadata, and playable audio bytes.
    return {
        "conversation_id": str(conversation_id),
        "transcript": transcript,
        "language": transcription.get("language"),
        "answer": answer,
        "intent": chat_result.get("intent", "unknown"),
        "workflow_status": chat_result.get(
            "workflow_status", "in_progress"
        ),
        "evidence": chat_result.get("evidence", []),
        "audio_mime_type": "audio/wav",
        "audio_base64": base64.b64encode(
            audio_response.body
        ).decode("ascii"),
    }
