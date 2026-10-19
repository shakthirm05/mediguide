
# src/backend/api/speech.py

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

from src.backend.auth.dependencies import get_current_user

import io
import wave
from pathlib import Path

from fastapi import Response
from pydantic import BaseModel, Field
from piper import PiperVoice

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/speech",
    tags=["Speech"],
)

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".webm",
    ".ogg",
    ".flac",
}


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """Load one local Whisper model and reuse it across requests."""
    model_name = os.getenv("WHISPER_MODEL", "base")

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )


@router.post("/transcribe")
def transcribe_audio(
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Transcribe an uploaded audio file locally.

    Authentication is required. Audio is temporarily written to disk
    for transcription and removed afterward.
    """
    suffix = Path(audio.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. Use WAV, MP3, M4A, "
                "WebM, OGG, or FLAC."
            ),
        )

    temp_path = None

    try:
        audio_bytes = audio.file.read(MAX_AUDIO_BYTES + 1)

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds the 20 MB limit.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        model = get_whisper_model()

        segments, info = model.transcribe(
            temp_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=(
                "MediGuide hospital assistant. "
                "Hospital departments include Radiology, Cardiology, "
                "Neurology, Orthopedics, and Pediatrics. "
                "Common questions include: Where is the Radiology Department? "
                "How do I book an appointment?"
            ),
        )

        # Consume the generator while the temporary file still exists.
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech could be recognized in the audio.",
            )

        return {
            "text": transcript,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Local Whisper transcription failed.")
        raise HTTPException(
            status_code=500,
            detail="Speech transcription failed.",
        )
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not remove temporary audio file."
                )

        audio.file.close()

# -----------------------------
# Text-to-Speech (Piper)
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PIPER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "en_US-lessac-medium.onnx"
)


class SpeechSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@lru_cache(maxsize=1)
def get_piper_voice() -> PiperVoice:
    """Load the local Piper voice once and reuse it."""
    if not PIPER_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Piper voice model not found: {PIPER_MODEL_PATH}"
        )

    return PiperVoice.load(str(PIPER_MODEL_PATH))


@router.post(
    "/synthesize",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/wav": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "description": "Generated WAV audio",
        }
    },
)
def synthesize_speech(
    request: SpeechSynthesisRequest,
    current_user: dict = Depends(get_current_user),
):
    """Convert text into WAV audio using the local Piper model."""
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty or whitespace.",
        )

    try:
        voice = get_piper_voice()

        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        audio_bytes = audio_buffer.getvalue()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    'inline; filename="mediguide_response.wav"'
                )
            },
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Piper text-to-speech generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Speech synthesis failed.",
        )

