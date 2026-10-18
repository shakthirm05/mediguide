
import os
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from pwdlib import PasswordHash

from src.backend.auth.dependencies import get_current_user


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

password_hasher = PasswordHash.recommended()
router = APIRouter(prefix="/api/v1/patient", tags=["Patient Identity"])


class LinkPatientRequest(BaseModel):
    patient_identifier: str = Field(min_length=1, max_length=50)
    claim_code: str = Field(min_length=1, max_length=200)


class LinkPatientResponse(BaseModel):
    status: str
    patient_identifier: str


@router.post("/link", response_model=LinkPatientResponse)
def link_patient(
    request: LinkPatientRequest,
    current_user: dict = Depends(get_current_user),
):
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="Database configuration is unavailable.",
        )

    user_id = UUID(str(current_user["user_id"]))
    patient_identifier = request.patient_identifier.strip()

    try:
        # The connection context commits only when the entire operation succeeds.
        # Any exception inside it rolls back both the registry update and profile
        # insertion.
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # A single account can have only one patient profile.
                cur.execute(
                    """
                    SELECT 1
                    FROM patient_profiles
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                if cur.fetchone():
                    raise HTTPException(
                        status_code=409,
                        detail="This account already has a linked patient profile.",
                    )

                # Serialize attempts to claim the same patient record.
                cur.execute(
                    """
                    SELECT claim_code_hash, linked_user_id
                    FROM patient_registry
                    WHERE patient_identifier = %s
                    FOR UPDATE
                    """,
                    (patient_identifier,),
                )
                registry_row = cur.fetchone()

                if registry_row is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid patient identifier or claim code.",
                    )

                stored_hash, linked_user_id = registry_row

                # Verify before revealing whether the record has been claimed.
                try:
                    code_is_valid = password_hasher.verify(
                        request.claim_code,
                        stored_hash,
                    )
                except Exception:
                    code_is_valid = False

                if not code_is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid patient identifier or claim code.",
                    )

                if linked_user_id is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="This patient record has already been linked.",
                    )

                # Conditional update provides an additional database-level guard.
                cur.execute(
                    """
                    UPDATE patient_registry
                    SET linked_user_id = %s,
                        linked_at = NOW()
                    WHERE patient_identifier = %s
                      AND linked_user_id IS NULL
                    RETURNING patient_identifier
                    """,
                    (user_id, patient_identifier),
                )

                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=409,
                        detail="This patient record has already been linked.",
                    )

                # Create the patient's profile in the same transaction.
                cur.execute(
                    """
                    INSERT INTO patient_profiles (
                        patient_id,
                        user_id,
                        patient_identifier,
                        preferred_language
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (uuid4(), user_id, patient_identifier, "en"),
                )

        return LinkPatientResponse(
            status="linked",
            patient_identifier=patient_identifier,
        )

    except HTTPException:
        raise
    except psycopg.errors.UniqueViolation:
        # A uniqueness constraint remains the final concurrency safeguard.
        raise HTTPException(
            status_code=409,
            detail="This account or patient record is already linked.",
        )
    except psycopg.Error:
        raise HTTPException(
            status_code=503,
            detail="Patient linking is temporarily unavailable.",
        )
