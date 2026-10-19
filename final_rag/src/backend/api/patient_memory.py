
# src/backend/api/patient_memory.py

from typing import Literal
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from psycopg.rows import dict_row

from src.backend.auth.dependencies import get_current_user
from src.backend.database.connection import DATABASE_URL


router = APIRouter(
    prefix="/api/v1/patient/memory",
    tags=["Patient Memory"],
)

MAX_PREFERENCE_LENGTH = 200

ALLOWED_PREFERENCE_KEYS = {
    "preferred_language",
    "response_style",
    "accessibility_support",
    "communication_preference",
    "preferred_route",
}


class MemoryConsentRequest(BaseModel):
    granted: bool


class MemoryItemRequest(BaseModel):
    memory_key: Literal[
        "preferred_language",
        "response_style",
        "accessibility_support",
        "communication_preference",
        "preferred_route",
    ]
    memory_value: str = Field(
        min_length=1,
        max_length=MAX_PREFERENCE_LENGTH,
    )


def _get_patient_id(cursor, authenticated_user_id: str) -> UUID:
    """Resolve the patient profile belonging to the authenticated user."""
    cursor.execute(
        """
        SELECT patient_id
        FROM patient_profiles
        WHERE user_id = %s
        """,
        (authenticated_user_id,),
    )
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=409,
            detail="Link your patient profile before using patient memory.",
        )

    return row["patient_id"]


def _require_consent(cursor, patient_id: UUID) -> None:
    """
    Require explicit consent and lock its row until the transaction ends.

    Consent revocation uses the same row, so a concurrent save cannot
    insert a preference after revocation has completed.
    """
    cursor.execute(
        """
        SELECT granted
        FROM patient_memory_consent
        WHERE patient_id = %s
        FOR UPDATE
        """,
        (patient_id,),
    )
    row = cursor.fetchone()

    if row is None or row["granted"] is not True:
        raise HTTPException(
            status_code=403,
            detail=(
                "Patient memory is disabled. Explicit consent is required "
                "before saving or retrieving it."
            ),
        )


@router.put("/consent")
def set_memory_consent(
    request: MemoryConsentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Grant or revoke persistent patient-memory consent."""
    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor,
                str(current_user["user_id"]),
            )

            # INSERT or UPDATE locks the consent row for this transaction.
            cursor.execute(
                """
                INSERT INTO patient_memory_consent (
                    patient_id,
                    granted,
                    granted_at,
                    updated_at
                )
                VALUES (
                    %s,
                    %s,
                    CASE WHEN %s THEN NOW() ELSE NULL END,
                    NOW()
                )
                ON CONFLICT (patient_id)
                DO UPDATE SET
                    granted = EXCLUDED.granted,
                    granted_at = CASE
                        WHEN EXCLUDED.granted THEN NOW()
                        ELSE NULL
                    END,
                    updated_at = NOW()
                RETURNING granted
                """,
                (
                    patient_id,
                    request.granted,
                    request.granted,
                ),
            )
            cursor.fetchone()

            if not request.granted:
                # Delete all stored preferences in the same transaction.
                cursor.execute(
                    """
                    DELETE FROM patient_memory_items
                    WHERE patient_id = %s
                    """,
                    (patient_id,),
                )

    # The connection context commits before execution reaches this return.
    return {
        "patient_memory_enabled": request.granted,
        "message": (
            "Patient memory enabled."
            if request.granted
            else "Patient memory disabled and saved preferences deleted."
        ),
    }


@router.get("")
def get_patient_memory(
    current_user: dict = Depends(get_current_user),
):
    """Return the authenticated patient's preferences when consented."""
    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor,
                str(current_user["user_id"]),
            )

            cursor.execute(
                """
                SELECT granted
                FROM patient_memory_consent
                WHERE patient_id = %s
                """,
                (patient_id,),
            )
            consent = cursor.fetchone()

            if consent is None or consent["granted"] is not True:
                return {
                    "patient_memory_enabled": False,
                    "items": [],
                }

            cursor.execute(
                """
                SELECT
                    memory_id,
                    memory_key,
                    memory_value,
                    created_at,
                    updated_at
                FROM patient_memory_items
                WHERE patient_id = %s
                ORDER BY memory_key
                """,
                (patient_id,),
            )
            items = cursor.fetchall()

    return {
        "patient_memory_enabled": True,
        "items": [
            {
                **item,
                "memory_id": str(item["memory_id"]),
            }
            for item in items
        ],
    }


@router.put("")
def save_patient_memory(
    request: MemoryItemRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create or update one allowlisted preference with explicit consent."""
    memory_value = request.memory_value.strip()

    if not memory_value:
        raise HTTPException(
            status_code=422,
            detail="Memory value cannot be blank.",
        )

    # Keep the saved value consistent with the RAG preference loader.
    memory_value = memory_value[:MAX_PREFERENCE_LENGTH]

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor,
                str(current_user["user_id"]),
            )

            # The consent row remains locked until this transaction commits.
            _require_consent(cursor, patient_id)

            cursor.execute(
                """
                INSERT INTO patient_memory_items (
                    memory_id,
                    patient_id,
                    memory_key,
                    memory_value
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (patient_id, memory_key)
                DO UPDATE SET
                    memory_value = EXCLUDED.memory_value,
                    updated_at = NOW()
                RETURNING
                    memory_id,
                    memory_key,
                    memory_value,
                    created_at,
                    updated_at
                """,
                (
                    uuid4(),
                    patient_id,
                    request.memory_key,
                    memory_value,
                ),
            )
            item = cursor.fetchone()

    return {
        **item,
        "memory_id": str(item["memory_id"]),
        "message": "Patient preference saved.",
    }


@router.delete("/{memory_id}")
def delete_patient_memory(
    memory_id: UUID,
    current_user: dict = Depends(get_current_user),
):
    """Delete an entry only when it belongs to the authenticated patient."""
    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor,
                str(current_user["user_id"]),
            )

            cursor.execute(
                """
                DELETE FROM patient_memory_items
                WHERE memory_id = %s
                  AND patient_id = %s
                RETURNING memory_id
                """,
                (memory_id, patient_id),
            )
            deleted = cursor.fetchone()

    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail="Memory entry not found.",
        )

    return {
        "deleted": True,
        "memory_id": str(deleted["memory_id"]),
    }
