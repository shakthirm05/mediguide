# src/backend/api/patient_memory.py

from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src.backend.auth.dependencies import get_current_user
from src.backend.database.connection import DATABASE_URL


router = APIRouter(
    prefix="/api/v1/patient/memory",
    tags=["Patient Memory"],
)


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
    memory_value: str = Field(min_length=1, max_length=500)


def _get_patient_id(cursor, authenticated_user_id: str) -> UUID:
    """Resolve the linked patient profile from the authenticated account."""
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
    cursor.execute(
        """
        SELECT granted
        FROM patient_memory_consent
        WHERE patient_id = %s
        """,
        (patient_id,),
    )
    row = cursor.fetchone()

    if row is None or not row["granted"]:
        raise HTTPException(
            status_code=403,
            detail=(
                "Patient memory is disabled. "
                "Explicit consent is required before saving or retrieving it."
            ),
        )


@router.put("/consent")
def set_memory_consent(
    request: MemoryConsentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Enable or revoke persistent patient-memory consent."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor, str(current_user["user_id"])
            )

            cursor.execute(
                """
                INSERT INTO patient_memory_consent (
                    patient_id, granted, granted_at, updated_at
                )
                VALUES (
                    %s, %s,
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
                """,
                (
                    patient_id,
                    request.granted,
                    request.granted,
                ),
            )

            # Revoking consent removes the saved preference entries.
            if not request.granted:
                cursor.execute(
                    """
                    DELETE FROM patient_memory_items
                    WHERE patient_id = %s
                    """,
                    (patient_id,),
                )

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
    """Return only the authenticated patient's saved memory."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor, str(current_user["user_id"])
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
            enabled = bool(consent and consent["granted"])

            if not enabled:
                return {
                    "patient_memory_enabled": False,
                    "items": [],
                }

            cursor.execute(
                """
                SELECT memory_id, memory_key, memory_value,
                       created_at, updated_at
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
    """Create or update one explicitly provided patient preference."""
    memory_value = request.memory_value.strip()

    if not memory_value:
        raise HTTPException(
            status_code=422,
            detail="Memory value cannot be blank.",
        )

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor, str(current_user["user_id"])
            )
            _require_consent(cursor, patient_id)

            cursor.execute(
                """
                INSERT INTO patient_memory_items (
                    memory_id, patient_id, memory_key, memory_value
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (patient_id, memory_key)
                DO UPDATE SET
                    memory_value = EXCLUDED.memory_value,
                    updated_at = NOW()
                RETURNING memory_id, memory_key, memory_value,
                          created_at, updated_at
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
    """Delete a memory entry only if it belongs to this patient."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            patient_id = _get_patient_id(
                cursor, str(current_user["user_id"])
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

    return {"deleted": True, "memory_id": str(deleted["memory_id"])}