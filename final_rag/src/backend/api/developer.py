
from fastapi import APIRouter, Depends

from src.backend.auth.dependencies import require_roles
from src.backend.database.connection import check_database_connection
from uuid import UUID

from fastapi import HTTPException, Request
from src.backend.database.connection import DATABASE_URL
import psycopg


router = APIRouter(
    prefix="/api/v1/developer",
    tags=["Developer Panel"],
    dependencies=[Depends(require_roles("admin"))],
)


@router.get("/overview")
def developer_overview():
    """Return service status without exposing patient information."""

    try:
        database_connected = check_database_connection()
    except Exception:
        database_connected = False

    return {
        "api": {
            "status": "ok",
            "service": "mediguide-api",
        },
        "database": {
            "status": (
                "connected" if database_connected else "unavailable"
            ),
        },
    }

def _safe_evidence_metadata(evidence):
    """Return evidence metadata without exposing retrieved document text."""
    if not isinstance(evidence, list):
        return []

    safe_items = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        safe = {}

        for key in ("id", "chunk_id", "source_id", "score", "rank"):
            value = item.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                safe[key] = value

        safe_items.append(safe)

    return safe_items


@router.get("/pipeline/{user_id}/{conversation_id}")
def inspect_pipeline(
    user_id: UUID,
    conversation_id: UUID,
    request: Request,
):
    """
    Inspect sanitized LangGraph checkpoint metadata.
    This endpoint is restricted by the router's existing admin-role guard.
    """
    graph = getattr(request.app.state, "chat_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Chat graph is not initialized.",
        )

    thread_id = f"{user_id}:{conversation_id}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = graph.get_state(config)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No checkpoint was found for this conversation.",
        )

    values = getattr(snapshot, "values", None) or {}

    if not values:
        raise HTTPException(
            status_code=404,
            detail="No checkpoint state is available for this conversation.",
        )

    slots = values.get("dialogue_slots") or {}
    evidence = values.get("evidence") or []

    if not isinstance(slots, dict):
        slots = {}

    if not isinstance(evidence, list):
        evidence = []

    return {
        "conversation_id": str(conversation_id),
        "intent": values.get("intent"),
        "dialogue_state": values.get("dialogue_state"),
        "active_workflow": values.get("active_workflow"),
        "workflow_status": values.get("workflow_status"),
        "next_nodes": list(getattr(snapshot, "next", ()) or ()),
        "dialogue_slots": {
            "filled_count": sum(
                value is not None and value != ""
                for value in slots.values()
            ),
            "slot_names": [
                key for key, value in slots.items()
                if value is not None and value != ""
            ],
        },
        "retrieval": {
            "evidence_count": len(evidence),
            "evidence": _safe_evidence_metadata(evidence),
        },
        "checkpoint_available": True,
    }


@router.get("/patient-data/{user_id}")
def get_patient_structured_data(user_id: UUID):
    """
    Read-only structured-data lookup for an existing user.
    Credentials, password hashes, claim-code hashes, and
    LangGraph checkpoint blobs are never returned.
    """
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, email, role, is_active, created_at
                    FROM app_users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                account = cur.fetchone()

                if not account:
                    raise HTTPException(
                        status_code=404,
                        detail="User not found.",
                    )

                cur.execute(
                    """
                    SELECT patient_id, patient_identifier,
                           preferred_language, created_at
                    FROM patient_profiles
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                profile_rows = cur.fetchall()

                profile = None
                patient_ids = []

                if profile_rows:
                    profile = {
                        "patient_id": str(profile_rows[0][0]),
                        "patient_identifier": profile_rows[0][1],
                        "preferred_language": profile_rows[0][2],
                        "created_at": profile_rows[0][3],
                    }
                    patient_ids = [row[0] for row in profile_rows]

                appointments = []
                complaints = []
                memory_consent = []
                memory_items = []

                for patient_id in patient_ids:
                    cur.execute(
                        """
                        SELECT appointment_id, booking_reference,
                               doctor_id, starts_at, ends_at, status,
                               reason, created_at, updated_at, cancelled_at
                        FROM appointments
                        WHERE patient_id = %s
                        ORDER BY starts_at DESC
                        LIMIT 100
                        """,
                        (patient_id,),
                    )
                    appointments.extend([
                        {
                            "appointment_id": str(r[0]),
                            "booking_reference": r[1],
                            "doctor_id": str(r[2]) if r[2] else None,
                            "starts_at": r[3],
                            "ends_at": r[4],
                            "status": r[5],
                            "reason": r[6],
                            "created_at": r[7],
                            "updated_at": r[8],
                            "cancelled_at": r[9],
                        }
                        for r in cur.fetchall()
                    ])

                    cur.execute(
                        """
                        SELECT complaint_id, complaint_reference,
                               category, description, status,
                               created_at, updated_at
                        FROM patient_complaints
                        WHERE patient_id = %s
                        ORDER BY created_at DESC
                        LIMIT 100
                        """,
                        (patient_id,),
                    )
                    complaints.extend([
                        {
                            "complaint_id": str(r[0]),
                            "complaint_reference": r[1],
                            "category": r[2],
                            "description": r[3],
                            "status": r[4],
                            "created_at": r[5],
                            "updated_at": r[6],
                        }
                        for r in cur.fetchall()
                    ])

                    cur.execute(
                        """
                        SELECT granted, granted_at, updated_at
                        FROM patient_memory_consent
                        WHERE patient_id = %s
                        """,
                        (patient_id,),
                    )
                    consent_rows = cur.fetchall()
                    memory_consent.extend([
                        {
                            "granted": r[0],
                            "granted_at": r[1],
                            "updated_at": r[2],
                        }
                        for r in consent_rows
                    ])

                    # Only read personal memory if consent is explicitly granted.
                    if any(r[0] is True for r in consent_rows):
                        cur.execute(
                            """
                            SELECT memory_id, memory_key, memory_value,
                                   created_at, updated_at
                            FROM patient_memory_items
                            WHERE patient_id = %s
                            ORDER BY updated_at DESC
                            LIMIT 100
                            """,
                            (patient_id,),
                        )
                        memory_items.extend([
                            {
                                "memory_id": str(r[0]),
                                "memory_key": r[1],
                                "memory_value": r[2],
                                "created_at": r[3],
                                "updated_at": r[4],
                            }
                            for r in cur.fetchall()
                        ])

                cur.execute(
                    """
                    SELECT conversation_id, created_at, updated_at
                    FROM conversations
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                    (user_id,),
                )
                conversations = [
                    {
                        "conversation_id": str(r[0]),
                        "created_at": r[1],
                        "updated_at": r[2],
                    }
                    for r in cur.fetchall()
                ]

        return {
            "account": {
                "user_id": str(account[0]),
                "email": account[1],
                "role": account[2],
                "is_active": account[3],
                "created_at": account[4],
            },
            "profile": profile,
            "appointments": appointments,
            "complaints": complaints,
            "memory_consent": memory_consent,
            "memory_items": memory_items,
            "conversations": conversations,
            "limits": {
                "appointments": 100,
                "complaints": 100,
                "memory_items": 100,
                "conversations": 100,
            },
        }

    except HTTPException:
        raise
    except psycopg.Error:
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve structured data.",
        )
