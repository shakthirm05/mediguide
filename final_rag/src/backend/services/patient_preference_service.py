
import logging

import psycopg

from src.backend.database.connection import DATABASE_URL

logger = logging.getLogger(__name__)

ALLOWED_PREFERENCE_KEYS = {
    "preferred_language",
    "response_style",
    "accessibility_support",
    "communication_preference",
    "preferred_route",
}

MAX_PREFERENCE_LENGTH = 200


def load_patient_preferences(user_id: str) -> dict[str, str]:
    """
    Load allowlisted preferences for the authenticated user only
    when exactly one consent record exists and consent is granted.

    Returns an empty dictionary when consent is absent, revoked,
    or ambiguous. Database errors propagate to the caller.
    """
    if not user_id:
        return {}

    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        with conn.cursor() as cursor:
            # Check consent separately from the preference rows.
            cursor.execute(
                """
                SELECT c.granted
                FROM patient_profiles AS p
                JOIN patient_memory_consent AS c
                  ON c.patient_id = p.patient_id
                WHERE p.user_id = %s
                """,
                (user_id,),
            )
            consent_rows = cursor.fetchall()

            # Missing or duplicate consent records are not treated
            # as permission to access personal memory.
            if len(consent_rows) != 1:
                if len(consent_rows) > 1:
                    logger.warning(
                        "Ambiguous memory consent records for user."
                    )
                return {}

            if consent_rows[0][0] is not True:
                return {}

            cursor.execute(
                """
                SELECT i.memory_key, i.memory_value
                FROM patient_profiles AS p
                JOIN patient_memory_items AS i
                  ON i.patient_id = p.patient_id
                WHERE p.user_id = %s
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

    preferences: dict[str, str] = {}

    for memory_key, memory_value in rows:
        if (
            memory_key in ALLOWED_PREFERENCE_KEYS
            and isinstance(memory_value, str)
            and memory_value.strip()
        ):
            preferences[memory_key] = memory_value.strip()[
                :MAX_PREFERENCE_LENGTH
            ]

    return preferences
