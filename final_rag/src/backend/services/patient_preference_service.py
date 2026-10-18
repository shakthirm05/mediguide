
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


def load_patient_preferences(user_id: str) -> dict[str, str]:
    """
    Return the authenticated patient's preferences only when consent
    has been granted. Returns an empty dictionary when consent is absent.
    Database errors are handled by the caller.
    """
    if not user_id:
        return {}

    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.granted, i.memory_key, i.memory_value
                FROM patient_profiles AS p
                JOIN patient_memory_consent AS c
                  ON c.patient_id = p.patient_id
                LEFT JOIN patient_memory_items AS i
                  ON i.patient_id = p.patient_id
                WHERE p.user_id = %s
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

    # No consent record, or consent was revoked.
    if not rows or not rows[0][0]:
        return {}

    preferences = {}

    for granted, memory_key, memory_value in rows:
        if (
            granted
            and memory_key in ALLOWED_PREFERENCE_KEYS
            and isinstance(memory_value, str)
            and memory_value.strip()
        ):
            preferences[memory_key] = memory_value.strip()[:200]

    return preferences
