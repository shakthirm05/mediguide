
import uuid
import secrets
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from src.backend.database.connection import DATABASE_URL


class ComplaintService:
    """Handles authenticated patients' service complaints."""

    def _connect(self):
        return psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
            row_factory=dict_row,
        )

    @staticmethod
    def _get_patient_id(cursor, user_id: str):
        cursor.execute(
            """
            SELECT patient_id
            FROM patient_profiles
            WHERE user_id = %s
            """,
            (user_id,),
        )
        patient = cursor.fetchone()

        if not patient:
            raise ValueError(
                "No patient profile is associated with this account."
            )

        return patient["patient_id"]

    @staticmethod
    def _generate_reference():
        return f"MG-CMP-{secrets.token_hex(4).upper()}"

    def create_complaint(
        self,
        user_id: str,
        description: str,
        category: str = "general",
    ):
        description = description.strip()
        category = category.strip().lower()

        if not 5 <= len(description) <= 3000:
            raise ValueError(
                "Complaint description must contain 5–3000 characters."
            )

        allowed_categories = {
            "general",
            "staff",
            "billing",
            "facilities",
            "appointment",
            "waiting_time",
            "other",
        }

        if category not in allowed_categories:
            raise ValueError(
                f"Invalid category. Choose from: "
                f"{', '.join(sorted(allowed_categories))}"
            )

        with self._connect() as conn:
            with conn.cursor() as cursor:
                patient_id = self._get_patient_id(cursor, user_id)

                complaint = None

                # Retry if the randomly generated reference already exists.
                for _ in range(5):
                    reference = self._generate_reference()

                    cursor.execute(
                        """
                        INSERT INTO patient_complaints (
                            complaint_id,
                            complaint_reference,
                            patient_id,
                            category,
                            description
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (complaint_reference) DO NOTHING
                        RETURNING
                            complaint_id,
                            complaint_reference,
                            category,
                            description,
                            status,
                            created_at,
                            updated_at
                        """,
                        (
                            uuid.uuid4(),
                            reference,
                            patient_id,
                            category,
                            description,
                        ),
                    )

                    complaint = cursor.fetchone()

                    if complaint:
                        break

                if not complaint:
                    raise RuntimeError(
                        "Could not generate a unique complaint reference."
                    )

                return complaint

    def list_complaints(self, user_id: str):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                patient_id = self._get_patient_id(cursor, user_id)

                cursor.execute(
                    """
                    SELECT
                        complaint_id,
                        complaint_reference,
                        category,
                        description,
                        status,
                        created_at,
                        updated_at
                    FROM patient_complaints
                    WHERE patient_id = %s
                    ORDER BY created_at DESC
                    """,
                    (patient_id,),
                )

                return cursor.fetchall()

    def get_complaint(self, user_id: str, complaint_reference: str):
        with self._connect() as conn:
            with conn.cursor() as cursor:
                patient_id = self._get_patient_id(cursor, user_id)

                cursor.execute(
                    """
                    SELECT
                        complaint_id,
                        complaint_reference,
                        category,
                        description,
                        status,
                        created_at,
                        updated_at
                    FROM patient_complaints
                    WHERE patient_id = %s
                      AND complaint_reference = %s
                    """,
                    (patient_id, complaint_reference.strip().upper()),
                )

                return cursor.fetchone()


complaint_service = ComplaintService()
