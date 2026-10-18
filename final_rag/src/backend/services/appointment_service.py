    
from __future__ import annotations

import secrets
import string
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import psycopg
from psycopg.rows import dict_row

from src.backend.database.connection import DATABASE_URL


class AppointmentError(Exception):
    """Base exception for appointment operations."""


class PatientProfileNotLinked(AppointmentError):
    pass


class ResourceNotFound(AppointmentError):
    pass


class SlotUnavailable(AppointmentError):
    pass


class InvalidAppointmentRequest(AppointmentError):
    pass


class AppointmentService:
    """PostgreSQL-backed appointment operations."""

    ACTIVE_STATUS = "booked"

    def _connect(self):
        return psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
            row_factory=dict_row,
        )

    @staticmethod
    def _as_uuid(value: str | UUID, field: str) -> UUID:
        try:
            return value if isinstance(value, UUID) else UUID(value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidAppointmentRequest(
                f"Invalid {field}."
            ) from exc

    @staticmethod
    def _as_date(value: str | date) -> date:
        try:
            return value if isinstance(value, date) and not isinstance(
                value, datetime
            ) else date.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise InvalidAppointmentRequest(
                "The appointment date must use YYYY-MM-DD."
            ) from exc

    @staticmethod
    def _as_time(value: str | time) -> time:
        try:
            result = value if isinstance(value, time) else time.fromisoformat(value)

            if result.tzinfo is not None:
                raise InvalidAppointmentRequest(
                    "Provide a local appointment time without a timezone."
                )

            return result
        except (ValueError, TypeError) as exc:
            raise InvalidAppointmentRequest(
                "The appointment time must use HH:MM."
            ) from exc

    @staticmethod
    def _zone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise AppointmentError(
                "The configured department timezone is invalid."
            ) from exc

    @staticmethod
    def _booking_reference() -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "MG-" + "".join(secrets.choice(alphabet) for _ in range(12))

    @staticmethod
    def _appointment_dict(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)

        for field in (
            "appointment_id",
            "patient_id",
            "doctor_id",
            "department_id",
        ):
            if result.get(field) is not None:
                result[field] = str(result[field])

        for field in (
            "starts_at",
            "ends_at",
            "created_at",
            "updated_at",
            "cancelled_at",
        ):
            if result.get(field) is not None:
                result[field] = result[field].isoformat()

        return result

    def _patient_id(self, cursor, authenticated_user_id: str | UUID) -> UUID:
        user_id = self._as_uuid(authenticated_user_id, "authenticated user ID")

        cursor.execute(
            """
            SELECT patient_id
            FROM patient_profiles
            WHERE user_id = %s
            """,
            (user_id,),
        )
        row = cursor.fetchone()

        if row is None:
            raise PatientProfileNotLinked(
                "The authenticated account has no linked patient profile."
            )

        return row["patient_id"]

    def list_departments(self) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT department_id, name, location_code, floor_label
                FROM departments
                WHERE is_active = TRUE
                ORDER BY name
                """
            )
            rows = cur.fetchall()

        return [
            {**row, "department_id": str(row["department_id"])}
            for row in rows
        ]
    
    def find_department(self, name: str) -> dict[str, Any] | None:
        """Find an active department by its full name, case-insensitively."""
        if not isinstance(name, str) or not name.strip():
            return None

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT department_id, name, location_code, floor_label
                FROM departments
                WHERE is_active = TRUE
                  AND LOWER(name) = LOWER(%s)
                LIMIT 1
                """,
                (name.strip(),),
            )
            row = cur.fetchone()

        if row is None:
            return None

        return {
            **row,
            "department_id": str(row["department_id"]),
        }

    def list_doctors(
        self,
        department_id: str | UUID | None = None,
    ) -> list[dict[str, Any]]:
        parsed_department_id = (
            self._as_uuid(department_id, "department ID")
            if department_id is not None
            else None
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.doctor_id,
                    d.display_name,
                    dep.department_id,
                    dep.name AS department_name
                FROM doctors d
                JOIN departments dep
                  ON dep.department_id = d.department_id
                WHERE d.is_active = TRUE
                  AND dep.is_active = TRUE
                  AND (%s::uuid IS NULL OR d.department_id = %s)
                ORDER BY dep.name, d.display_name
                """,
                (parsed_department_id, parsed_department_id),
            )
            rows = cur.fetchall()

        return [
            {
                **row,
                "doctor_id": str(row["doctor_id"]),
                "department_id": str(row["department_id"]),
            }
            for row in rows
        ]

    def _doctor_details(self, cursor, doctor_id: UUID) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT
                d.doctor_id,
                d.display_name,
                d.is_active AS doctor_active,
                dep.department_id,
                dep.name AS department_name,
                dep.is_active AS department_active,
                dep.timezone_name
            FROM doctors d
            JOIN departments dep
              ON dep.department_id = d.department_id
            WHERE d.doctor_id = %s
            """,
            (doctor_id,),
        )

        row = cursor.fetchone()

        if row is None:
            raise ResourceNotFound("The requested doctor does not exist.")

        if not row["doctor_active"] or not row["department_active"]:
            raise ResourceNotFound(
                "The requested doctor or department is inactive."
            )

        # Validate timezone configuration before using it.
        self._zone(row["timezone_name"])
        return row

    def _get_slots(
        self,
        cursor,
        doctor_id: UUID,
        requested_date: date,
    ) -> list[dict[str, Any]]:
        """Calculate slots from active schedules and booked appointments."""
        doctor = self._doctor_details(cursor, doctor_id)
        zone = self._zone(doctor["timezone_name"])

        local_day_start = datetime.combine(
            requested_date, time.min, tzinfo=zone
        )
        local_day_end = local_day_start + timedelta(days=1)

        cursor.execute(
            """
            SELECT
                start_time,
                end_time,
                slot_duration_minutes
            FROM doctor_schedules
            WHERE doctor_id = %s
              AND weekday = %s
              AND is_active = TRUE
              AND valid_from <= %s
              AND (valid_until IS NULL OR valid_until >= %s)
            ORDER BY start_time
            """,
            (
                doctor_id,
                requested_date.isoweekday(),
                requested_date,
                requested_date,
            ),
        )
        schedules = cursor.fetchall()

        cursor.execute(
            """
            SELECT starts_at, ends_at
            FROM appointments
            WHERE doctor_id = %s
              AND status = 'booked'
              AND starts_at < %s
              AND ends_at > %s
            ORDER BY starts_at
            """,
            (
                doctor_id,
                local_day_end.astimezone(timezone.utc),
                local_day_start.astimezone(timezone.utc),
            ),
        )
        booked = cursor.fetchall()

        now = datetime.now(timezone.utc)
        slots: list[dict[str, Any]] = []

        for schedule in schedules:
            current = datetime.combine(
                requested_date,
                schedule["start_time"],
                tzinfo=zone,
            )
            schedule_end = datetime.combine(
                requested_date,
                schedule["end_time"],
                tzinfo=zone,
            )
            duration = timedelta(
                minutes=schedule["slot_duration_minutes"]
            )

            while current + duration <= schedule_end:
                slot_end = current + duration
                start_utc = current.astimezone(timezone.utc)
                end_utc = slot_end.astimezone(timezone.utc)

                # Exclude past slots and slots overlapping an active booking.
                is_booked = any(
                    existing["starts_at"] < end_utc
                    and existing["ends_at"] > start_utc
                    for existing in booked
                )

                if start_utc > now and not is_booked:
                    slots.append({
                        "doctor_id": str(doctor_id),
                        "doctor_name": doctor["display_name"],
                        "department_name": doctor["department_name"],
                        "date": requested_date.isoformat(),
                        "starts_at": current.isoformat(),
                        "ends_at": slot_end.isoformat(),
                        "timezone": doctor["timezone_name"],
                    })

                current = slot_end

        slots.sort(key=lambda item: item["starts_at"])
        return slots

    def get_availability(
        self,
        doctor_id: str | UUID,
        requested_date: str | date,
    ) -> list[dict[str, Any]]:
        doctor_uuid = self._as_uuid(doctor_id, "doctor ID")
        day = self._as_date(requested_date)

        with self._connect() as conn, conn.cursor() as cur:
            return self._get_slots(cur, doctor_uuid, day)

    def book(
        self,
        authenticated_user_id: str | UUID,
        doctor_id: str | UUID,
        starts_at: str | datetime,
        reason: str | None = None,
    ) -> dict[str, Any]:
        doctor_uuid = self._as_uuid(doctor_id, "doctor ID")

        with self._connect() as conn, conn.cursor() as cur:
            patient_id = self._patient_id(cur, authenticated_user_id)
            doctor = self._doctor_details(cur, doctor_uuid)
            zone = self._zone(doctor["timezone_name"])

            try:
                requested_start = (
                    starts_at
                    if isinstance(starts_at, datetime)
                    else datetime.fromisoformat(starts_at)
                )
            except (ValueError, TypeError) as exc:
                raise InvalidAppointmentRequest(
                    "Provide a valid appointment start datetime."
                ) from exc

            if requested_start.tzinfo is None:
                requested_start = requested_start.replace(tzinfo=zone)

            requested_start = requested_start.astimezone(timezone.utc)
            local_start = requested_start.astimezone(zone)

            if requested_start <= datetime.now(timezone.utc):
                raise InvalidAppointmentRequest(
                    "Appointments must be booked in the future."
                )

            # Serialize booking attempts for this doctor.
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(doctor_uuid),),
            )

            slots = self._get_slots(cur, doctor_uuid, local_start.date())
            matching_slot = next(
                (
                    slot for slot in slots
                    if datetime.fromisoformat(
                        slot["starts_at"]
                    ).astimezone(timezone.utc) == requested_start
                ),
                None,
            )

            if matching_slot is None:
                raise SlotUnavailable(
                    "That slot is unavailable. Please select a current "
                    "available slot."
                )

            requested_end = datetime.fromisoformat(
                matching_slot["ends_at"]
            ).astimezone(timezone.utc)

            appointment_id = uuid4()
            reference = self._booking_reference()

            cur.execute(
                """
                INSERT INTO appointments (
                    appointment_id,
                    booking_reference,
                    patient_id,
                    doctor_id,
                    starts_at,
                    ends_at,
                    status,
                    reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'booked', %s)
                RETURNING
                    appointment_id, booking_reference, patient_id,
                    doctor_id, starts_at, ends_at, status, reason,
                    created_at, updated_at
                """,
                (
                    appointment_id,
                    reference,
                    patient_id,
                    doctor_uuid,
                    requested_start,
                    requested_end,
                    reason.strip() if isinstance(reason, str) and reason.strip()
                    else None,
                ),
            )
            row = cur.fetchone()

            cur.execute(
                """
                SELECT name AS department_name
                FROM departments
                WHERE department_id = %s
                """,
                (doctor["department_id"],),
            )
            row["department_name"] = cur.fetchone()["department_name"]
            row["doctor_name"] = doctor["display_name"]

        return self._appointment_dict(row)

    def list_my_appointments(
        self,
        authenticated_user_id: str | UUID,
        include_past: bool = False,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            patient_id = self._patient_id(cur, authenticated_user_id)

            cur.execute(
                """
                SELECT
                    a.appointment_id,
                    a.booking_reference,
                    a.patient_id,
                    a.doctor_id,
                    a.starts_at,
                    a.ends_at,
                    a.status,
                    a.reason,
                    a.created_at,
                    a.updated_at,
                    a.cancelled_at,
                    d.display_name AS doctor_name,
                    dep.department_id,
                    dep.name AS department_name,
                    dep.location_code,
                    dep.floor_label
                FROM appointments a
                JOIN doctors d ON d.doctor_id = a.doctor_id
                JOIN departments dep ON dep.department_id = d.department_id
                WHERE a.patient_id = %s
                  AND (%s OR a.starts_at >= NOW())
                ORDER BY a.starts_at
                """,
                (patient_id, include_past),
            )
            rows = cur.fetchall()

        return [self._appointment_dict(row) for row in rows]

    def cancel(
        self,
        authenticated_user_id: str | UUID,
        booking_reference: str,
    ) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            patient_id = self._patient_id(cur, authenticated_user_id)

            cur.execute(
                """
                SELECT appointment_id, doctor_id
                FROM appointments
                WHERE booking_reference = %s
                  AND patient_id = %s
                  AND status = 'booked'
                FOR UPDATE
                """,
                (booking_reference.strip(), patient_id),
            )
            appointment = cur.fetchone()

            if appointment is None:
                raise ResourceNotFound(
                    "No active appointment matching that reference "
                    "was found for your account."
                )

            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(appointment["doctor_id"]),),
            )

            cur.execute(
                """
                UPDATE appointments
                SET status = 'cancelled',
                    cancelled_at = NOW(),
                    updated_at = NOW()
                WHERE appointment_id = %s
                  AND patient_id = %s
                  AND status = 'booked'
                RETURNING appointment_id, booking_reference, status,
                          cancelled_at, updated_at
                """,
                (appointment["appointment_id"], patient_id),
            )
            row = cur.fetchone()

            if row is None:
                raise SlotUnavailable(
                    "The appointment changed before cancellation completed."
                )

        return self._appointment_dict(row)

    def reschedule(
        self,
        authenticated_user_id: str | UUID,
        booking_reference: str,
        new_starts_at: str | datetime,
    ) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            patient_id = self._patient_id(cur, authenticated_user_id)

            cur.execute(
                """
                SELECT appointment_id, doctor_id, starts_at, ends_at
                FROM appointments
                WHERE booking_reference = %s
                  AND patient_id = %s
                  AND status = 'booked'
                FOR UPDATE
                """,
                (booking_reference.strip(), patient_id),
            )
            existing = cur.fetchone()

            if existing is None:
                raise ResourceNotFound(
                    "No active appointment matching that reference "
                    "was found for your account."
                )

            doctor_uuid = existing["doctor_id"]
            doctor = self._doctor_details(cur, doctor_uuid)
            zone = self._zone(doctor["timezone_name"])

            try:
                requested_start = (
                    new_starts_at
                    if isinstance(new_starts_at, datetime)
                    else datetime.fromisoformat(new_starts_at)
                )
            except (ValueError, TypeError) as exc:
                raise InvalidAppointmentRequest(
                    "Provide a valid new appointment datetime."
                ) from exc

            if requested_start.tzinfo is None:
                requested_start = requested_start.replace(tzinfo=zone)

            requested_start = requested_start.astimezone(timezone.utc)

            if requested_start <= datetime.now(timezone.utc):
                raise InvalidAppointmentRequest(
                    "Appointments must be rescheduled to a future time."
                )

            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(doctor_uuid),),
            )

            slots = self._get_slots(
                cur,
                doctor_uuid,
                requested_start.astimezone(zone).date(),
            )

            matching_slot = next(
                (
                    slot for slot in slots
                    if datetime.fromisoformat(
                        slot["starts_at"]
                    ).astimezone(timezone.utc) == requested_start
                ),
                None,
            )

            if matching_slot is None:
                raise SlotUnavailable(
                    "The requested new slot is not available."
                )

            requested_end = datetime.fromisoformat(
                matching_slot["ends_at"]
            ).astimezone(timezone.utc)

            cur.execute(
                """
                UPDATE appointments
                SET starts_at = %s,
                    ends_at = %s,
                    updated_at = NOW()
                WHERE appointment_id = %s
                  AND patient_id = %s
                  AND status = 'booked'
                RETURNING appointment_id, booking_reference, patient_id,
                          doctor_id, starts_at, ends_at, status, reason,
                          created_at, updated_at
                """,
                (
                    requested_start,
                    requested_end,
                    existing["appointment_id"],
                    patient_id,
                ),
            )
            row = cur.fetchone()

            if row is None:
                raise SlotUnavailable(
                    "The appointment changed before rescheduling completed."
                )

            row["doctor_name"] = doctor["display_name"]
            row["department_name"] = doctor["department_name"]

        return self._appointment_dict(row)
