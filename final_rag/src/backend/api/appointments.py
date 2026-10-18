
from datetime import date, datetime
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.backend.auth.dependencies import get_current_user
from src.backend.services.appointment_service import (
    AppointmentError,
    AppointmentService,
    InvalidAppointmentRequest,
    PatientProfileNotLinked,
    ResourceNotFound,
    SlotUnavailable,
)

router = APIRouter(prefix="/api/v1", tags=["Appointments"])
appointment_service = AppointmentService()


class BookingRequest(BaseModel):
    doctor_id: str
    starts_at: datetime
    reason: str | None = Field(default=None, max_length=500)


class RescheduleRequest(BaseModel):
    new_starts_at: datetime


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PatientProfileNotLinked):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    if isinstance(exc, ResourceNotFound):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    if isinstance(exc, SlotUnavailable):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    if isinstance(exc, InvalidAppointmentRequest):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    if isinstance(exc, AppointmentError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Appointment service could not complete the request.",
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected appointment error occurred.",
    )


def _run(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except HTTPException:
        raise
    except psycopg.errors.ExclusionViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That time overlaps another booking. Please choose another slot.",
        )
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A booking reference or record conflict occurred. Please retry.",
        )
    except AppointmentError as exc:
        raise _handle_service_error(exc)
    except psycopg.Error:
        # Do not expose SQL, credentials, or database internals to clients.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The appointment database could not complete the request.",
        )


def _user_id(current_user: dict[str, Any]) -> str:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity is missing.",
        )
    return str(user_id)


@router.get("/directory/departments")
def list_departments(
    current_user: dict = Depends(get_current_user),
):
    return _run(appointment_service.list_departments)


@router.get("/directory/doctors")
def list_doctors(
    department_id: str | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.list_doctors,
        department_id=department_id,
    )


@router.get("/appointments/availability")
def get_availability(
    doctor_id: str,
    appointment_date: date = Query(..., alias="date"),
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.get_availability,
        doctor_id=doctor_id,
        requested_date=appointment_date,
    )


@router.post(
    "/appointments",
    status_code=status.HTTP_201_CREATED,
)
def book_appointment(
    request: BookingRequest,
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.book,
        authenticated_user_id=_user_id(current_user),
        doctor_id=request.doctor_id,
        starts_at=request.starts_at,
        reason=request.reason,
    )


@router.get("/appointments/me")
def list_my_appointments(
    include_past: bool = Query(default=False),
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.list_my_appointments,
        authenticated_user_id=_user_id(current_user),
        include_past=include_past,
    )


@router.post("/appointments/{booking_reference}/cancel")
def cancel_appointment(
    booking_reference: str,
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.cancel,
        authenticated_user_id=_user_id(current_user),
        booking_reference=booking_reference,
    )


@router.post("/appointments/{booking_reference}/reschedule")
def reschedule_appointment(
    booking_reference: str,
    request: RescheduleRequest,
    current_user: dict = Depends(get_current_user),
):
    return _run(
        appointment_service.reschedule,
        authenticated_user_id=_user_id(current_user),
        booking_reference=booking_reference,
        new_starts_at=request.new_starts_at,
    )
