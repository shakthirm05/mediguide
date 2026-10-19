
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.backend.auth.dependencies import get_current_user
from src.backend.services.complaint_service import complaint_service


router = APIRouter(
    prefix="/complaints",
    tags=["Patient Complaints"],
)


class ComplaintCreateRequest(BaseModel):
    description: str = Field(min_length=5, max_length=3000)
    category: Literal[
        "general",
        "staff",
        "billing",
        "facilities",
        "appointment",
        "waiting_time",
        "other",
    ] = "general"


class ComplaintResponse(BaseModel):
    complaint_id: UUID
    complaint_reference: str
    category: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime


def authenticated_user_id(current_user: dict) -> str:
    user_id = current_user.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user ID is missing.",
        )

    return str(user_id)


@router.post(
    "",
    response_model=ComplaintResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_complaint(
    request: ComplaintCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return complaint_service.create_complaint(
            user_id=authenticated_user_id(current_user),
            description=request.description,
            category=request.category,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[ComplaintResponse])
def list_my_complaints(
    current_user: dict = Depends(get_current_user),
):
    try:
        return complaint_service.list_complaints(
            user_id=authenticated_user_id(current_user)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/{complaint_reference}", response_model=ComplaintResponse)
def get_my_complaint(
    complaint_reference: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        complaint = complaint_service.get_complaint(
            user_id=authenticated_user_id(current_user),
            complaint_reference=complaint_reference,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return complaint

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
