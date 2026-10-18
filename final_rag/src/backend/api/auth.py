
from re import fullmatch
from uuid import uuid4

import psycopg
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from fastapi import Depends
from src.backend.auth.dependencies import get_current_user


from src.backend.auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from src.backend.database.connection import DATABASE_URL


router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()

        if not fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Enter a valid email address")

        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
)
def register(request: RegisterRequest):
    user_id = uuid4()
    password_hash = hash_password(request.password)

    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO app_users (
                        user_id,
                        email,
                        password_hash,
                        role
                    )
                    VALUES (%s, %s, %s, 'patient')
                    RETURNING user_id, email, role
                    """,
                    (
                        user_id,
                        request.email,
                        password_hash,
                    ),
                )
                user = cursor.fetchone()

        return {
            "user_id": str(user[0]),
            "email": user[1],
            "role": user[2],
            "message": "Account created. Please log in.",
        }

    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    except psycopg.Error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account service is temporarily unavailable",
        )


@router.post("/login")
def login(request: LoginRequest):
    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, password_hash, is_active
                    FROM app_users
                    WHERE email = %s
                    """,
                    (request.email,),
                )
                user = cursor.fetchone()

    except psycopg.Error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is temporarily unavailable",
        )

    # Use the same public error for unknown email and wrong password.
    if user is None or not verify_password(
        request.password,
        user[1],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user[2]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is inactive",
        )

    try:
        access_token = create_access_token(user[0])
    except (RuntimeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication configuration is unavailable",
        )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,
    }
@router.get("/me")
def get_my_account(
    current_user: dict = Depends(get_current_user),
):
    return {
        "user_id": str(current_user["user_id"]),
        "email": current_user["email"],
        "role": current_user["role"],
    }