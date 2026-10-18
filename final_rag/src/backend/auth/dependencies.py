
from collections.abc import Callable

import psycopg
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.backend.auth.security import decode_access_token
from src.backend.database.connection import DATABASE_URL

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        user_id = decode_access_token(credentials.credentials)

        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, email, role, is_active
                    FROM app_users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()

        if row is None or not row[3]:
            raise unauthorized

        return {
            "user_id": row[0],
            "email": row[1],
            "role": row[2],
        }

    except HTTPException:
        raise
    except Exception:
        raise unauthorized


def require_roles(*allowed_roles: str) -> Callable:
    """Require an authenticated user with one of the specified roles."""

    def role_checker(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )

        return current_user

    return role_checker
