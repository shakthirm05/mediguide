
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import jwt
from dotenv import load_dotenv
from pwdlib import PasswordHash

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def create_access_token(user_id: UUID) -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is not configured")

    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    expires_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }

    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(token: str) -> UUID:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is not configured")

    algorithm = os.getenv("JWT_ALGORITHM", "HS256")

    payload = jwt.decode(
        token,
        secret,
        algorithms=[algorithm],
        options={"require": ["sub", "iat", "exp"]},
    )

    return UUID(payload["sub"])
