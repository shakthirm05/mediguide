
import getpass
import re
from uuid import uuid4

import psycopg

from src.backend.auth.security import hash_password
from src.backend.database.connection import DATABASE_URL


def main():
    email = input("Developer email: ").strip().lower()

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise SystemExit("Enter a valid email address.")

    password = getpass.getpass("New developer password (12+ chars): ")
    confirmation = getpass.getpass("Confirm password: ")

    if len(password) < 12 or len(password) > 128:
        raise SystemExit("Password must be 12–128 characters.")

    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    password_hash = hash_password(password)

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO app_users (
                        user_id, email, password_hash, role
                    )
                    VALUES (%s, %s, %s, 'admin')
                    RETURNING user_id, email, role
                    """,
                    (uuid4(), email, password_hash),
                )
                user_id, created_email, role = cursor.fetchone()

        print("Developer account created.")
        print(f"User ID: {user_id}")
        print(f"Role: {role}")
        print(f"Email: {created_email}")

    except psycopg.errors.UniqueViolation:
        raise SystemExit(
            "That email already exists. No existing account was changed."
        )


if __name__ == "__main__":
    main()
