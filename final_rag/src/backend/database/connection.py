
import os

import psycopg
from dotenv import load_dotenv

# Load environment variables from the project root.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. Configure it in the project-root .env file."
    )


def check_database_connection() -> bool:
    """Verify that PostgreSQL is reachable and accepts the app credentials."""
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

    return result == (1,)
