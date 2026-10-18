
from pathlib import Path

import psycopg

from src.backend.database.connection import DATABASE_URL


schema_path = Path("src/backend/database/patient_memory.sql")

if not schema_path.is_file():
    raise FileNotFoundError(f"Schema file not found: {schema_path}")

sql = schema_path.read_text(encoding="utf-8")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

try:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)

    print("Patient-memory schema applied successfully.")

except Exception:
    print("Schema application failed. Check the database error above.")
    raise
