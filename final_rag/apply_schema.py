
from pathlib import Path

import psycopg

from src.backend.database.connection import DATABASE_URL

schema_path = Path("src/backend/database/schema.sql")
schema_sql = schema_path.read_text(encoding="utf-8")

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cursor:
        cursor.execute(schema_sql)

print("Schema applied successfully.")
