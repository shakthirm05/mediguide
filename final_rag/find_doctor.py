from src.backend.database.connection import DATABASE_URL
import psycopg
from psycopg.rows import dict_row

sql = """
SELECT doctor_id, display_name, department_id
FROM doctors
WHERE display_name ILIKE '%Karthikeyan%';
"""

with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            print(dict(row))
