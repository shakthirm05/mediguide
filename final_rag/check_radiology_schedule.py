from src.backend.database.connection import DATABASE_URL
import psycopg
from psycopg.rows import dict_row

sql = """
SELECT
    d.display_name,
    s.weekday,
    s.start_time,
    s.end_time,
    s.slot_duration_minutes,
    s.valid_from,
    s.valid_until,
    s.is_active
FROM doctor_schedules s
JOIN doctors d ON d.doctor_id = s.doctor_id
JOIN departments dep ON dep.department_id = d.department_id
WHERE dep.name ILIKE '%Radiology%'
  AND d.display_name ILIKE '%Karthikeyan%'
ORDER BY s.weekday, s.start_time;
"""

with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

        if not rows:
            print("No schedule rows found for this doctor in Radiology.")
        else:
            print(f"Found {len(rows)} schedule row(s):")
            for row in rows:
                print(dict(row))
