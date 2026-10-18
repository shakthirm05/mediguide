BEGIN;

ALTER TABLE departments
    ADD COLUMN IF NOT EXISTS source_code VARCHAR(40),
    ADD COLUMN IF NOT EXISTS access_class VARCHAR(40),
    ADD COLUMN IF NOT EXISTS service_scope TEXT;

ALTER TABLE doctors
    ADD COLUMN IF NOT EXISTS source_code VARCHAR(40),
    ADD COLUMN IF NOT EXISTS specialty VARCHAR(150),
    ADD COLUMN IF NOT EXISTS location_code VARCHAR(50);

ALTER TABLE doctor_schedules
    ADD COLUMN IF NOT EXISTS source_schedule_id VARCHAR(40),
    ADD COLUMN IF NOT EXISTS department_id UUID
        REFERENCES departments(department_id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS location_code VARCHAR(50),
    ADD COLUMN IF NOT EXISTS booking_mode VARCHAR(80),
    ADD COLUMN IF NOT EXISTS availability_status VARCHAR(120),
    ADD COLUMN IF NOT EXISTS notes TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_departments_source_code
    ON departments(source_code)
    WHERE source_code IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_doctors_source_code
    ON doctors(source_code)
    WHERE source_code IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_doctor_schedules_source_schedule_id
    ON doctor_schedules(source_schedule_id)
    WHERE source_schedule_id IS NOT NULL;

COMMIT;
