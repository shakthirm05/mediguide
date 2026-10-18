
BEGIN;

-- Department-local timezone for interpreting recurring schedules.
ALTER TABLE departments
    ADD COLUMN IF NOT EXISTS timezone_name VARCHAR(64)
    NOT NULL DEFAULT 'Asia/Kolkata';

-- A doctor can have multiple recurring schedules.
-- weekday: ISO weekday convention, Monday = 1, Sunday = 7.
CREATE TABLE IF NOT EXISTS doctor_schedules (
    schedule_id UUID PRIMARY KEY,
    doctor_id UUID NOT NULL
        REFERENCES doctors(doctor_id) ON DELETE RESTRICT,

    weekday SMALLINT NOT NULL
        CHECK (weekday BETWEEN 1 AND 7),

    start_time TIME NOT NULL,
    end_time TIME NOT NULL,

    slot_duration_minutes SMALLINT NOT NULL
        CHECK (slot_duration_minutes BETWEEN 5 AND 240),

    valid_from DATE NOT NULL,
    valid_until DATE,

    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT doctor_schedule_time_order
        CHECK (end_time > start_time),

    CONSTRAINT doctor_schedule_validity_order
        CHECK (valid_until IS NULL OR valid_until >= valid_from)
);

CREATE INDEX IF NOT EXISTS idx_doctor_schedules_lookup
    ON doctor_schedules (doctor_id, weekday, valid_from)
    WHERE is_active = TRUE;

-- A single appointment row represents a booking for one patient and doctor.
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id UUID PRIMARY KEY,
    booking_reference VARCHAR(32) UNIQUE NOT NULL,

    patient_id UUID NOT NULL
        REFERENCES patient_profiles(patient_id) ON DELETE RESTRICT,

    doctor_id UUID NOT NULL
        REFERENCES doctors(doctor_id) ON DELETE RESTRICT,

    starts_at TIMESTAMPTZ NOT NULL,
    ends_at TIMESTAMPTZ NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'booked'
        CHECK (
            status IN (
                'booked',
                'cancelled',
                'completed',
                'no_show'
            )
        ),

    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cancelled_at TIMESTAMPTZ,

    CONSTRAINT appointment_time_order
        CHECK (ends_at > starts_at),

    CONSTRAINT appointment_cancellation_consistency
        CHECK (
            (status = 'cancelled' AND cancelled_at IS NOT NULL)
            OR
            (status <> 'cancelled' AND cancelled_at IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_appointments_doctor_time
    ON appointments (doctor_id, starts_at, ends_at)
    WHERE status = 'booked';

CREATE INDEX IF NOT EXISTS idx_appointments_patient_time
    ON appointments (patient_id, starts_at DESC);

CREATE INDEX IF NOT EXISTS idx_appointments_status_time
    ON appointments (status, starts_at);

-- Serialize concurrent writes for the same doctor and reject overlapping
-- active bookings. This protection applies to direct SQL writes as well
-- as writes made by the application.
CREATE OR REPLACE FUNCTION prevent_doctor_appointment_overlap()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = 'booked' THEN
        PERFORM pg_advisory_xact_lock(
            hashtextextended(NEW.doctor_id::TEXT, 0)
        );

        IF EXISTS (
            SELECT 1
            FROM appointments existing
            WHERE existing.doctor_id = NEW.doctor_id
              AND existing.status = 'booked'
              AND existing.appointment_id <> NEW.appointment_id
              AND existing.starts_at < NEW.ends_at
              AND existing.ends_at > NEW.starts_at
        ) THEN
            RAISE EXCEPTION 'Appointment time overlaps an existing booking'
                USING ERRCODE = '23P01';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_doctor_appointment_overlap
    ON appointments;

CREATE TRIGGER trg_prevent_doctor_appointment_overlap
BEFORE INSERT OR UPDATE OF doctor_id, starts_at, ends_at, status
ON appointments
FOR EACH ROW
EXECUTE FUNCTION prevent_doctor_appointment_overlap();

COMMIT;
