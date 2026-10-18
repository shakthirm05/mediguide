BEGIN;

CREATE TABLE IF NOT EXISTS patient_memory_consent (
    patient_id UUID PRIMARY KEY
        REFERENCES patient_profiles(patient_id) ON DELETE CASCADE,
    granted BOOLEAN NOT NULL DEFAULT FALSE,
    granted_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS patient_memory_items (
    memory_id UUID PRIMARY KEY,
    patient_id UUID NOT NULL
        REFERENCES patient_profiles(patient_id) ON DELETE CASCADE,
    memory_key VARCHAR(80) NOT NULL,
    memory_value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (patient_id, memory_key),
    CONSTRAINT patient_memory_key_not_empty
        CHECK (length(trim(memory_key)) > 0),
    CONSTRAINT patient_memory_value_not_empty
        CHECK (length(trim(memory_value)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_patient_memory_items_patient
    ON patient_memory_items(patient_id);

COMMIT;