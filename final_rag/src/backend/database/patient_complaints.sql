
CREATE TABLE IF NOT EXISTS patient_complaints (
    complaint_id UUID PRIMARY KEY,
    complaint_reference VARCHAR(20) NOT NULL UNIQUE,

    patient_id UUID NOT NULL
        REFERENCES patient_profiles(patient_id),

    category VARCHAR(50) NOT NULL DEFAULT 'general',
    description TEXT NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'submitted'
        CHECK (
            status IN (
                'submitted',
                'under_review',
                'resolved',
                'withdrawn'
            )
        ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT patient_complaints_description_length
        CHECK (
            LENGTH(TRIM(description)) BETWEEN 5 AND 3000
        )
);

CREATE INDEX IF NOT EXISTS idx_patient_complaints_patient_created
    ON patient_complaints (patient_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_patient_complaints_status
    ON patient_complaints (status);
