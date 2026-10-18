
-- Trusted registry of synthetic hospital patient identities.
-- Records must be provisioned by a trusted operator.

CREATE TABLE IF NOT EXISTS patient_registry (
    patient_identifier VARCHAR(50) PRIMARY KEY,
    claim_code_hash TEXT NOT NULL,
    linked_user_id UUID UNIQUE
        REFERENCES app_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    linked_at TIMESTAMPTZ,

    CONSTRAINT patient_registry_link_state_check
        CHECK (
            (linked_user_id IS NULL AND linked_at IS NULL)
            OR
            (linked_user_id IS NOT NULL AND linked_at IS NOT NULL)
        )
);
