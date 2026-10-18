
-- MediGuide initial relational schema
-- Run against the mediguide database as mediguide_app.

CREATE TABLE IF NOT EXISTS app_users (
    user_id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'patient'
        CHECK (role IN ('patient', 'admin', 'staff')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    patient_id UUID PRIMARY KEY,
    user_id UUID UNIQUE NOT NULL
        REFERENCES app_users(user_id) ON DELETE RESTRICT,
    patient_identifier VARCHAR(50) UNIQUE NOT NULL,
    preferred_language VARCHAR(20) NOT NULL DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS departments (
    department_id UUID PRIMARY KEY,
    name VARCHAR(120) UNIQUE NOT NULL,
    location_code VARCHAR(50) NOT NULL,
    floor_label VARCHAR(50),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id UUID PRIMARY KEY,
    department_id UUID NOT NULL
        REFERENCES departments(department_id) ON DELETE RESTRICT,
    display_name VARCHAR(150) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY,
    user_id UUID NOT NULL
        REFERENCES app_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doctors_department
    ON doctors(department_id);

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations(user_id, updated_at DESC);
