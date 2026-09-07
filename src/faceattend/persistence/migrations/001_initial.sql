CREATE TABLE people (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    created_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE consent_records (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    purpose TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    withdrawn_at TEXT
);

CREATE TABLE model_versions (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (role, name, version, checksum)
);

CREATE TABLE configuration_versions (
    id TEXT PRIMARY KEY,
    configuration_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE enrollment_sessions (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    consent_record_id TEXT NOT NULL REFERENCES consent_records(id),
    configuration_version_id TEXT NOT NULL REFERENCES configuration_versions(id),
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    failure_reason TEXT
);

CREATE TABLE embedding_templates (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    enrollment_session_id TEXT NOT NULL REFERENCES enrollment_sessions(id) ON DELETE CASCADE,
    model_version_id TEXT NOT NULL REFERENCES model_versions(id),
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    dtype TEXT NOT NULL CHECK (dtype = 'float32'),
    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    pose_bin TEXT,
    quality_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE attendance_sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    opened_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE attendance_records (
    id TEXT PRIMARY KEY,
    attendance_session_id TEXT NOT NULL REFERENCES attendance_sessions(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    checked_in_at TEXT NOT NULL,
    match_score REAL NOT NULL,
    match_threshold REAL NOT NULL,
    ambiguity_margin REAL NOT NULL,
    configuration_version_id TEXT NOT NULL REFERENCES configuration_versions(id),
    UNIQUE (attendance_session_id, person_id)
);

CREATE TABLE liveness_attempts (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES people(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,
    challenge_sequence_json TEXT NOT NULL,
    completed_challenges_json TEXT NOT NULL,
    active_decision TEXT NOT NULL,
    passive_decision TEXT NOT NULL,
    failure_reason TEXT,
    passive_median REAL,
    passive_minimum REAL,
    suspicious_frame_count INTEGER NOT NULL DEFAULT 0,
    processing_failure_count INTEGER NOT NULL DEFAULT 0,
    configuration_version_id TEXT NOT NULL REFERENCES configuration_versions(id),
    created_at TEXT NOT NULL
);

CREATE TABLE audit_events (
    id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_id TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_embedding_templates_person ON embedding_templates(person_id);
CREATE INDEX idx_attendance_records_session ON attendance_records(attendance_session_id);
CREATE INDEX idx_liveness_attempts_person ON liveness_attempts(person_id);
CREATE INDEX idx_audit_events_target ON audit_events(target_id);
