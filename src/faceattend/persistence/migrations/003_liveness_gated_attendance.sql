ALTER TABLE attendance_records
ADD COLUMN liveness_attempt_id TEXT REFERENCES liveness_attempts(id);

CREATE TRIGGER attendance_records_require_liveness_insert
BEFORE INSERT ON attendance_records
WHEN NEW.liveness_attempt_id IS NULL
BEGIN
    SELECT RAISE(ABORT, 'attendance requires liveness evidence');
END;

CREATE TRIGGER attendance_records_require_liveness_update
BEFORE UPDATE OF liveness_attempt_id ON attendance_records
WHEN NEW.liveness_attempt_id IS NULL
BEGIN
    SELECT RAISE(ABORT, 'attendance requires liveness evidence');
END;

CREATE INDEX idx_attendance_records_liveness
ON attendance_records(liveness_attempt_id);
