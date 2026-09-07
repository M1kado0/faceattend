"""End-to-end behavior test for the local persistence smoke command."""

from pathlib import Path

from scripts.test_local_persistence import run_smoke


def test_smoke_flow_survives_restart_and_erases_templates(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "smoke.sqlite3")

    assert result.templates_after_restart == 3
    assert result.same_person_status == "matched"
    assert result.unknown_status == "unknown"
    assert result.first_attendance_was_duplicate is False
    assert result.second_attendance_was_duplicate is True
    assert result.erased_template_count == 3
    assert result.templates_after_erasure == 0
    assert result.audit_preserved_after_erasure is True
