"""Public site test fixtures."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    public_site_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(public_site_root))

    from main import app  # noqa: PLC0415

    return TestClient(app)
