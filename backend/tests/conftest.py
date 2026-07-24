import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="crushpilot-tests-"))
os.environ["DEMO_MODE"] = "true"
os.environ["DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["DATABASE_URL"] = ""


@pytest.fixture
def client() -> Iterator[TestClient]:
    from app.core.config import get_settings
    from app.main import create_app

    with TestClient(create_app(get_settings())) as test_client:
        yield test_client


@pytest.fixture
def device_headers() -> dict[str, str]:
    return {"X-Device-Id": "c544979f-8d93-4c16-aa79-c330aa51ee65"}


def pytest_sessionfinish() -> None:
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
