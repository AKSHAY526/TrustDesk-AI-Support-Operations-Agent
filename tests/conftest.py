"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, llm  # noqa: E402
from app.db import connect, init_db  # noqa: E402
from app.loader import load_all  # noqa: E402

AGENT = {"Authorization": "Bearer demo-agent-token"}
MANAGER = {"Authorization": "Bearer demo-manager-token"}
ADMIN = {"Authorization": "Bearer demo-admin-token"}


@pytest.fixture(scope="session", autouse=True)
def _use_hashing_embedder():
    """Avoid downloading a model during tests."""
    embeddings.set_embedder(embeddings.HashingEmbedder())
    yield
    embeddings.set_embedder(None)


@pytest.fixture(autouse=True)
def _pin_stand_in_model():
    """Use the stand-in model even if .env points at a real provider."""
    llm.set_llm(llm.HeuristicLLM())
    yield
    llm.set_llm(None)


@pytest.fixture
def seeded_db(tmp_path) -> Path:
    """A temporary database with the full seed data loaded."""
    path = tmp_path / "trustdesk_test.db"
    load_all(path)
    return path


@pytest.fixture
def conn(seeded_db):
    connection = connect(seeded_db)
    yield connection
    connection.close()


@pytest.fixture
def client(seeded_db):
    """TestClient wired to the temporary database."""
    from fastapi.testclient import TestClient

    from app.deps import get_db
    from app.main import app

    def _override():
        connection = init_db(seeded_db)
        try:
            yield connection
        finally:
            connection.close()

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
