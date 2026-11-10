import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_review_unknown_thread_needs_agent():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_anthropic")
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    r = TestClient(app).post("/review", json={"thread_id": "nope", "approved": True})
    assert r.status_code == 404
