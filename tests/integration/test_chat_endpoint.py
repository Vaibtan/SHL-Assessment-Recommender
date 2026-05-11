"""Integration test for the FastAPI /chat endpoint via TestClient.

Exercises the full HTTP path including lifespan loading. Uses a fake LLM by
patching the agent's llm attribute after startup.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shl_recommender.schemas import ChatResponse
from tests.integration.fakes import FakeLLMClient, stub_json


@pytest.fixture()
def client(tmp_path: Path):
    """Build an isolated TestClient that loads the real index but uses a fake LLM."""
    os.environ["SHL_INDEX_DIR"] = str(Path(__file__).resolve().parents[2] / "data" / "build")
    from shl_recommender.main import app

    fake = FakeLLMClient()
    fake.router_replies.append(
        stub_json(
            {
                "intent": "clarify",
                "clarifying_question": "What role and at what seniority?",
                "is_final_turn": False,
            }
        )
    )

    with TestClient(app) as c:
        # Replace the LLM after lifespan loaded the agent.
        c.app.state.app_state.agent.llm = fake
        yield c


def test_health_returns_200_when_index_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_clarify_path_via_http(client: TestClient) -> None:
    payload = {"messages": [{"role": "user", "content": "I need an assessment"}]}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    body = ChatResponse.model_validate(response.json())
    assert body.recommendations == []
    assert body.end_of_conversation is False
    assert "?" in body.reply


def test_chat_rejects_invalid_role(client: TestClient) -> None:
    payload = {"messages": [{"role": "robot", "content": "hi"}]}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
