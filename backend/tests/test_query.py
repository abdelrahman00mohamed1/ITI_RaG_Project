import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.query import Source


@pytest.fixture
def client():
    """Test client with mocked services to ensure instantaneous, deterministic, offline testing."""
    # Mock retrieval service
    mock_retrieval = MagicMock()
    mock_retrieval.collection = MagicMock()
    mock_retrieval.collection.count.return_value = 100
    mock_retrieval.model_name = "test-e5-model"
    mock_retrieval.retrieve.return_value = (
        "[Book: Harry Potter and the Half-Blood Prince | Page: 450]\nSeverus Snape revealed himself as the Half-Blood Prince.",
        [
            Source(
                book_name="Harry Potter and the Half-Blood Prince",
                page_number=450,
                score=0.95,
                chunk_id="hp6_p450_c1",
                content_snippet="Severus Snape revealed himself as the Half-Blood Prince.",
            )
        ],
    )

    # Mock generation service
    mock_generation = MagicMock()
    mock_generation.route_query.side_effect = lambda q: (
        "chitchat" if "hello" in q.lower() else "off-topic" if "weather" in q.lower() else "retrieve"
    )
    mock_generation.handle_chitchat.return_value = "Greetings, wizard! How may the Hogwarts Library assist you?"
    mock_generation.handle_off_topic.return_value = (
        "I am a specialized Harry Potter assistant and cannot answer outside questions."
    )
    mock_generation.generate_rag_answer.return_value = (
        "Severus Snape is the Half-Blood Prince, as revealed in Book 6."
    )

    # Patch the real lifespan initializations so it never tries to download the 2.2GB E5 model during tests
    with patch("app.main.ChromaRetrievalService.initialize", return_value=None), \
         patch("app.main.GenerationService.initialize", return_value=None):
        with TestClient(app) as test_client:
            # Attach mocks to app state
            app.state.retrieval_service = mock_retrieval
            app.state.generation_service = mock_generation
            yield test_client


def test_health_endpoint_success(client):
    """Test GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_store_loaded"] is True


def test_query_invalid_empty_payload(client):
    """Test POST /query with empty json body returns 422 Unprocessable Entity."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_invalid_blank_query(client):
    """Test POST /query with whitespace query returns 422."""
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422


def test_query_retrieve_happy_path(client):
    """Test POST /query with a valid question executes retrieval and returns sources."""
    response = client.post("/query", json={"query": "Who is the Half-Blood Prince?"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Who is the Half-Blood Prince?"
    assert data["route"] == "retrieve"
    assert "Severus Snape" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["book_name"] == "Harry Potter and the Half-Blood Prince"
    assert data["sources"][0]["page_number"] == 450


def test_query_question_field_alias(client):
    """Test POST /query accepting 'question' parameter alias from official criteria."""
    response = client.post("/query", json={"question": "Who is the Half-Blood Prince?"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "retrieve"
    assert len(data["sources"]) > 0


def test_query_chitchat_route(client):
    """Test POST /query routing chitchat without vector store retrieval."""
    response = client.post("/query", json={"query": "Hello there!"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "chitchat"
    assert len(data["sources"]) == 0
    assert "Hogwarts" in data["answer"]


def test_query_off_topic_route(client):
    """Test POST /query routing off-topic queries politely."""
    response = client.post("/query", json={"query": "What is the weather in Paris?"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "off-topic"
    assert len(data["sources"]) == 0
    assert "specialized" in data["answer"]
