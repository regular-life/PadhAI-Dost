import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure python-rag root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)

def test_health_check():
    """Verify health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "ok"]

def test_chunk_text_api():
    """Verify text chunking endpoint."""
    payload = {
        "text": "Header 1\nThis is paragraph one.\n\nHeader 2\nThis is paragraph two.",
        "chunk_size": 100,
        "overlap": 10
    }
    response = client.post("/chunk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0

def test_embed_api_validation():
    """Verify embedding validation on empty text."""
    payload = {"text": ""}
    response = client.post("/embed", json=payload)
    # Validation error or 400
    assert response.status_code in [400, 422]
