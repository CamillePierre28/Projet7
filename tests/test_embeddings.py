import json
from pathlib import Path


EMBEDDINGS_PATH = Path("data/processed/events_embeddings.json")


def test_embeddings_file_exists():
    assert EMBEDDINGS_PATH.exists()


def test_embeddings_have_expected_structure():
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    assert len(data) > 0

    first = data[0]

    assert "uid" in first
    assert "title" in first
    assert "text_for_embedding" in first
    assert "embedding" in first
    assert "embedding_model" in first
    assert "embedding_dim" in first

    assert first["embedding_model"] == "mistral-embed"
    assert isinstance(first["embedding"], list)
    assert len(first["embedding"]) == first["embedding_dim"]
    assert first["embedding_dim"] == 1024