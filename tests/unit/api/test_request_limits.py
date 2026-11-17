"""Request-size ceilings: every bounded field feeds an LLM prompt or a graph
traversal, so an oversized request must be rejected before any work (and any
spend) happens. GraphStore and the embedding provider are mocked.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from graphrag_lab.api import routes
from graphrag_lab.api.app import app
from graphrag_lab.api.dependencies import get_embedding_provider, get_graph_store
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.vector.embeddings import EmbeddingProvider


@pytest.fixture
def client():
    graph_store = AsyncMock(spec=GraphStore)
    embedding_provider = AsyncMock(spec=EmbeddingProvider)
    app.dependency_overrides[get_graph_store] = lambda: graph_store
    app.dependency_overrides[get_embedding_provider] = lambda: embedding_provider
    yield TestClient(app), graph_store, embedding_provider
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "overrides",
    [
        {"content": "x" * (routes.MAX_CONTENT_CHARS + 1)},
        {"source": "s" * (routes.MAX_SOURCE_CHARS + 1)},
        {"evidence": ["e"] * (routes.MAX_EVIDENCE_ITEMS + 1)},
        {"evidence": ["e" * (routes.MAX_EVIDENCE_CHARS + 1)]},
        {"extra_scope": {f"k{i}": i for i in range(routes.MAX_SCOPE_KEYS + 1)}},
        {"tenant_id": "t" * 41},
    ],
)
def test_oversized_memory_is_rejected_before_any_work(client, overrides):
    test_client, graph_store, embedding_provider = client

    resp = test_client.post("/v1/memories", json={"content": "c", "source": "s", **overrides})

    assert resp.status_code == 422
    assert graph_store.mock_calls == []
    assert embedding_provider.mock_calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"query": "q" * (routes.MAX_QUERY_CHARS + 1)},
        {"limit": routes.MAX_SEED_LIMIT + 1},
        {"traversal_depth": routes.MAX_TRAVERSAL_DEPTH + 1},
        {"max_entities": routes.MAX_ENTITIES_CAP + 1},
    ],
)
def test_oversized_recall_is_rejected_before_any_work(client, overrides):
    test_client, graph_store, embedding_provider = client

    resp = test_client.post("/v1/recall", json={"query": "q", **overrides})

    assert resp.status_code == 422
    assert graph_store.mock_calls == []
    assert embedding_provider.mock_calls == []


def test_requests_at_the_ceiling_are_accepted(client):
    test_client, graph_store, embedding_provider = client
    graph_store.find_similar_entities.return_value = []
    graph_store.find_by_fulltext.return_value = []
    embedding_provider.embed.return_value = [1.0, 0.0]

    resp = test_client.post(
        "/v1/recall",
        json={
            "query": "q" * routes.MAX_QUERY_CHARS,
            "limit": routes.MAX_SEED_LIMIT,
            "traversal_depth": routes.MAX_TRAVERSAL_DEPTH,
            "max_entities": routes.MAX_ENTITIES_CAP,
        },
    )

    assert resp.status_code == 200
