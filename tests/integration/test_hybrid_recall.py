"""Hybrid recall (vector + full-text, fused) against Neo4j.

Entities are written with hand-picked embeddings and the query embedding is
fixed, so the vector ranking is deterministic. The case of interest is an
identifier ("RG-88") whose embedding is unrelated to the query: only
full-text search can find it.
"""

import pytest

from graphrag_lab.core.service import recall
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.memory import Memory
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import EmbeddingProvider

QUERY_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return len(QUERY_EMBEDDING)

    """Always returns the same fixed vector, regardless of input text --
    lets the test control vector similarity precisely without a real
    embedding API call."""

    async def embed(self, text: str) -> list[float]:
        return QUERY_EMBEDDING

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [QUERY_EMBEDDING for _ in texts]


@pytest.fixture
async def seeded_scope(store, tenant_id):
    """Two decoys close to QUERY_EMBEDDING by cosine similarity (so vector
    search ranks them #1 and #2), and one exact-term entity ("RG-88")
    whose embedding is orthogonal to QUERY_EMBEDDING and whose name shares
    no tokens with the decoys -- vector search at limit=2 never sees it,
    full-text search on the query string "RG-88" only sees it.
    """
    scope = Scope.from_dict({"tenant_id": tenant_id, "user_id": "u_test"})
    await store.ensure_graph_initialized(tenant_id, embedding_dimension=4)
    provenance = Provenance(source="integration_test")

    decoy_close = Entity(
        name="Battery Module Alpha",
        entity_type="component",
        scope=scope,
        provenance=provenance,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    decoy_far = Entity(
        name="Battery Module Beta",
        entity_type="component",
        scope=scope,
        provenance=provenance,
        embedding=[0.9, 0.1, 0.0, 0.0],
    )
    exact_term = Entity(
        name="RG-88",
        entity_type="component",
        scope=scope,
        provenance=provenance,
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    await store.write_memory(
        Memory(
            content="three components",
            entities=[decoy_close, decoy_far, exact_term],
            scope=scope,
            provenance=provenance,
        )
    )
    return scope


async def test_vector_only_misses_the_exact_term_entity(store, seeded_scope):
    result = await recall(
        query="RG-88",
        scope=seeded_scope,
        graph_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        limit=2,
        retrieval_mode="vector",
    )

    names = {entity.name for entity in result.entities}
    assert names == {"Battery Module Alpha", "Battery Module Beta"}
    assert "RG-88" not in names


async def test_fulltext_only_finds_the_exact_term_entity(store, seeded_scope):
    result = await recall(
        query="RG-88",
        scope=seeded_scope,
        graph_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        limit=2,
        retrieval_mode="fulltext",
    )

    names = {entity.name for entity in result.entities}
    assert names == {"RG-88"}


async def test_hybrid_recovers_the_exact_term_entity_vector_alone_misses(store, seeded_scope):
    result = await recall(
        query="RG-88",
        scope=seeded_scope,
        graph_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        limit=2,
        retrieval_mode="hybrid",
    )

    names = {entity.name for entity in result.entities}
    assert "RG-88" in names
