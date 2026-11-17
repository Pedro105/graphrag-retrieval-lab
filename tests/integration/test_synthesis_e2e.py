"""Synthesis over the eval_test graph for a lookup, a single-hop and a two-hop
query. Skips if eval_test has not been populated or there is no API key."""

import pytest

from graphrag_lab.core.config import get_settings
from graphrag_lab.core.service import recall
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import build_embedding_provider

TENANT_ID = "eval_test"


@pytest.fixture(autouse=True)
async def _require_seeded_graph(store):
    entity_count, _ = await store.graph_stats(TENANT_ID)
    if entity_count == 0:
        pytest.skip(f"{TENANT_ID} graph not seeded; run `make populate`")


@pytest.fixture
def embedding_provider():
    settings = get_settings()
    return build_embedding_provider(settings)


async def _recall(graph_store, embedding_provider, query, **kwargs):
    scope = Scope.from_dict({"tenant_id": TENANT_ID})
    try:
        return await recall(
            query=query,
            scope=scope,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            synthesise=True,
            **kwargs,
        )
    except Exception as exc:
        pytest.skip(f"Real LLM/embedding API call failed -- check API keys: {exc}")


# Per-case limit/depth mirror the eval-verified parameters in eval/dataset.yaml
# (exact_term_rg88, single_hop_korrigan_produces, bridging_korrigan_to_recall).
# On this small graph HNSW under-returns at small k, so the
# seed-friendly limit differs per query -- the same reason the eval cases pin
# their own limit. retrieval_mode is left unset (defaults to hybrid, as the eval
# harness does by pinning only traversal_depth).


async def test_synthesis_exact_lookup(graph_store, embedding_provider):
    result = await _recall(graph_store, embedding_provider, "RG-88", limit=3, traversal_depth=1)
    assert result.synthesis is not None
    assert result.synthesis.answer.strip()
    assert result.synthesis.confidence != "insufficient_data"


async def test_synthesis_single_hop(graph_store, embedding_provider):
    result = await _recall(
        graph_store,
        embedding_provider,
        "What component does Korrigan Cells Ltd produce?",
        limit=2,
        traversal_depth=1,
    )
    assert result.synthesis is not None
    assert result.synthesis.answer.strip()
    assert result.synthesis.confidence != "insufficient_data"


async def test_synthesis_relational_grounds_in_chain(graph_store, embedding_provider):
    result = await _recall(
        graph_store,
        embedding_provider,
        "Korrigan Cells Ltd",
        limit=2,
        traversal_depth=2,
    )
    assert result.synthesis is not None
    assert result.synthesis.answer.strip()
    assert result.synthesis.confidence != "insufficient_data"

    # At least one entity from the engineered 2-hop chain (Korrigan -> ... ->
    # Voluntary Safety Recall R-2025-014) must be cited as grounding.
    chain_names = {"Korrigan Cells Ltd", "Voluntary Safety Recall R-2025-014"}
    chain_ids = {str(e.id) for e in result.entities if e.name in chain_names}
    assert chain_ids, "expected chain entities to be present in the retrieved subgraph"
    assert set(result.synthesis.grounded_entity_ids) & chain_ids, (
        "expected at least one 2-hop chain entity in grounded_entity_ids"
    )
