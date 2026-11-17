"""Stating the same fact twice, in different words and in separate writes,
must leave one claim with a support count of two, not two parallel edges.

Uses real extraction, so it skips without an Anthropic key.
"""

import pytest

from graphrag_lab.core.config import get_settings
from graphrag_lab.graph import queries
from graphrag_lab.core.service import remember
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import build_embedding_provider


async def test_repeated_relation_between_same_entities_merges_not_duplicates(
    graph_store, tenant_id
):
    settings = get_settings()
    embedding_provider = build_embedding_provider(settings)
    scope = Scope.from_dict({"tenant_id": tenant_id, "user_id": "u_test"})

    sentence_1 = "Pedro works at ASML as a software engineer."
    sentence_2 = "Pedro has been working at ASML for the last two years."

    try:
        memory_1, _tokens_1 = await remember(
            content=sentence_1,
            scope=scope,
            source="dedup_regression_test",
            graph_store=graph_store,
            embedding_provider=embedding_provider,
        )
        memory_2, _tokens_2 = await remember(
            content=sentence_2,
            scope=scope,
            source="dedup_regression_test",
            graph_store=graph_store,
            embedding_provider=embedding_provider,
        )
    except Exception as exc:
        pytest.skip(
            "Real LLM/embedding API call failed -- check ANTHROPIC_API_KEY and "
            f"OPENAI_API_KEY are set to real, valid keys: {exc}"
        )

    pedro_ids = {
        e.id for m in (memory_1, memory_2) for e in m.entities if "pedro" in e.name.lower()
    }
    asml_ids = {e.id for m in (memory_1, memory_2) for e in m.entities if "asml" in e.name.lower()}

    # Both writes must resolve to the same two entities, or the claim check
    # below means nothing.
    assert len(pedro_ids) == 1, f"expected Pedro to resolve to one entity, got {pedro_ids}"
    assert len(asml_ids) == 1, f"expected ASML to resolve to one entity, got {asml_ids}"

    pedro_id = next(iter(pedro_ids))
    asml_id = next(iter(asml_ids))

    claims = [
        claim
        for claim in await graph_store._claims(queries.PROJECT_LIVE_CLAIMS, scope)
        if claim.subject_id == pedro_id and claim.object_id == asml_id
    ]
    assert len(claims) == 1, f"expected one claim between Pedro and ASML, got {claims}"
    assert claims[0].status == "active"
    assert claims[0].support_count == 2
