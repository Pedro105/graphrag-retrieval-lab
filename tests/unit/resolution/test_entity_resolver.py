"""Unit tests for entity resolution thresholds, fully mocked (no real API calls)."""

from unittest.mock import AsyncMock

import pytest

from graphrag_lab.extraction.schemas import ExtractedEntity
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.claim import apply_claim, derive_active_view
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope
from graphrag_lab.resolution.entity_resolver import resolve_entity
from graphrag_lab.vector.embeddings import EmbeddingProvider

SCOPE = Scope.from_dict({"tenant_id": "acme", "user_id": "u_1"})
PROVENANCE = Provenance(source="agent_1")
EXTRACTED = ExtractedEntity(name="Pedro Costa", entity_type="person")
QUERY_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


def make_existing_entity(**overrides) -> Entity:
    defaults = dict(
        name="Pedro",
        entity_type="person",
        scope=SCOPE,
        provenance=PROVENANCE,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    defaults.update(overrides)
    return Entity(**defaults)


def make_mocks(candidates: list[tuple[Entity, float]]) -> tuple[GraphStore, EmbeddingProvider]:
    graph_store = AsyncMock(spec=GraphStore)
    graph_store.find_similar_entities.return_value = candidates
    embedding_provider = AsyncMock(spec=EmbeddingProvider)
    embedding_provider.embed.return_value = QUERY_EMBEDDING
    return graph_store, embedding_provider


async def test_no_candidates_creates_new_entity():
    graph_store, embedding_provider = make_mocks([])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.name == "Pedro Costa"
    assert result.entity_type == "person"
    assert result.merge_candidates == []
    assert result.embedding == QUERY_EMBEDDING
    assert result.scope == SCOPE
    assert result.provenance == PROVENANCE


async def test_below_candidate_threshold_creates_new_entity_with_no_merge_candidates():
    existing = make_existing_entity()
    graph_store, embedding_provider = make_mocks([(existing, 0.5)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id != existing.id
    assert result.merge_candidates == []


async def test_candidate_threshold_range_records_merge_candidate():
    existing = make_existing_entity()
    graph_store, embedding_provider = make_mocks([(existing, 0.85)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id != existing.id
    assert result.merge_candidates == [existing.id]


async def test_at_or_above_merge_threshold_merges_into_existing_entity():
    existing = make_existing_entity()
    graph_store, embedding_provider = make_mocks([(existing, 0.95)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    # Same entity, plus a claim recording that this source mentioned it.
    assert result.id == existing.id
    assert result.name == existing.name
    assert result.embedding == existing.embedding
    assert len(result.claims) == len(existing.claims) + 1
    assert result.contributing_sources == ["agent_1"]


async def test_merge_records_conflicting_property_and_supersedes_prior_claim():
    prior_claims = apply_claim([], Provenance(source="agent_1"), {"price": "4.20"})
    prior_properties, prior_provenance, prior_sources = derive_active_view(prior_claims)
    existing = make_existing_entity(
        properties=prior_properties,
        provenance=prior_provenance,
        claims=prior_claims,
        contributing_sources=prior_sources,
    )
    graph_store, embedding_provider = make_mocks([(existing, 0.95)])
    extracted = ExtractedEntity(
        name="Pedro Costa", entity_type="person", properties={"price": "4.50"}
    )

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.properties == {"price": "4.50"}
    price_claims = [c for c in result.claims if c.property_name == "price"]
    assert len(price_claims) == 2
    old_claim, new_claim = price_claims
    assert old_claim.value == "4.20"
    assert new_claim.value == "4.50"
    assert new_claim.provenance.supersedes == [str(old_claim.id)]


async def test_merge_with_no_new_properties_still_records_a_touch_claim():
    existing = make_existing_entity()
    assert existing.claims == []
    graph_store, embedding_provider = make_mocks([(existing, 0.95)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert len(result.claims) == 1
    assert result.claims[0].property_name is None
    assert result.contributing_sources == ["agent_1"]


@pytest.mark.parametrize(
    "score,expect_reuse",
    [
        (0.92, True),
        (0.9199, "candidate"),
        (0.80, "candidate"),
        (0.7999, False),
    ],
)
async def test_threshold_boundaries(score, expect_reuse):
    existing = make_existing_entity()
    graph_store, embedding_provider = make_mocks([(existing, score)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    if expect_reuse is True:
        assert result.id == existing.id
    elif expect_reuse == "candidate":
        assert result.id != existing.id
        assert result.merge_candidates == [existing.id]
    else:
        assert result.id != existing.id
        assert result.merge_candidates == []


async def test_same_name_different_case_merges_despite_subthreshold_score():
    # 0.90 is the real similarity of these two strings, under the 0.92 merge
    # threshold. The name match has to carry the merge.
    existing = make_existing_entity(name="Ada Lovelace")
    graph_store, embedding_provider = make_mocks([(existing, 0.9019)])
    extracted = ExtractedEntity(name="Ada lovelace", entity_type="person")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id == existing.id
    assert result.name == existing.name
    assert result.merge_candidates == []
    assert len(result.claims) == len(existing.claims) + 1


async def test_same_name_extra_whitespace_merges():
    existing = make_existing_entity(name="Ada Lovelace")
    graph_store, embedding_provider = make_mocks([(existing, 0.88)])
    extracted = ExtractedEntity(name="  Ada   Lovelace ", entity_type="person")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id == existing.id


async def test_name_match_beats_a_higher_scored_different_name_candidate():
    # The exact-name candidate need not be first in the list: a closer-by-vector
    # but differently-named candidate must not win over an exact name match.
    other = make_existing_entity(name="Ada Byron", entity_type="person")
    exact = make_existing_entity(name="Ada Lovelace", entity_type="person")
    graph_store, embedding_provider = make_mocks([(other, 0.91), (exact, 0.905)])
    extracted = ExtractedEntity(name="ada lovelace", entity_type="person")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id == exact.id


async def test_embeds_name_only_not_type():
    graph_store, embedding_provider = make_mocks([])

    await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    embedding_provider.embed.assert_awaited_once_with("Pedro Costa")


async def test_searches_without_a_hard_type_filter():
    # Candidates of every type are fetched; the resolver weighs type itself.
    graph_store, embedding_provider = make_mocks([])

    await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    graph_store.find_similar_entities.assert_awaited_once()
    args = graph_store.find_similar_entities.await_args.args
    assert args[0] == SCOPE
    assert args[1] == QUERY_EMBEDDING
    assert args[2] is None


# --- type is a tiebreaker, not a filter


async def test_same_name_different_type_merges_into_one_node():
    """ "Acme" the Organization and "Acme" the Product become one entity with
    both types recorded."""
    existing = make_existing_entity(
        name="Acme", entity_type="Organization", observed_types=["Organization"]
    )
    graph_store, embedding_provider = make_mocks([(existing, 0.97)])
    extracted = ExtractedEntity(name="Acme", entity_type="Product")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id == existing.id
    assert result.entity_type == "Organization"  # the first type seen is kept
    assert set(result.observed_types) == {"Organization", "Product"}


async def test_same_name_different_type_merges_even_below_vector_threshold():
    existing = make_existing_entity(
        name="Acme", entity_type="Organization", observed_types=["Organization"]
    )
    graph_store, embedding_provider = make_mocks([(existing, 0.40)])
    extracted = ExtractedEntity(name="Acme", entity_type="Product")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id == existing.id
    assert set(result.observed_types) == {"Organization", "Product"}


async def test_different_name_different_type_stays_separate():
    existing = make_existing_entity(
        name="Acme", entity_type="Organization", observed_types=["Organization"]
    )
    graph_store, embedding_provider = make_mocks([(existing, 0.55)])
    extracted = ExtractedEntity(name="Globex", entity_type="Product")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id != existing.id
    assert result.entity_type == "Product"
    assert result.observed_types == ["Product"]


async def test_type_mismatch_penalty_keeps_borderline_vector_match_as_candidate():
    """A different-type candidate sitting just above the merge threshold is
    nudged down by the penalty to a candidate (not an auto-merge), so type
    still carries weight on the fuzzy path without hard-blocking."""
    existing = make_existing_entity(
        name="Acme Corp", entity_type="Organization", observed_types=["Organization"]
    )
    # 0.93 raw - 0.05 penalty = 0.88 -> below 0.92 merge, above 0.80 candidate.
    graph_store, embedding_provider = make_mocks([(existing, 0.93)])
    extracted = ExtractedEntity(name="Acme Product Line", entity_type="Product")

    result = await resolve_entity(extracted, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.id != existing.id
    assert result.merge_candidates == [existing.id]


async def test_new_entity_records_its_own_type_in_observed_types():
    graph_store, embedding_provider = make_mocks([])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    assert result.observed_types == ["person"]


async def test_only_top_candidate_decides_outcome_even_with_multiple_results():
    best = make_existing_entity(name="Best")
    second = make_existing_entity(name="Second")
    graph_store, embedding_provider = make_mocks([(best, 0.95), (second, 0.99)])

    result = await resolve_entity(EXTRACTED, SCOPE, PROVENANCE, graph_store, embedding_provider)

    # First candidate in the list wins regardless of relative score ordering --
    # find_similar_entities is responsible for returning best-first.
    assert result.id == best.id
    assert result.name == best.name


async def test_second_mention_in_one_write_merges_into_the_pending_entity():
    # Nothing is stored yet, so the store reports no candidates for either mention.
    graph_store, embedding_provider = make_mocks([])
    first = await resolve_entity(
        ExtractedEntity(name="Acme", entity_type="org", properties={"hq": "Lisbon"}),
        SCOPE,
        PROVENANCE,
        graph_store,
        embedding_provider,
    )

    second = await resolve_entity(
        ExtractedEntity(name="ACME", entity_type="org", properties={"size": 40}),
        SCOPE,
        PROVENANCE,
        graph_store,
        embedding_provider,
        pending=[first],
    )

    assert second.id == first.id
    assert second.properties == {"hq": "Lisbon", "size": 40}


async def test_repeat_merge_into_a_stored_entity_builds_on_the_pending_state():
    stored = make_existing_entity(name="Acme Corporation", entity_type="org")
    graph_store, embedding_provider = make_mocks([(stored, 0.95)])
    first = await resolve_entity(
        ExtractedEntity(name="Acme Corp", entity_type="org", properties={"hq": "Lisbon"}),
        SCOPE,
        PROVENANCE,
        graph_store,
        embedding_provider,
    )

    # The store still returns the stale stored entity for the second mention.
    second = await resolve_entity(
        ExtractedEntity(name="Acme Inc", entity_type="org", properties={"size": 40}),
        SCOPE,
        PROVENANCE,
        graph_store,
        embedding_provider,
        pending=[first],
    )

    assert second.id == stored.id
    assert second.properties == {"hq": "Lisbon", "size": 40}
