"""The full write and read path against a real Neo4j and the real local
embedding model, with only the LLM extraction step stubbed. Needs no API key,
so CI exercises entity resolution through Neo4j's vector index, claim
adjudication, and hybrid recall on every run.

Scenario: two sources agree on where Sarah Chen works (written with different
surface forms, which must resolve to the same entities), then a third source
asserts a different, current employer, which must surface as a dispute rather
than silently overwrite the others.
"""

import pytest

from graphrag_lab.core import service
from graphrag_lab.core.config import get_settings
from graphrag_lab.extraction.schemas import ExtractedEntity, ExtractedRelation
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import EmbeddingProvider, build_embedding_provider

EXTRACTIONS: dict[str, tuple[list[ExtractedEntity], list[ExtractedRelation]]] = {
    "write-1": (
        [
            ExtractedEntity(name="Dr. Sarah Chen", entity_type="Person"),
            ExtractedEntity(name="Helios Therapeutics", entity_type="Organization"),
        ],
        [
            ExtractedRelation(
                source_name="Dr. Sarah Chen",
                target_name="Helios Therapeutics",
                relation_type="works_at",
            )
        ],
    ),
    "write-2": (
        [
            ExtractedEntity(name="Sarah Chen", entity_type="Person"),
            ExtractedEntity(name="Helios Therapeutics Inc.", entity_type="Organization"),
        ],
        [
            ExtractedRelation(
                source_name="Sarah Chen",
                target_name="Helios Therapeutics Inc.",
                relation_type="works_at",
            )
        ],
    ),
    "write-3": (
        [
            ExtractedEntity(name="Sarah Chen", entity_type="Person"),
            ExtractedEntity(name="Cascade Labs", entity_type="Organization"),
        ],
        [
            ExtractedRelation(
                source_name="Sarah Chen",
                target_name="Cascade Labs",
                relation_type="works_at",
                replaces_hint=True,
            )
        ],
    ),
}


@pytest.fixture
def stub_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract(
        content: str, existing_relation_types: list[str] | None = None
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation], int]:
        entities, relations = EXTRACTIONS[content]
        return entities, relations, 0

    monkeypatch.setattr(service, "extract", fake_extract)


@pytest.fixture(scope="module")
def embeddings() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())


async def test_resolution_corroboration_dispute_and_recall(
    store, tenant_id, embeddings, stub_extraction
):
    scope = Scope.from_dict({"tenant_id": tenant_id})

    async def write(content: str, source: str) -> None:
        await service.remember(content, scope, source, store, embeddings)

    await write("write-1", "agent_a")
    await write("write-2", "agent_b")

    # Surface variants resolved onto the first writer's entities via vector search.
    entities = {e.name: e for e in await store.find_entities(scope)}
    assert set(entities) == {"Dr. Sarah Chen", "Helios Therapeutics"}

    (claim,) = await store.project_display_edges(scope)
    assert claim.status == "active"
    assert sorted(claim.contributing_sources) == ["agent_a", "agent_b"]

    # A different source asserts a new current employer: dispute, not overwrite.
    await write("write-3", "agent_c")
    (cascade,) = await store.find_entities(scope, name="Cascade Labs")
    disputed = await store.find_disputed_claims(scope)
    assert {c.object_id for c in disputed} == {entities["Helios Therapeutics"].id, cascade.id}
    assert all(c.status == "disputed" for c in disputed)

    # Hybrid recall seeds on the person and reaches both contested employers.
    result = await service.recall(
        "Where does Sarah Chen work?",
        scope,
        store,
        embeddings,
        limit=1,
        retrieval_mode="hybrid",
        traversal_depth=1,
    )
    names = {entity.name for entity in result.entities}
    assert {"Dr. Sarah Chen", "Helios Therapeutics", "Cascade Labs"} <= names
    assert {relation.status for relation in result.relations} == {"disputed"}
