"""recall(): `limit` bounds the seeds only, `max_entities` bounds everything."""

from unittest.mock import AsyncMock

from graphrag_lab.core.service import recall
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import EmbeddingProvider

SCOPE = Scope.from_dict({"tenant_id": "acme", "user_id": "u_1"})
PROVENANCE = Provenance(source="agent_1")
QUERY_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


def make_entity(**overrides) -> Entity:
    defaults = dict(scope=SCOPE, provenance=PROVENANCE)
    defaults.update(overrides)
    return Entity(**defaults)


def make_mocks(
    seeds: list[tuple[Entity, float]],
    traversal_results: list[tuple[Entity, list[Relation]]],
    truncated: bool = False,
) -> tuple[GraphStore, EmbeddingProvider]:
    graph_store = AsyncMock(spec=GraphStore)
    graph_store.find_similar_entities.return_value = seeds
    graph_store.find_by_fulltext.return_value = []
    # traverse_from_seeds returns the seeds along with what they reach.
    reached_entities = [entity for entity, _ in seeds]
    reached_relations: list[Relation] = []
    for entity, relations in traversal_results:
        reached_entities.append(entity)
        reached_relations.extend(relations)
    graph_store.traverse_from_seeds.return_value = (
        reached_entities,
        reached_relations,
        truncated,
    )
    embedding_provider = AsyncMock(spec=EmbeddingProvider)
    embedding_provider.embed.return_value = QUERY_EMBEDDING
    return graph_store, embedding_provider


async def test_traversal_neighbours_survive_a_tight_seed_limit():
    seed = make_entity(name="Acme Corp", entity_type="Organization")
    neighbour = make_entity(name="Product Y", entity_type="Product")
    relation = Relation(
        source_entity_id=seed.id,
        target_entity_id=neighbour.id,
        relation_type="depends_on",
        scope=SCOPE,
        provenance=PROVENANCE,
    )
    graph_store, embedding_provider = make_mocks(
        seeds=[(seed, 0.95)],
        traversal_results=[(neighbour, [relation])],
    )

    result = await recall(
        query="Acme Corp customer shipment",
        scope=SCOPE,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        limit=1,
        # Pinning the mode skips the classifier.
        retrieval_mode="hybrid",
    )

    # With limit=1 the neighbour must still be there.
    entity_ids = {entity.id for entity in result.entities}
    assert entity_ids == {seed.id, neighbour.id}
    assert result.relations == [relation]


async def test_limit_bounds_the_vector_seed_step_not_the_final_result():
    seed = make_entity(name="Acme Corp", entity_type="Organization")
    graph_store, embedding_provider = make_mocks(seeds=[(seed, 0.95)], traversal_results=[])

    await recall(
        query="Acme Corp customer shipment",
        scope=SCOPE,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        limit=1,
        retrieval_mode="hybrid",
    )

    graph_store.find_similar_entities.assert_awaited_once()
    args = graph_store.find_similar_entities.await_args.args
    assert args[3] == 1


async def test_max_entities_caps_the_seeds_too():
    seeds = [
        (make_entity(name=f"Seed {i}", entity_type="Organization"), 0.9 - i * 0.01)
        for i in range(5)
    ]
    graph_store, embedding_provider = make_mocks(seeds, [])
    kept = [entity for entity, _ in seeds[:2]]
    graph_store.traverse_from_seeds.return_value = (kept, [], False)

    result = await recall(
        "q",
        SCOPE,
        graph_store,
        embedding_provider,
        limit=5,
        traversal_depth=1,
        max_entities=2,
    )

    assert [entity.id for entity in result.entities] == [entity.id for entity in kept]
    assert result.truncated is True
    assert graph_store.traverse_from_seeds.await_args.args[1] == [str(e.id) for e in kept]
