"""The two graph retrievers over a small in-memory graph:

Marie Curie -- Warsaw -- Poland -- Europe
     |
   Paris        Tokyo -- Japan   (unconnected to the rest)
"""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from graphrag_lab.benchmark.context import Triple, count_tokens, pack
from graphrag_lab.benchmark.retrievers import BfsRetriever, GraphSnapshot, PageRankRetriever
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope

SCOPE = Scope.from_dict({"tenant_id": "bench"})
PROVENANCE = Provenance(source="test")
FACTS = [
    ("Marie Curie", "born_in", "Warsaw"),
    ("Marie Curie", "worked_in", "Paris"),
    ("Warsaw", "capital_of", "Poland"),
    ("Poland", "part_of", "Europe"),
    ("Tokyo", "capital_of", "Japan"),
]


@pytest.fixture
async def graph() -> tuple[GraphSnapshot, dict[str, UUID]]:
    names = sorted({name for subject, _, obj in FACTS for name in (subject, obj)})
    entities = {
        name: Entity(name=name, entity_type="thing", scope=SCOPE, provenance=PROVENANCE)
        for name in names
    }
    relations = [
        Relation(
            source_entity_id=entities[subject].id,
            target_entity_id=entities[obj].id,
            relation_type=predicate,
            scope=SCOPE,
            provenance=PROVENANCE,
        )
        for subject, predicate, obj in FACTS
    ]
    store = AsyncMock(spec=GraphStore)
    store.find_entities.return_value = list(entities.values())
    store.project_display_edges.return_value = relations
    snapshot = await GraphSnapshot.load(store, SCOPE)
    return snapshot, {name: entity.id for name, entity in entities.items()}


def seeder(*ids: UUID):
    async def seed(question: str) -> list[UUID]:
        return list(ids)

    return seed


def facts(context) -> list[tuple[str, str, str]]:
    return [(t.subject, t.predicate, t.object) for t in context.triples]


async def test_bfs_stops_at_its_depth(graph):
    snapshot, ids = graph
    retriever = BfsRetriever(snapshot, seeder(ids["Marie Curie"]), depth=1)

    context = await retriever.retrieve("q", token_budget=1000)

    assert set(facts(context)) == {FACTS[0], FACTS[1]}


async def test_bfs_puts_nearer_facts_first(graph):
    snapshot, ids = graph
    retriever = BfsRetriever(snapshot, seeder(ids["Marie Curie"]), depth=2)

    context = await retriever.retrieve("q", token_budget=1000)

    assert set(facts(context)[:2]) == {FACTS[0], FACTS[1]}
    assert facts(context)[2:] == [FACTS[2]]


async def test_pagerank_reaches_past_any_fixed_depth_but_not_other_components(graph):
    snapshot, ids = graph
    retriever = PageRankRetriever(snapshot, seeder(ids["Marie Curie"]))

    context = await retriever.retrieve("q", token_budget=1000)

    assert set(facts(context)) == set(FACTS[:4])
    assert facts(context)[-1] == FACTS[3]  # the farthest fact ranks last


async def test_pagerank_favours_the_path_between_two_seeds(graph):
    snapshot, ids = graph
    retriever = PageRankRetriever(snapshot, seeder(ids["Marie Curie"], ids["Poland"]))

    context = await retriever.retrieve("q", token_budget=1000)

    ranked = facts(context)
    assert ranked.index(FACTS[0]) < ranked.index(FACTS[1])


async def test_retrievers_respect_the_token_budget(graph):
    snapshot, ids = graph
    budget = 9
    for retriever in (
        BfsRetriever(snapshot, seeder(ids["Marie Curie"]), depth=2),
        PageRankRetriever(snapshot, seeder(ids["Marie Curie"])),
    ):
        context = await retriever.retrieve("q", token_budget=budget)

        assert 0 < context.tokens <= budget
        assert context.tokens == count_tokens(context.text)


async def test_no_seeds_means_an_empty_context(graph):
    snapshot, _ = graph
    for retriever in (BfsRetriever(snapshot, seeder()), PageRankRetriever(snapshot, seeder())):
        context = await retriever.retrieve("q", token_budget=1000)

        assert context.triples == () and context.tokens == 0


async def test_snapshot_drops_edges_to_entities_outside_the_scope(graph):
    snapshot, ids = graph
    store = AsyncMock(spec=GraphStore)
    visible = [
        Entity(id=ids[name], name=name, entity_type="thing", scope=SCOPE, provenance=PROVENANCE)
        for name in ("Marie Curie", "Warsaw")
    ]
    store.find_entities.return_value = visible
    store.project_display_edges.return_value = list(snapshot.relations)

    partial = await GraphSnapshot.load(store, SCOPE)

    assert [partial.triple(r).text for r in partial.relations] == ["Marie Curie born in Warsaw."]


def test_pack_keeps_a_prefix_of_the_ranking():
    def triple(obj: str) -> Triple:
        return Triple("A", "knows", obj, UUID(int=0))

    # The long second triple does not fit, so the short third one is not
    # taken either: the context must stay a prefix of the ranking.
    ranked = [triple("B"), triple("C D E F G H I J"), triple("K")]

    context = pack(ranked, token_budget=6)

    assert [t.object for t in context.triples] == ["B"]
    assert context.tokens == 4
