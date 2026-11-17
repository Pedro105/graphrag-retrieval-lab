"""Graph retrievers behind one interface: `retrieve(question, token_budget)`.

Both retrievers here start from the same seed entities and work on the same
snapshot of the graph. They differ only in how they rank the facts around
the seeds, which is the thing the benchmark compares.
"""

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from graphrag_lab.benchmark.context import RetrievedContext, Triple, pack
from graphrag_lab.core.service import seed_entities
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.recall import RetrievalMode
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope
from graphrag_lab.retrieval.pagerank import personalized_pagerank
from graphrag_lab.vector.embeddings import EmbeddingProvider

# Maps a question to seed entity ids, best first.
Seeder = Callable[[str], Awaitable[list[UUID]]]


class Retriever(Protocol):
    name: str

    async def retrieve(self, question: str, token_budget: int) -> RetrievedContext: ...


@dataclass(frozen=True)
class GraphSnapshot:
    """Every entity name and live fact of one graph, held in memory. Benchmark
    graphs do not change during a run, so one load serves every question."""

    names: dict[UUID, str]
    relations: tuple[Relation, ...]

    @classmethod
    async def load(cls, graph_store: GraphStore, scope: Scope) -> "GraphSnapshot":
        entities = await graph_store.find_entities(scope)
        names = {entity.id: entity.name for entity in entities}
        relations = [
            relation
            for relation in await graph_store.project_display_edges(scope)
            if relation.source_entity_id in names and relation.target_entity_id in names
        ]
        # Sorted so that ties in a ranking break the same way on every run.
        relations.sort(key=lambda relation: str(relation.id))
        return cls(names, tuple(relations))

    def triple(self, relation: Relation) -> Triple:
        return Triple(
            subject=self.names[relation.source_entity_id],
            predicate=relation.relation_type,
            object=self.names[relation.target_entity_id],
            claim_id=relation.id,
        )


def make_seeder(
    graph_store: GraphStore,
    embedding_provider: EmbeddingProvider,
    scope: Scope,
    limit: int = 10,
    mode: RetrievalMode = "hybrid",
) -> Seeder:
    """The seeding step recall() uses, with the mode pinned so that no
    classifier call can change it between runs."""

    async def seed(question: str) -> list[UUID]:
        entities = await seed_entities(
            question, scope, graph_store, embedding_provider, limit, mode
        )
        return [entity.id for entity in entities]

    return seed


class BfsRetriever:
    """Facts within `depth` hops of the seeds, nearest first.

    BFS has no notion of relevance beyond distance, so within one hop the
    order is arbitrary (but fixed). That is the weakness the ranked
    retrievers are meant to fix.
    """

    def __init__(self, snapshot: GraphSnapshot, seeder: Seeder, depth: int = 2) -> None:
        self.name = f"bfs-{depth}"
        self._snapshot = snapshot
        self._seeder = seeder
        self._depth = depth
        self._neighbours: dict[UUID, list[UUID]] = {}
        for relation in snapshot.relations:
            a, b = relation.source_entity_id, relation.target_entity_id
            self._neighbours.setdefault(a, []).append(b)
            self._neighbours.setdefault(b, []).append(a)

    def _distances(self, seeds: list[UUID]) -> dict[UUID, int]:
        distance = {seed: 0 for seed in seeds if seed in self._snapshot.names}
        queue = deque(distance)
        while queue:
            node = queue.popleft()
            if distance[node] == self._depth:
                continue
            for neighbour in self._neighbours.get(node, ()):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        return distance

    async def retrieve(self, question: str, token_budget: int) -> RetrievedContext:
        distance = self._distances(await self._seeder(question))
        reached = [
            relation
            for relation in self._snapshot.relations
            if relation.source_entity_id in distance and relation.target_entity_id in distance
        ]
        # A fact is as far away as its farther end.
        reached.sort(key=lambda r: max(distance[r.source_entity_id], distance[r.target_entity_id]))
        return pack((self._snapshot.triple(relation) for relation in reached), token_budget)


class PageRankRetriever:
    """Facts ranked by personalized PageRank from the seeds (as in HippoRAG).

    A fact scores the sum of its two entities' scores. There is no depth
    limit: the walk decides how far is worth going.
    """

    def __init__(self, snapshot: GraphSnapshot, seeder: Seeder, damping: float = 0.5) -> None:
        self.name = "ppr"
        self._snapshot = snapshot
        self._seeder = seeder
        self._damping = damping
        self._edges = [(r.source_entity_id, r.target_entity_id) for r in snapshot.relations]

    async def retrieve(self, question: str, token_budget: int) -> RetrievedContext:
        seeds = [seed for seed in await self._seeder(question) if seed in self._snapshot.names]
        scores = personalized_pagerank(self._edges, seeds, damping=self._damping)

        def score(relation: Relation) -> float:
            return scores.get(relation.source_entity_id, 0.0) + scores.get(
                relation.target_entity_id, 0.0
            )

        ranked = sorted(
            (relation for relation in self._snapshot.relations if score(relation) > 0.0),
            key=score,
            reverse=True,
        )
        return pack((self._snapshot.triple(relation) for relation in ranked), token_budget)
