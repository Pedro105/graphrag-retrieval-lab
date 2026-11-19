"""Graph side of the benchmark: loading the gold graph into Neo4j, and the
graph retrievers that query it.

The gold triples are written through the same `:Entity` / `:Claim` schema
the API produces, so `GraphBFSRetriever` is literally the system's own
`recall()` (hybrid seeding + breadth-first traversal) and every graph
retriever reads from Neo4j rather than from an in-memory copy.
"""

import uuid
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import structlog
from scipy import sparse

from graphrag_lab.bench.context import UnitKind
from graphrag_lab.bench.data import Corpus, Question, Triple
from graphrag_lab.bench.encoding import Encoder, EncoderEmbeddingProvider
from graphrag_lab.bench.retrievers import Ranking, Retriever, triple_unit
from graphrag_lab.core.service import normalize_predicate, recall, seed_entities
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope

logger = structlog.get_logger()

SOURCE = "2wikimultihopqa"
_NAMESPACE = uuid.UUID("6f1d4c56-1b8e-4f4a-9d1e-2b7f0c0a7e11")


def entity_id(name: str) -> uuid.UUID:
    """Deterministic ids, so reloading the same corpus yields the same graph."""
    return uuid.uuid5(_NAMESPACE, f"entity:{name}")


def claim_id(triple: Triple) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"claim:{triple.text}")


async def load_gold_graph(
    store: GraphStore, corpus: Corpus, encoder: Encoder, tenant_id: str
) -> None:
    """Write the corpus's gold triples into `tenant_id`, unless an identical
    graph is already there. Entity names are embedded the same way the API
    embeds them (no query instruction), so seeding behaves as in production."""
    names = corpus.entities
    if await store.graph_stats(tenant_id) == (len(names), len(corpus.triples)):
        logger.info("bench.graph_reused", tenant_id=tenant_id, entities=len(names))
        return

    await store.drop_graph(tenant_id)
    scope = Scope.from_dict({"tenant_id": tenant_id})
    provenance = Provenance(source=SOURCE)
    embeddings = encoder.documents(names, "entities")
    await store.ensure_graph_initialized(tenant_id, int(embeddings.shape[1]))

    entities = [
        Entity(
            id=entity_id(name),
            name=name,
            entity_type="Entity",
            observed_types=["Entity"],
            scope=scope,
            provenance=provenance,
            embedding=[float(x) for x in vector],
            contributing_sources=[SOURCE],
        )
        for name, vector in zip(names, embeddings, strict=True)
    ]
    claims = [
        FactClaim(
            id=claim_id(triple),
            predicate=normalize_predicate(triple.predicate),
            raw_predicate=triple.predicate,
            subject_id=entity_id(triple.subject),
            object_id=entity_id(triple.object),
            status="active",
            asserted_by=[SOURCE],
            scope=scope,
        )
        for triple in corpus.triples
    ]
    await store.bulk_write(tenant_id, entities, claims, memory_id=f"bench-{corpus.fingerprint}")
    logger.info(
        "bench.graph_loaded", tenant_id=tenant_id, entities=len(entities), claims=len(claims)
    )


class GraphBFSRetriever(Retriever):
    """The system as built: hybrid (vector + full-text) entity seeding, then
    breadth-first expansion over live claims. Triples come back in discovery
    order -- the traversal does not rank what it finds."""

    unit_kind: UnitKind = "triple"

    def __init__(
        self,
        store: GraphStore,
        tenant_id: str,
        encoder: Encoder,
        depth: int,
        n_seeds: int = 5,
        max_entities: int = 200,
    ) -> None:
        self.name = f"graph_bfs_d{depth}"
        self._store = store
        self._scope = Scope.from_dict({"tenant_id": tenant_id})
        self._provider = EncoderEmbeddingProvider(encoder)
        self._depth = depth
        self._n_seeds = n_seeds
        self._max_entities = max_entities

    async def rank(self, question: Question) -> Ranking:
        result = await recall(
            question.text,
            self._scope,
            self._store,
            self._provider,
            limit=self._n_seeds,
            traversal_depth=self._depth,
            retrieval_mode="hybrid",
            max_entities=self._max_entities,
        )
        names = {entity.id: entity.name for entity in result.entities}
        seeds = [entity.name for entity in result.entities[: result.stats.seeds_found]]
        units = [
            triple_unit(
                Triple(names[r.source_entity_id], r.relation_type, names[r.target_entity_id])
            )
            for r in result.relations
        ]
        return Ranking(
            units,
            {"seed_hit": question.topic_entity in seeds, "truncated": result.truncated},
        )


def personalized_pagerank(
    adjacency: sparse.csr_matrix,
    seeds: Sequence[int],
    restart: float = 0.15,
    iterations: int = 50,
    tolerance: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Random walk with restart to `seeds` on an undirected graph.

    `adjacency` is a symmetric (n x n) matrix; columns are normalized into a
    transition matrix. Returns the stationary visit probabilities.
    """
    n = adjacency.shape[0]
    personalization = np.zeros(n)
    if not seeds:
        return personalization
    personalization[list(seeds)] = 1.0 / len(seeds)
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree, dtype=float), where=degree > 0)
    transition = adjacency @ sparse.diags(inverse)
    scores = personalization.copy()
    for _ in range(iterations):
        updated = restart * personalization + (1 - restart) * (transition @ scores)
        if np.abs(updated - scores).sum() < tolerance:
            return np.asarray(updated)
        scores = updated
    return np.asarray(scores)


class GraphPPRRetriever(Retriever):
    """HippoRAG-style: the same seeds as the system, then personalized
    PageRank over the whole graph; a triple scores the sum of its endpoints'
    PageRank. Ranks facts by structural proximity to the question's entities
    instead of returning every neighbour unranked."""

    name = "graph_ppr"
    unit_kind: UnitKind = "triple"

    def __init__(
        self,
        store: GraphStore,
        tenant_id: str,
        encoder: Encoder,
        n_seeds: int = 5,
        n_triples: int = 150,
        restart: float = 0.15,
    ) -> None:
        self._store = store
        self._scope = Scope.from_dict({"tenant_id": tenant_id})
        self._provider = EncoderEmbeddingProvider(encoder)
        self._n_seeds = n_seeds
        self._n_triples = n_triples
        self._restart = restart
        self._index: dict[uuid.UUID, int] = {}
        self._triples: list[Triple] = []
        self._endpoints = np.zeros((0, 2), dtype=np.int64)
        self._adjacency = sparse.csr_matrix((0, 0))

    async def build(self) -> "GraphPPRRetriever":
        """Read the graph out of Neo4j once and build the sparse adjacency."""
        entities = await self._store.find_entities(self._scope)
        relations = await self._store.project_display_edges(self._scope)
        self._index = {entity.id: i for i, entity in enumerate(entities)}
        names = [entity.name for entity in entities]
        self._triples = [
            Triple(
                names[self._index[r.source_entity_id]],
                r.relation_type,
                names[self._index[r.target_entity_id]],
            )
            for r in relations
        ]
        self._endpoints = np.array(
            [(self._index[r.source_entity_id], self._index[r.target_entity_id]) for r in relations],
            dtype=np.int64,
        ).reshape(-1, 2)
        n = len(entities)
        rows = np.concatenate([self._endpoints[:, 0], self._endpoints[:, 1]])
        cols = np.concatenate([self._endpoints[:, 1], self._endpoints[:, 0]])
        self._adjacency = sparse.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
        return self

    async def rank(self, question: Question) -> Ranking:
        seeds = await seed_entities(
            question.text, self._scope, self._store, self._provider, self._n_seeds, "hybrid"
        )
        seed_rows = [self._index[s.id] for s in seeds if s.id in self._index]
        scores = personalized_pagerank(self._adjacency, seed_rows, restart=self._restart)
        triple_scores = scores[self._endpoints[:, 0]] + scores[self._endpoints[:, 1]]
        order = np.argsort(-triple_scores, kind="stable")[: self._n_triples]
        units = [triple_unit(self._triples[i]) for i in order if triple_scores[i] > 0]
        return Ranking(units, {"seed_hit": question.topic_entity in {s.name for s in seeds}})
