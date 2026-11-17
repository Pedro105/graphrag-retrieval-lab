"""GraphStore on Neo4j.

Facts are `:Claim` nodes: `(:Claim)-[:SUBJECT]->(:Entity)` and
`(:Claim)-[:OBJECT]->(:Entity)`, with the predicate and status as properties.
Cypher is in graph/queries.py, conversion to and from models in
graph/schema.py.

Neo4j Community has one database, so a tenant is a label, `:Tenant_<id>`, on
every node it owns, with its own vector and full-text index (ADR 0004). A
vector search therefore ranks one tenant's entities and nothing else.
"""

import re
from typing import Any
from uuid import UUID

import structlog
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    EagerResult,
    NotificationDisabledClassification,
)
from neo4j.exceptions import ClientError

from graphrag_lab.graph import queries
from graphrag_lab.graph.schema import (
    claim_to_node_props,
    claim_to_relation,
    entity_to_node_props,
    node_to_claim,
    node_to_entity,
)
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.memory import Memory
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope
from graphrag_lab.retrieval.traversal import breadth_first_traverse

logger = structlog.get_logger()

_SAFE_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_FULLTEXT_TOKEN = re.compile(r"[A-Za-z0-9]+")
_LUCENE_OPERATORS = frozenset({"AND", "OR", "NOT"})
# Lucene refuses a query with more than 1024 clauses.
_MAX_FULLTEXT_TOKENS = 256
_DELETE_BATCH = 5_000


def tenant_label(tenant_id: str) -> str:
    """The tenant's label, backtick-quoted. Labels cannot be query parameters,
    so the id is checked against a strict character set first."""
    if not _SAFE_TENANT_ID.match(tenant_id):
        raise ValueError(f"tenant_id {tenant_id!r} must match ^[A-Za-z0-9_-]+$")
    return f"`Tenant_{tenant_id}`"


def tenant_query(template: str, tenant_id: str) -> str:
    return template.replace("__T__", tenant_label(tenant_id))


def vector_index_name(tenant_id: str) -> str:
    tenant_label(tenant_id)  # validates
    return f"entity_embedding_{tenant_id}"


def fulltext_index_name(tenant_id: str) -> str:
    tenant_label(tenant_id)  # validates
    return f"entity_name_{tenant_id}"


def sanitize_fulltext_query(query_text: str) -> str:
    """Reduce free text to alphanumeric tokens joined with OR.

    Lucene reads punctuation as syntax, so a raw "RG-88" means "RG NOT 88"
    and matches nothing. An uppercase AND/OR/NOT is an operator as well and
    is lowercased into an ordinary word.
    """
    tokens: dict[str, None] = {}
    for token in _FULLTEXT_TOKEN.findall(query_text):
        tokens.setdefault(token.lower() if token in _LUCENE_OPERATORS else token)
        if len(tokens) >= _MAX_FULLTEXT_TOKENS:
            break
    return " OR ".join(tokens)


def cosine_from_neo4j_score(score: float) -> float:
    """Neo4j reports cosine similarity as (1 + cos) / 2. Convert it back."""
    return 2.0 * score - 1.0


def _is_missing_index(exc: ClientError) -> bool:
    if exc.code == "Neo.ClientError.Schema.IndexNotFound":
        return True
    message = (exc.message or "").lower()
    return "no such" in message and "index" in message


class Neo4jGraphStore(GraphStore):
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "graphrag-lab-dev",
        database: str = "neo4j",
        driver: AsyncDriver | None = None,
    ) -> None:
        # An empty tenant has no :Claim nodes yet, which Neo4j would warn about
        # on every query.
        self._driver = driver or AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED
            ],
        )
        self._database = database
        self._constraints_ready = False
        self._initialized: set[str] = set()

    async def _run(
        self, query: str, params: dict[str, Any] | None = None, *, tenant_id: str | None = None
    ) -> EagerResult:
        if tenant_id is not None:
            query = tenant_query(query, tenant_id)
        return await self._driver.execute_query(
            query, parameters_=params or {}, database_=self._database
        )

    # --- schema --------------------------------------------------------------

    async def _ensure_constraints(self) -> None:
        if self._constraints_ready:
            return
        for statement in queries.CONSTRAINTS:
            await self._run(statement)
        self._constraints_ready = True

    async def ensure_graph_initialized(self, tenant_id: str, embedding_dimension: int) -> None:
        if tenant_id in self._initialized:
            return
        await self._ensure_constraints()
        label = tenant_label(tenant_id)
        vector_index = vector_index_name(tenant_id)
        existing = await self._run(
            "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
            {"name": vector_index},
        )
        if existing.records:
            config = existing.records[0]["options"]["indexConfig"]
            existing_dim = int(config["vector.dimensions"])
            if existing_dim != embedding_dimension:
                raise ValueError(
                    f"tenant {tenant_id!r} has a {existing_dim}-d vector index but the "
                    f"embedding provider produces {embedding_dimension}-d vectors; "
                    "drop the graph or switch providers"
                )
        else:
            # Index DDL takes no parameters.
            await self._run(
                f"CREATE VECTOR INDEX `{vector_index}` IF NOT EXISTS "
                f"FOR (n:{label}) ON n.embedding "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(embedding_dimension)}, "
                "`vector.similarity_function`: 'cosine'}}"
            )
        await self._run(
            f"CREATE FULLTEXT INDEX `{fulltext_index_name(tenant_id)}` IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [n.name]"
        )
        # Queries against an index fail until it has finished building.
        await self._run("CALL db.awaitIndexes(120)")
        self._initialized.add(tenant_id)

    # --- writes --------------------------------------------------------------

    async def write_memory(self, memory: Memory) -> str:
        memory_id = str(memory.id)
        tenant_id = memory.scope.tenant_id
        for entity in memory.entities:
            await self._run(
                queries.UPSERT_ENTITY,
                {"id": str(entity.id), "props": entity_to_node_props(entity, memory_id)},
                tenant_id=tenant_id,
            )
            if entity.embedding is not None:
                await self._run(
                    queries.SET_ENTITY_EMBEDDING,
                    {"id": str(entity.id), "embedding": entity.embedding},
                    tenant_id=tenant_id,
                )
        return memory_id

    async def upsert_claim(self, claim: FactClaim, memory_id: str) -> None:
        await self._run(
            queries.UPSERT_CLAIM,
            {
                "id": str(claim.id),
                "props": claim_to_node_props(claim, memory_id),
                "subject_id": str(claim.subject_id),
                "object_id": str(claim.object_id),
            },
            tenant_id=claim.scope.tenant_id,
        )

    # --- claim reads ---------------------------------------------------------

    async def _claims(
        self, query: str, scope: Scope, params: dict[str, Any] | None = None
    ) -> list[FactClaim]:
        result = await self._run(query, params, tenant_id=scope.tenant_id)
        claims = (node_to_claim(dict(record["c"])) for record in result.records)
        return [claim for claim in claims if scope.includes(claim.scope)]

    async def find_active_claims(
        self, scope: Scope, subject_id: UUID, predicate: str
    ) -> list[FactClaim]:
        return await self._claims(
            queries.FIND_ACTIVE_CLAIMS,
            scope,
            {"subject_id": str(subject_id), "predicate": predicate},
        )

    async def project_display_edges(self, scope: Scope) -> list[Relation]:
        return [
            claim_to_relation(claim)
            for claim in await self._claims(queries.PROJECT_LIVE_CLAIMS, scope)
        ]

    async def find_disputed_claims(self, scope: Scope) -> list[FactClaim]:
        return await self._claims(queries.DISPUTED_CLAIMS, scope)

    async def find_predicates(self, scope: Scope) -> list[str]:
        result = await self._run(queries.DISTINCT_PREDICATES, tenant_id=scope.tenant_id)
        return sorted({record["predicate"] for record in result.records if record["predicate"]})

    # --- entity reads --------------------------------------------------------

    async def get_entity(self, entity_id: UUID, tenant_id: str) -> Entity | None:
        result = await self._run(queries.GET_ENTITY, {"id": str(entity_id)}, tenant_id=tenant_id)
        if not result.records:
            return None
        return node_to_entity(dict(result.records[0]["n"]))

    async def find_entities(
        self,
        scope: Scope,
        name: str | None = None,
        entity_type: str | None = None,
    ) -> list[Entity]:
        result = await self._run(
            queries.FIND_ENTITIES,
            {"name": name, "entity_type": entity_type},
            tenant_id=scope.tenant_id,
        )
        entities = (node_to_entity(dict(record["n"])) for record in result.records)
        return [entity for entity in entities if scope.includes(entity.scope)]

    async def find_similar_entities(
        self,
        scope: Scope,
        embedding: list[float],
        entity_type: str | None,
        k: int,
    ) -> list[tuple[Entity, float]]:
        try:
            query = queries.VECTOR_SEARCH.replace(
                "__VINDEX__", f"`{vector_index_name(scope.tenant_id)}`"
            )
            result = await self._run(
                query, {"k": k, "embedding": embedding}, tenant_id=scope.tenant_id
            )
        except ClientError as exc:
            if _is_missing_index(exc):
                return []  # nothing has been written to this tenant
            raise
        matches: list[tuple[Entity, float]] = []
        for record in result.records:
            entity = node_to_entity(dict(record["node"]))
            if entity_type is not None and entity.entity_type != entity_type:
                continue
            if not scope.includes(entity.scope):
                continue
            matches.append((entity, cosine_from_neo4j_score(record["score"])))
        return matches

    async def find_by_fulltext(
        self,
        scope: Scope,
        query_text: str,
        limit: int,
    ) -> list[tuple[UUID, int]]:
        sanitized = sanitize_fulltext_query(query_text)
        if not sanitized:
            return []
        try:
            result = await self._run(
                queries.FULLTEXT_SEARCH,
                {"index": fulltext_index_name(scope.tenant_id), "q": sanitized, "limit": limit},
            )
        except ClientError as exc:
            if _is_missing_index(exc):
                return []
            logger.warning(
                "find_by_fulltext.query_failed", tenant_id=scope.tenant_id, error=str(exc)
            )
            return []
        matches: list[tuple[UUID, int]] = []
        for record in result.records:
            entity = node_to_entity(dict(record["node"]))
            if scope.includes(entity.scope):
                matches.append((entity.id, len(matches) + 1))
        return matches

    async def traverse_from_seeds(
        self,
        scope: Scope,
        seed_ids: list[str],
        depth: int,
        max_entities: int = 200,
    ) -> tuple[list[Entity], list[Relation], bool]:
        """One Cypher query per hop; the BFS itself is in retrieval/traversal.py."""
        tenant_id = scope.tenant_id

        async def fetch_seeds(ids: list[str]) -> list[Entity]:
            result = await self._run(queries.GET_ENTITIES, {"ids": ids}, tenant_id=tenant_id)
            entities = (node_to_entity(dict(record["n"])) for record in result.records)
            return [entity for entity in entities if scope.includes(entity.scope)]

        async def fetch_neighbours(frontier_ids: list[str]) -> list[tuple[Entity, Relation]]:
            result = await self._run(
                queries.FETCH_CLAIM_NEIGHBOURS, {"frontier_ids": frontier_ids}, tenant_id=tenant_id
            )
            pairs: list[tuple[Entity, Relation]] = []
            for record in result.records:
                neighbour = node_to_entity(dict(record["n"]))
                claim = node_to_claim(dict(record["c"]))
                # An edge outside the caller's scope is not walked, even if
                # both of its ends are visible.
                if scope.includes(neighbour.scope) and scope.includes(claim.scope):
                    pairs.append((neighbour, claim_to_relation(claim)))
            return pairs

        return await breadth_first_traverse(
            seed_ids, depth, max_entities, fetch_seeds, fetch_neighbours, tenant_id=tenant_id
        )

    # --- deletes, stats, lifecycle -------------------------------------------

    async def delete_memory(self, memory_id: str, tenant_id: str) -> bool:
        result = await self._run(
            queries.DELETE_MEMORY_WITH_CLAIMS, {"memory_id": memory_id}, tenant_id=tenant_id
        )
        return bool(result.summary.counters.nodes_deleted > 0)

    async def delete_entity(self, entity_id: UUID, tenant_id: str) -> bool:
        result = await self._run(
            queries.DELETE_ENTITY_WITH_CLAIMS, {"id": str(entity_id)}, tenant_id=tenant_id
        )
        return bool(result.summary.counters.nodes_deleted > 0)

    async def graph_stats(self, tenant_id: str) -> tuple[int, int]:
        entities = await self._run(queries.COUNT_ENTITIES, tenant_id=tenant_id)
        claims = await self._run(queries.COUNT_LIVE_CLAIMS, tenant_id=tenant_id)
        return int(entities.records[0]["count"]), int(claims.records[0]["count"])

    async def drop_graph(self, tenant_id: str) -> None:
        while True:
            result = await self._run(
                queries.DELETE_TENANT_BATCH, {"batch": _DELETE_BATCH}, tenant_id=tenant_id
            )
            if int(result.records[0]["deleted"]) == 0:
                break
        for index in (vector_index_name(tenant_id), fulltext_index_name(tenant_id)):
            await self._run(f"DROP INDEX `{index}` IF EXISTS")
        self._initialized.discard(tenant_id)

    async def health_check(self) -> bool:
        try:
            await self._driver.verify_connectivity()
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._driver.close()
