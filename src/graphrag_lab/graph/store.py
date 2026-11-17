"""The interface the rest of the code uses to reach the graph database."""

from abc import ABC, abstractmethod
from uuid import UUID

from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.memory import Memory
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope


class GraphStore(ABC):
    """Entities and claims in a property graph, one isolated graph per tenant.

    Methods that search take a `Scope` and filter on all of its keys. Methods
    that address one known id take just the `tenant_id`.
    """

    @abstractmethod
    async def ensure_graph_initialized(self, tenant_id: str, embedding_dimension: int) -> None:
        """Create the tenant's indexes if they are missing. Safe to call on
        every write."""

    @abstractmethod
    async def write_memory(self, memory: Memory) -> str:
        """Upsert the memory's entities and return the memory id. Facts are
        written separately with `upsert_claim`, once adjudicated."""

    @abstractmethod
    async def find_active_claims(
        self, scope: Scope, subject_id: UUID, predicate: str
    ) -> list[FactClaim]:
        """Active claims with this subject and normalized predicate: what a
        new assertion is adjudicated against."""

    @abstractmethod
    async def upsert_claim(self, claim: FactClaim, memory_id: str) -> None:
        """Create or update a claim by id and link it to its subject and
        object. Status changes go through here too."""

    @abstractmethod
    async def project_display_edges(self, scope: Scope) -> list[Relation]:
        """Every live (active or disputed) claim as a subject -> object edge."""

    @abstractmethod
    async def find_predicates(self, scope: Scope) -> list[str]:
        """Distinct raw predicates of the tenant's live claims."""

    @abstractmethod
    async def find_disputed_claims(self, scope: Scope) -> list[FactClaim]:
        """All disputed claims in scope."""

    @abstractmethod
    async def get_entity(self, entity_id: UUID, tenant_id: str) -> Entity | None:
        """One entity by id, or None."""

    @abstractmethod
    async def find_entities(
        self,
        scope: Scope,
        name: str | None = None,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Entities in scope, optionally filtered by exact name and type."""

    @abstractmethod
    async def find_similar_entities(
        self,
        scope: Scope,
        embedding: list[float],
        entity_type: str | None,
        k: int,
    ) -> list[tuple[Entity, float]]:
        """The `k` nearest entities to `embedding` in the tenant, as (entity,
        cosine similarity), most similar first.

        Scope and type are filtered after the search, so fewer than `k` can
        come back.
        """

    @abstractmethod
    async def find_by_fulltext(
        self,
        scope: Scope,
        query_text: str,
        limit: int,
    ) -> list[tuple[UUID, int]]:
        """Entities whose name matches `query_text`, as (entity_id, rank) with
        rank 1 the best match.

        Returns an empty list rather than raising when the tenant has no
        index yet or the query has no searchable tokens.
        """

    @abstractmethod
    async def traverse_from_seeds(
        self,
        scope: Scope,
        seed_ids: list[str],
        depth: int,
        max_entities: int = 200,
    ) -> tuple[list[Entity], list[Relation], bool]:
        """Breadth-first expansion over live claims, `depth` hops from the seeds.

        Returns (entities, relations, truncated): the seeds and everything
        reached, the relations among them, and whether `max_entities` cut the
        expansion short.
        """

    @abstractmethod
    async def delete_memory(self, memory_id: str, tenant_id: str) -> bool:
        """Delete the entities a memory created, with their claims. True if
        anything was deleted."""

    @abstractmethod
    async def delete_entity(self, entity_id: UUID, tenant_id: str) -> bool:
        """Delete one entity and the claims attached to it. False if the
        tenant has no such entity."""

    @abstractmethod
    async def graph_stats(self, tenant_id: str) -> tuple[int, int]:
        """(entity count, live claim count) for a tenant."""

    @abstractmethod
    async def drop_graph(self, tenant_id: str) -> None:
        """Delete everything a tenant owns, indexes included. A no-op for a
        tenant that does not exist."""

    @abstractmethod
    async def health_check(self) -> bool:
        """True if the database is reachable."""
