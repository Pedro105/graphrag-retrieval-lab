"""v1 API routes: ingest (remember), query (recall), and graph inspection.

There is no authentication: this is a local research service. Every request
names the graph it operates on via `tenant_id` (default "default"); each
tenant is a structurally separate FalkorDB graph (see docs/decisions/0002).
Callers may add other scope keys (user_id, project, ...) via `extra_scope`,
used for in-graph filtering (Scope.includes).
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from contextstore.api.dependencies import EmbeddingProviderDep, GraphStoreDep
from contextstore.core.service import recall as recall_service
from contextstore.core.service import remember as remember_service
from contextstore.models.claim import Claim
from contextstore.models.entity import Entity
from contextstore.models.memory import Memory
from contextstore.models.recall import RecallResult, RetrievalMode
from contextstore.models.relation import Relation
from contextstore.models.scope import Scope, ScopeValue

router = APIRouter(prefix="/v1")

DEFAULT_TENANT = "default"
_TENANT_PATTERN = r"^[A-Za-z0-9_-]+$"

TenantQuery = Query(default=DEFAULT_TENANT, pattern=_TENANT_PATTERN)


def _build_scope(tenant_id: str, extra_scope: dict[str, ScopeValue]) -> Scope:
    merged: dict[str, ScopeValue] = {k: v for k, v in extra_scope.items() if k != "tenant_id"}
    merged["tenant_id"] = tenant_id
    return Scope.from_dict(merged)


class ScopedRequest(BaseModel):
    tenant_id: str = Field(default=DEFAULT_TENANT, pattern=_TENANT_PATTERN)
    extra_scope: dict[str, ScopeValue] = Field(
        default_factory=dict,
        description="Optional extra scope keys (e.g. user_id, project) within the graph.",
    )


class RememberRequest(ScopedRequest):
    content: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Who asserted this (agent / document id).")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] | None = None


class RecallRequest(ScopedRequest):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, description="Max seed entities before traversal.")
    traversal_depth: int | None = Field(
        default=None,
        ge=1,
        description="Hops from the seeds. Unset: the query classifier chooses.",
    )
    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        description="Seeding method: vector, fulltext or hybrid. Unset: the classifier chooses.",
    )
    max_entities: int = Field(default=200, ge=1, description="Hard cap on traversal size.")
    synthesise: bool = Field(default=False, description="Also generate a prose answer.")


@router.post("/memories", response_model=Memory)
async def create_memory(
    body: RememberRequest,
    graph_store: GraphStoreDep,
    embedding_provider: EmbeddingProviderDep,
) -> Memory:
    memory, _tokens_used = await remember_service(
        content=body.content,
        scope=_build_scope(body.tenant_id, body.extra_scope),
        source=body.source,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        confidence=body.confidence,
        evidence=body.evidence,
    )
    return memory


@router.post("/recall", response_model=RecallResult)
async def recall_memories(
    body: RecallRequest,
    graph_store: GraphStoreDep,
    embedding_provider: EmbeddingProviderDep,
) -> RecallResult:
    return await recall_service(
        query=body.query,
        scope=_build_scope(body.tenant_id, body.extra_scope),
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        limit=body.limit,
        traversal_depth=body.traversal_depth,
        retrieval_mode=body.retrieval_mode,
        max_entities=body.max_entities,
        synthesise=body.synthesise,
    )


class GraphSnapshot(BaseModel):
    """Every entity and live relation in one graph, for inspection/visualization."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


@router.get("/graph", response_model=GraphSnapshot)
async def get_graph(graph_store: GraphStoreDep, tenant_id: str = TenantQuery) -> GraphSnapshot:
    scope = Scope.from_dict({"tenant_id": tenant_id})
    entities = await graph_store.find_entities(scope)
    if not entities:
        return GraphSnapshot()
    relations = await graph_store.project_display_edges(scope)
    return GraphSnapshot(entities=entities, relations=relations)


@router.delete("/graph", status_code=status.HTTP_204_NO_CONTENT)
async def drop_graph(graph_store: GraphStoreDep, tenant_id: str = TenantQuery) -> None:
    """Delete an entire graph. Idempotent."""
    await graph_store.drop_graph(tenant_id)


class AnnotatedClaim(BaseModel):
    claim: Claim
    superseded: bool


class EntityClaimsResponse(BaseModel):
    entity_id: UUID
    entity_name: str
    entity_type: str
    claims: list[AnnotatedClaim] = Field(default_factory=list)


@router.get("/entities/{entity_id}/claims", response_model=EntityClaimsResponse)
async def get_entity_claims(
    entity_id: UUID, graph_store: GraphStoreDep, tenant_id: str = TenantQuery
) -> EntityClaimsResponse:
    """Full property-claim history for one entity, oldest first, each flagged
    as superseded or not."""
    entity = await graph_store.get_entity(entity_id, tenant_id)
    if entity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found.")
    superseded = {
        UUID(claim_id)
        for claim in entity.claims
        for claim_id in (claim.provenance.supersedes or [])
    }
    ordered = sorted(entity.claims, key=lambda claim: claim.provenance.created_at)
    return EntityClaimsResponse(
        entity_id=entity.id,
        entity_name=entity.name,
        entity_type=entity.entity_type,
        claims=[AnnotatedClaim(claim=c, superseded=c.id in superseded) for c in ordered],
    )


class ConflictingClaim(BaseModel):
    claim_id: UUID
    object_id: UUID
    object_name: str
    asserted_by: list[str]
    confidence: float
    asserted_at: datetime
    status: str


class Conflict(BaseModel):
    """A subject + predicate with two or more live, mutually disputed claims."""

    subject_id: UUID
    subject_name: str
    predicate: str
    claims: list[ConflictingClaim]


@router.get("/conflicts", response_model=list[Conflict])
async def list_conflicts(graph_store: GraphStoreDep, tenant_id: str = TenantQuery) -> list[Conflict]:
    """Live disputes, grouped by subject + predicate, with each competing
    claim's asserters. Superseded/retracted history is excluded."""
    scope = Scope.from_dict({"tenant_id": tenant_id})
    disputed = await graph_store.find_disputed_claims(scope)
    if not disputed:
        return []
    names = {entity.id: entity.name for entity in await graph_store.find_entities(scope)}

    conflicts: dict[tuple[UUID, str], Conflict] = {}
    for claim in disputed:
        key = (claim.subject_id, claim.predicate)
        if key not in conflicts:
            conflicts[key] = Conflict(
                subject_id=claim.subject_id,
                subject_name=names.get(claim.subject_id, str(claim.subject_id)),
                predicate=claim.predicate,
                claims=[],
            )
        conflicts[key].claims.append(
            ConflictingClaim(
                claim_id=claim.id,
                object_id=claim.object_id,
                object_name=names.get(claim.object_id, str(claim.object_id)),
                asserted_by=claim.asserted_by,
                confidence=claim.confidence,
                asserted_at=claim.valid_from,
                status=claim.status,
            )
        )
    return list(conflicts.values())


@router.delete("/entities/{entity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entity(
    entity_id: UUID, graph_store: GraphStoreDep, tenant_id: str = TenantQuery
) -> None:
    """Hard-delete one entity and its incident claims."""
    if not await graph_store.delete_entity(entity_id, tenant_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found.")
