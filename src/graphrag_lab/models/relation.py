"""Relation: a typed edge between two entities in the knowledge graph."""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graphrag_lab.models.claim import Claim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope


class Relation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID
    target_entity_id: UUID
    relation_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    scope: Scope
    provenance: Provenance
    # Same history as Entity.claims.
    claims: list[Claim] = Field(default_factory=list)
    contributing_sources: list[str] = Field(default_factory=list)
    # Set when this edge was projected from a stored claim: "disputed" means
    # sources disagree, and `disputed_with` lists the competing claim ids.
    status: str = "active"
    disputed_with: list[UUID] = Field(default_factory=list)
