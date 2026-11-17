"""Entity: a node in the knowledge graph (a person, thing, or concept)."""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graphrag_lab.models.claim import Claim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope


class Entity(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    # `entity_type` is the first type seen and never changes. Types other
    # writers used for the same entity are collected in `observed_types`.
    observed_types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    scope: Scope
    provenance: Provenance
    embedding: list[float] | None = None
    merge_candidates: list[UUID] = Field(default_factory=list)
    # Every source's claims about this entity's properties. `properties` and
    # `provenance` above are the current view derived from them.
    claims: list[Claim] = Field(default_factory=list)
    contributing_sources: list[str] = Field(default_factory=list)
