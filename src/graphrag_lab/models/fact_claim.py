"""FactClaim: a fact stored as its own `:Claim` node between a subject and an
object entity.

A plain edge cannot say who asserted it, whether it is still true, or what
it replaced. A claim node can, which lets several sources assert, confirm or
contest the same fact, and lets a fact be superseded without being deleted.

Not to be confused with `models/claim.py::Claim`, which tracks the history of
one property. A FactClaim holds a list of those for its own properties.
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graphrag_lab.extraction.schemas import AssertionType
from graphrag_lab.models.claim import Claim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope

ClaimStatus = Literal["active", "superseded", "disputed", "retracted"]
# Statuses that count as current. Superseded and retracted claims are history.
LIVE_STATUSES: tuple[ClaimStatus, ...] = ("active", "disputed")


def _now() -> datetime:
    return datetime.now(UTC)


class FactClaim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    # `predicate` is normalized and used for matching; `raw_predicate` is the
    # relation type as extracted, used for display.
    predicate: str = Field(min_length=1)
    raw_predicate: str = Field(min_length=1)
    subject_id: UUID
    object_id: UUID
    status: ClaimStatus = "active"
    # Every source that has asserted this exact fact.
    asserted_by: list[str] = Field(default_factory=list)
    support_count: int = 1
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    valid_from: datetime = Field(default_factory=_now)
    # Set when the claim is superseded or retracted.
    valid_to: datetime | None = None
    # Ids of the claims this one replaced.
    supersedes: list[UUID] = Field(default_factory=list)
    # Ids of the claims this one conflicts with; the link is mutual.
    disputed_with: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    last_seen: datetime = Field(default_factory=_now)
    # Current properties and their history, e.g. a revised price.
    properties: dict[str, Any] = Field(default_factory=dict)
    claims: list[Claim] = Field(default_factory=list)
    scope: Scope


class ClaimAssertion(BaseModel):
    """An extracted relation with its entities resolved, ready to adjudicate.
    Never stored."""

    subject_id: UUID
    object_id: UUID
    subject_name: str = Field(min_length=1)
    object_name: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    raw_predicate: str = Field(min_length=1)
    assertion_type: AssertionType = "asserted"
    as_of: str | None = None
    replaces_hint: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    scope: Scope

    @property
    def asserter(self) -> str:
        return self.provenance.source

    @property
    def has_recency_signal(self) -> bool:
        """The text marked this as the new state ("now works at", "since March")."""
        return self.replaces_hint or self.as_of is not None
