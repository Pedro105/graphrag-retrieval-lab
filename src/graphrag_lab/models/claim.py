"""Claim: one source's statement about one property of an entity or relation.

Every claim is kept, so a later writer never silently replaces an earlier
one. `apply_claim` adds a write's claims to the history and
`derive_active_view` reduces a history to the current values.
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from graphrag_lab.models.provenance import Provenance


class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    property_name: str | None = None
    value: Any | None = None
    provenance: Provenance


def _active_claim(claims: list[Claim], property_name: str) -> Claim | None:
    """Most recent non-superseded claim for `property_name`, or None."""
    superseded_ids = {
        UUID(claim_id) for claim in claims for claim_id in (claim.provenance.supersedes or [])
    }
    candidates = [
        claim
        for claim in claims
        if claim.property_name == property_name and claim.id not in superseded_ids
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda claim: claim.provenance.created_at)


def apply_claim(
    claims: list[Claim],
    provenance: Provenance,
    properties: dict[str, Any],
) -> list[Claim]:
    """Add one write's claims to a history: one per property, or a single
    property-less claim if the write stated none, so the source is still
    recorded.

    A claim whose value differs from the current one supersedes it, whoever
    made either. The older claim stays in the list.
    """
    updated = list(claims)

    if not properties:
        updated.append(Claim(provenance=provenance))
        return updated

    for property_name, value in properties.items():
        active = _active_claim(updated, property_name)
        claim_provenance = provenance
        if active is not None and active.value != value:
            claim_provenance = provenance.model_copy(
                update={"supersedes": [*(provenance.supersedes or []), str(active.id)]}
            )
        updated.append(Claim(property_name=property_name, value=value, provenance=claim_provenance))

    return updated


def derive_active_view(claims: list[Claim]) -> tuple[dict[str, Any], Provenance, list[str]]:
    """Reduce a history to (properties, provenance, contributing_sources).

    The newest non-superseded claim wins for each property, the provenance is
    that of the newest claim overall, and the sources are everyone who ever
    made a claim.
    """
    property_names = {claim.property_name for claim in claims if claim.property_name is not None}
    properties: dict[str, Any] = {}
    for property_name in property_names:
        active = _active_claim(claims, property_name)
        if active is not None:
            properties[property_name] = active.value

    contributing_sources = sorted({claim.provenance.source for claim in claims})

    if not claims:
        return properties, Provenance(source="unknown"), contributing_sources

    latest = max(claims, key=lambda claim: claim.provenance.created_at)
    return properties, latest.provenance, contributing_sources
