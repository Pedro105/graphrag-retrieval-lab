"""Conversion between models and Neo4j node properties.

Neo4j properties can be scalars, dates and flat lists, but not maps. Lists
and timestamps are stored natively so Cypher can query them; nested values
(free-form properties, claim history, scope, provenance) go in `*_json`
string properties.
"""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from graphrag_lab.models.claim import Claim
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import ClaimStatus, FactClaim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope


def to_datetime(value: Any) -> datetime:
    """Neo4j returns its own DateTime type."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"cannot convert {value!r} to datetime")


def _claims_to_json(claims: list[Claim]) -> str:
    return json.dumps([claim.model_dump(mode="json") for claim in claims])


def _claims_from_json(raw: str | None) -> list[Claim]:
    return [Claim.model_validate(item) for item in json.loads(raw or "[]")]


def entity_to_node_props(entity: Entity, memory_id: str) -> dict[str, Any]:
    """Everything but `id` (the MERGE key) and `embedding` (set separately)."""
    return {
        "name": entity.name,
        "entity_type": entity.entity_type,
        "tenant_id": entity.scope.tenant_id,
        "observed_types": entity.observed_types or [entity.entity_type],
        "merge_candidates": [str(eid) for eid in entity.merge_candidates],
        "contributing_sources": entity.contributing_sources,
        "properties_json": json.dumps(entity.properties),
        "scope_json": json.dumps(entity.scope.to_query_dict()),
        "provenance_json": entity.provenance.model_dump_json(),
        "claims_json": _claims_to_json(entity.claims),
        "memory_id": memory_id,
    }


def node_to_entity(props: dict[str, Any]) -> Entity:
    return Entity(
        id=UUID(props["id"]),
        name=props["name"],
        entity_type=props["entity_type"],
        observed_types=list(props.get("observed_types") or [props["entity_type"]]),
        properties=json.loads(props.get("properties_json") or "{}"),
        scope=Scope.from_dict(json.loads(props["scope_json"])),
        provenance=Provenance.model_validate_json(props["provenance_json"]),
        embedding=list(props["embedding"]) if props.get("embedding") is not None else None,
        merge_candidates=[UUID(eid) for eid in props.get("merge_candidates") or []],
        claims=_claims_from_json(props.get("claims_json")),
        contributing_sources=list(props.get("contributing_sources") or []),
    )


def claim_to_node_props(claim: FactClaim, memory_id: str) -> dict[str, Any]:
    """Everything but `id`. A null `valid_to` removes the property."""
    return {
        "predicate": claim.predicate,
        "raw_predicate": claim.raw_predicate,
        "subject_id": str(claim.subject_id),
        "object_id": str(claim.object_id),
        "tenant_id": claim.scope.tenant_id,
        "status": claim.status,
        "asserted_by": claim.asserted_by,
        "support_count": claim.support_count,
        "confidence": claim.confidence,
        "valid_from": claim.valid_from,
        "valid_to": claim.valid_to,
        "supersedes": [str(cid) for cid in claim.supersedes],
        "disputed_with": [str(cid) for cid in claim.disputed_with],
        "created_at": claim.created_at,
        "last_seen": claim.last_seen,
        "properties_json": json.dumps(claim.properties),
        "claims_json": _claims_to_json(claim.claims),
        "scope_json": json.dumps(claim.scope.to_query_dict()),
        "memory_id": memory_id,
    }


def node_to_claim(props: dict[str, Any]) -> FactClaim:
    status: ClaimStatus = props["status"]
    valid_to = props.get("valid_to")
    return FactClaim(
        id=UUID(props["id"]),
        predicate=props["predicate"],
        raw_predicate=props["raw_predicate"],
        subject_id=UUID(props["subject_id"]),
        object_id=UUID(props["object_id"]),
        status=status,
        asserted_by=list(props.get("asserted_by") or []),
        support_count=int(props.get("support_count", 1)),
        confidence=float(props.get("confidence", 1.0)),
        valid_from=to_datetime(props["valid_from"]),
        valid_to=to_datetime(valid_to) if valid_to is not None else None,
        supersedes=[UUID(cid) for cid in props.get("supersedes") or []],
        disputed_with=[UUID(cid) for cid in props.get("disputed_with") or []],
        created_at=to_datetime(props["created_at"]),
        last_seen=to_datetime(props["last_seen"]),
        properties=json.loads(props.get("properties_json") or "{}"),
        claims=_claims_from_json(props.get("claims_json")),
        scope=Scope.from_dict(json.loads(props["scope_json"])),
    )


def claim_to_relation(claim: FactClaim) -> Relation:
    """A claim as a plain subject -> object edge, for callers that want a graph."""
    asserter = claim.asserted_by[0] if claim.asserted_by else "unknown"
    return Relation(
        id=claim.id,
        source_entity_id=claim.subject_id,
        target_entity_id=claim.object_id,
        relation_type=claim.raw_predicate,
        properties=claim.properties,
        scope=claim.scope,
        provenance=Provenance(
            source=asserter, created_at=claim.valid_from, confidence=claim.confidence
        ),
        claims=claim.claims,
        contributing_sources=claim.asserted_by,
        status=claim.status,
        disputed_with=claim.disputed_with,
    )
