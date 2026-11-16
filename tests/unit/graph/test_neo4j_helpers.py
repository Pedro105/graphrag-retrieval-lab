"""Pure helpers of the Neo4j store and node (de)serialization. No database."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from neo4j.time import DateTime as Neo4jDateTime

from graphrag_lab.graph.neo4j_store import (
    cosine_from_neo4j_score,
    sanitize_fulltext_query,
    tenant_label,
    tenant_query,
    vector_index_name,
)
from graphrag_lab.graph.schema import (
    claim_to_node_props,
    entity_to_node_props,
    node_to_claim,
    node_to_entity,
    to_datetime,
)
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope

SCOPE = Scope.from_dict({"tenant_id": "t1", "project": "alpha"})


def test_sanitize_strips_lucene_operators_and_or_joins_tokens() -> None:
    assert sanitize_fulltext_query("RG-88") == "RG OR 88"
    assert sanitize_fulltext_query("Korrigan Cells Ltd") == "Korrigan OR Cells OR Ltd"
    assert sanitize_fulltext_query("name:Recall R-2025-014") == "name OR Recall OR R OR 2025 OR 014"
    assert sanitize_fulltext_query("-- !!") == ""


@pytest.mark.parametrize(("score", "cosine"), [(1.0, 1.0), (0.5, 0.0), (0.0, -1.0), (0.96, 0.92)])
def test_neo4j_cosine_score_maps_back_to_plain_cosine(score: float, cosine: float) -> None:
    assert cosine_from_neo4j_score(score) == pytest.approx(cosine)


def test_tenant_label_quotes_valid_ids_and_rejects_unsafe_ones() -> None:
    assert tenant_label("eval_test") == "`Tenant_eval_test`"
    assert tenant_label("a-b") == "`Tenant_a-b`"
    for bad in ["", "a b", "x`) DETACH DELETE n //", "ü"]:
        with pytest.raises(ValueError):
            tenant_label(bad)
    with pytest.raises(ValueError):
        vector_index_name("bad name")


def test_tenant_query_substitutes_every_placeholder() -> None:
    query = tenant_query("MATCH (a:Entity:__T__)--(b:__T__) RETURN a", "t1")
    assert query == "MATCH (a:Entity:`Tenant_t1`)--(b:`Tenant_t1`) RETURN a"


def test_entity_round_trip_with_native_lists() -> None:
    entity = Entity(
        name="Helios Therapeutics",
        entity_type="Organization",
        observed_types=["Organization", "Company"],
        properties={"founded": 2019, "hq": {"city": "Boston"}},
        scope=SCOPE,
        provenance=Provenance(source="agent_a"),
        merge_candidates=[uuid4()],
        contributing_sources=["agent_a"],
    )
    props = entity_to_node_props(entity, "mem-1")
    assert props["observed_types"] == ["Organization", "Company"]
    assert props["tenant_id"] == "t1"
    assert "embedding" not in props  # written separately as a vector property

    restored = node_to_entity({"id": str(entity.id), **props, "embedding": [0.1, 0.2]})
    assert restored.model_dump(exclude={"embedding"}) == entity.model_dump(exclude={"embedding"})
    assert restored.embedding == [0.1, 0.2]


def test_claim_round_trip_with_native_datetimes() -> None:
    claim = FactClaim(
        predicate="works_at",
        raw_predicate="works_at",
        subject_id=uuid4(),
        object_id=uuid4(),
        status="superseded",
        asserted_by=["agent_a", "agent_b"],
        support_count=2,
        valid_to=datetime(2026, 1, 2, tzinfo=UTC),
        supersedes=[uuid4()],
        properties={"role": "engineer"},
        scope=SCOPE,
    )
    props = claim_to_node_props(claim, "mem-1")
    assert isinstance(props["valid_from"], datetime)
    assert props["asserted_by"] == ["agent_a", "agent_b"]

    # Simulate what the driver hands back: Neo4j DateTime objects.
    from_db = {
        "id": str(claim.id),
        **props,
        "valid_from": Neo4jDateTime.from_native(claim.valid_from),
        "valid_to": Neo4jDateTime.from_native(claim.valid_to),  # type: ignore[arg-type]
    }
    assert node_to_claim(from_db) == claim


def test_open_claim_has_null_valid_to() -> None:
    claim = FactClaim(
        predicate="p", raw_predicate="p", subject_id=uuid4(), object_id=uuid4(), scope=SCOPE
    )
    props = claim_to_node_props(claim, "m")
    assert props["valid_to"] is None
    restored = node_to_claim(
        {"id": str(claim.id), **{k: v for k, v in props.items() if v is not None}}
    )
    assert restored.valid_to is None


def test_to_datetime_accepts_native_neo4j_and_iso() -> None:
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
    assert to_datetime(now) == now
    assert to_datetime(Neo4jDateTime.from_native(now)) == now
    assert to_datetime(now.isoformat()) == now
    with pytest.raises(TypeError):
        to_datetime(42)
