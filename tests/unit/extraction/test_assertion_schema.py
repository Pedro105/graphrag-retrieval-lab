"""The assertion-semantics fields on ExtractedRelation: schema defaults,
backward compatibility, and that they stay off the display Relation model
(they feed claim adjudication instead). No API key needed.
"""

from uuid import uuid4

from graphrag_lab.extraction.schemas import ExtractedRelation
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope

SCOPE = Scope.from_dict({"tenant_id": "acme"})


def test_assertion_type_defaults_to_asserted():
    rel = ExtractedRelation(source_name="Pedro", target_name="ASML", relation_type="works_at")
    assert rel.assertion_type == "asserted"
    assert rel.as_of is None
    assert rel.replaces_hint is False


def test_terminated_and_negated_are_representable():
    terminated = ExtractedRelation(
        source_name="Pedro",
        target_name="ASML",
        relation_type="works_at",
        assertion_type="terminated",
        as_of="2024",
    )
    negated = ExtractedRelation(
        source_name="Pedro",
        target_name="ASML",
        relation_type="works_at",
        assertion_type="negated",
    )
    assert terminated.assertion_type == "terminated"
    assert terminated.as_of == "2024"
    negated_assertion: str = negated.assertion_type
    assert negated_assertion == "negated"


def test_replaces_hint_carries_through():
    rel = ExtractedRelation(
        source_name="Pedro",
        target_name="ABN AMRO",
        relation_type="works_at",
        assertion_type="asserted",
        as_of="now",
        replaces_hint=True,
    )
    assert rel.replaces_hint is True
    assert rel.as_of == "now"


def test_new_fields_do_not_leak_into_the_relation_model():
    """Assertion semantics live on claims (via ClaimAssertion), not on the
    display Relation model built in core/service.remember."""
    extracted = ExtractedRelation(
        source_name="Pedro",
        target_name="ASML",
        relation_type="works_at",
        assertion_type="terminated",
    )
    # Build a Relation exactly as core/service.remember does today.
    relation = Relation(
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relation_type=extracted.relation_type,
        properties=extracted.properties,
        scope=SCOPE,
        provenance=Provenance(source="agent_1"),
    )
    assert not hasattr(relation, "assertion_type")
