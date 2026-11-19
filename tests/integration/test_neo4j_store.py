"""Integration tests for Neo4jGraphStore against a real Neo4j instance.

Each test gets a fresh, randomly named tenant (its own label and indexes),
dropped in teardown, so tests don't interfere or leave state behind.
"""

import uuid

import pytest

from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.memory import Memory
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope


def make_claim(scope, subject_id, object_id, **overrides) -> FactClaim:
    """A minimal active FactClaim for store-level tests (the adjudicator builds
    these in the real write path; here we exercise persistence directly)."""
    defaults = dict(
        predicate="works_at",
        raw_predicate="works_at",
        subject_id=subject_id,
        object_id=object_id,
        asserted_by=["integration_test"],
        scope=scope,
    )
    defaults.update(overrides)
    return FactClaim(**defaults)


def make_scope(tenant_id: str, **kv: str) -> Scope:
    return Scope.from_dict({"tenant_id": tenant_id, **kv})


def make_provenance(**overrides) -> Provenance:
    defaults = dict(source="integration_test")
    defaults.update(overrides)
    return Provenance(**defaults)


async def test_health_check_true_when_reachable(store):
    assert await store.health_check() is True


async def test_write_memory_then_get_entity(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    memory = Memory(content="Pedro exists.", entities=[pedro], scope=scope, provenance=provenance)

    memory_id = await store.write_memory(memory)
    assert memory_id == str(memory.id)

    fetched = await store.get_entity(pedro.id, tenant_id)
    assert fetched is not None
    assert fetched.id == pedro.id
    assert fetched.name == "Pedro"
    assert fetched.entity_type == "person"
    assert fetched.scope == scope


async def test_get_entity_returns_none_when_missing(store, tenant_id):
    assert await store.get_entity(uuid.uuid4(), tenant_id) is None


async def test_write_memory_persists_entities_only(store, tenant_id):
    # In the reified-claim model, write_memory persists entities; facts are
    # written separately as :Claim nodes via upsert_claim.
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    acme = Entity(name="Acme", entity_type="org", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(
            content="Pedro works at Acme.",
            entities=[pedro, acme],
            scope=scope,
            provenance=provenance,
        )
    )

    assert (await store.get_entity(pedro.id, tenant_id)) is not None
    # No claims written yet -> no display edges.
    assert await store.project_display_edges(scope) == []


async def test_upsert_claim_round_trips_and_projects_as_directed_edge(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    acme = Entity(name="Acme", entity_type="org", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(content="seed", entities=[pedro, acme], scope=scope, provenance=provenance)
    )

    claim = make_claim(scope, pedro.id, acme.id, raw_predicate="works_at", predicate="works_at")
    await store.upsert_claim(claim, memory_id="m1")

    # Candidate lookup finds the active claim.
    candidates = await store.find_active_claims(scope, pedro.id, "works_at")
    assert [c.id for c in candidates] == [claim.id]

    # Projection collapses it to a directed display edge.
    edges = await store.project_display_edges(scope)
    assert len(edges) == 1
    assert edges[0].source_entity_id == pedro.id
    assert edges[0].target_entity_id == acme.id
    assert edges[0].relation_type == "works_at"
    assert edges[0].status == "active"


async def test_superseded_claim_drops_out_of_active_and_projection(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    acme = Entity(name="Acme", entity_type="org", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(content="seed", entities=[pedro, acme], scope=scope, provenance=provenance)
    )
    claim = make_claim(scope, pedro.id, acme.id)
    await store.upsert_claim(claim, memory_id="m1")

    # Transition the same claim id to superseded (upsert is the transition path).
    await store.upsert_claim(claim.model_copy(update={"status": "superseded"}), memory_id="m2")

    assert await store.find_active_claims(scope, pedro.id, "works_at") == []
    assert await store.project_display_edges(scope) == []


async def test_find_entities_by_name_and_type(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    acme = Entity(name="Acme", entity_type="org", scope=scope, provenance=provenance)
    memory = Memory(
        content="Pedro and Acme.", entities=[pedro, acme], scope=scope, provenance=provenance
    )
    await store.write_memory(memory)

    by_type = await store.find_entities(scope, entity_type="org")
    assert [e.id for e in by_type] == [acme.id]

    by_name = await store.find_entities(scope, name="Pedro")
    assert [e.id for e in by_name] == [pedro.id]


async def test_find_entities_respects_scope(store, tenant_id):
    scope_a = make_scope(tenant_id, user_id="u_1")
    scope_b = make_scope(tenant_id, user_id="u_2")
    provenance = make_provenance()
    entity_a = Entity(name="OnlyA", entity_type="person", scope=scope_a, provenance=provenance)
    entity_b = Entity(name="OnlyB", entity_type="person", scope=scope_b, provenance=provenance)
    memory = Memory(
        content="two scopes",
        entities=[entity_a, entity_b],
        scope=make_scope(tenant_id, workflow="shared"),
        provenance=provenance,
    )
    await store.write_memory(memory)

    found = await store.find_entities(scope_a, name="OnlyA")
    assert [e.id for e in found] == [entity_a.id]

    not_found = await store.find_entities(scope_a, name="OnlyB")
    assert not_found == []


async def test_delete_memory_removes_its_entities(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    memory = Memory(content="Pedro.", entities=[pedro], scope=scope, provenance=provenance)
    memory_id = await store.write_memory(memory)

    deleted = await store.delete_memory(memory_id, tenant_id)
    assert deleted is True

    assert await store.get_entity(pedro.id, tenant_id) is None


async def test_delete_memory_returns_false_when_nothing_to_delete(store, tenant_id):
    assert await store.delete_memory(str(uuid.uuid4()), tenant_id) is False


async def test_tenant_isolation_is_structural(store, tenant_id):
    other_tenant_id = f"test{uuid.uuid4().hex[:16]}"
    scope_a = make_scope(tenant_id, user_id="u_1")
    scope_b = make_scope(other_tenant_id, user_id="u_1")
    provenance = make_provenance()

    pedro_a = Entity(name="Pedro", entity_type="person", scope=scope_a, provenance=provenance)
    pedro_b = Entity(name="Pedro", entity_type="person", scope=scope_b, provenance=provenance)

    try:
        await store.write_memory(
            Memory(content="A", entities=[pedro_a], scope=scope_a, provenance=provenance)
        )
        await store.write_memory(
            Memory(content="B", entities=[pedro_b], scope=scope_b, provenance=provenance)
        )

        # Tenant A's graph store calls never see tenant B's entity, and vice versa.
        assert await store.get_entity(pedro_b.id, tenant_id) is None
        assert await store.get_entity(pedro_a.id, other_tenant_id) is None
        assert [e.id for e in await store.find_entities(scope_a)] == [pedro_a.id]
        assert [e.id for e in await store.find_entities(scope_b)] == [pedro_b.id]
    finally:
        await store.drop_graph(other_tenant_id)


async def test_reusing_an_entity_id_in_another_tenant_fails_loudly(store, tenant_id):
    """Ids are unique database-wide. Writing an existing id under another
    tenant must raise, never merge the two tenants onto one node."""
    from neo4j.exceptions import ConstraintError

    other_tenant_id = f"test{uuid.uuid4().hex[:16]}"
    scope_a, scope_b = make_scope(tenant_id), make_scope(other_tenant_id)
    provenance = make_provenance()
    entity_a = Entity(name="Pedro", entity_type="person", scope=scope_a, provenance=provenance)
    entity_b = entity_a.model_copy(update={"scope": scope_b})

    try:
        await store.write_memory(
            Memory(content="A", entities=[entity_a], scope=scope_a, provenance=provenance)
        )
        with pytest.raises(ConstraintError):
            await store.write_memory(
                Memory(content="B", entities=[entity_b], scope=scope_b, provenance=provenance)
            )
        assert await store.get_entity(entity_a.id, other_tenant_id) is None
        assert await store.get_entity(entity_a.id, tenant_id) is not None
    finally:
        await store.drop_graph(other_tenant_id)


async def test_find_similar_entities_returns_empty_for_uninitialized_tenant(store, tenant_id):
    scope = make_scope(tenant_id, user_id="u_1")
    results = await store.find_similar_entities(scope, [1.0, 0.0, 0.0, 0.0], None, 5)
    assert results == []


async def test_find_similar_entities_orders_by_similarity_descending(store, tenant_id):
    await store.ensure_graph_initialized(tenant_id, embedding_dimension=4)
    scope = make_scope(tenant_id, user_id="u_1")
    provenance = make_provenance()

    identical = Entity(
        name="Identical",
        entity_type="person",
        scope=scope,
        provenance=provenance,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    similar = Entity(
        name="Similar",
        entity_type="person",
        scope=scope,
        provenance=provenance,
        embedding=[0.9, 0.1, 0.0, 0.0],
    )
    orthogonal = Entity(
        name="Orthogonal",
        entity_type="person",
        scope=scope,
        provenance=provenance,
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    memory = Memory(
        content="three entities",
        entities=[identical, similar, orthogonal],
        scope=scope,
        provenance=provenance,
    )
    await store.write_memory(memory)

    results = await store.find_similar_entities(scope, [1.0, 0.0, 0.0, 0.0], None, 5)

    names_in_order = [entity.name for entity, _ in results]
    assert names_in_order == ["Identical", "Similar", "Orthogonal"]

    scores = dict((entity.name, score) for entity, score in results)
    assert scores["Identical"] == pytest.approx(1.0, abs=1e-4)
    assert scores["Orthogonal"] == pytest.approx(0.0, abs=1e-4)
    assert scores["Orthogonal"] < scores["Similar"] < scores["Identical"]


async def test_find_similar_entities_filters_by_entity_type_and_scope(store, tenant_id):
    await store.ensure_graph_initialized(tenant_id, embedding_dimension=4)
    scope_a = make_scope(tenant_id, user_id="u_1")
    scope_b = make_scope(tenant_id, user_id="u_2")
    provenance = make_provenance()

    person = Entity(
        name="Pedro",
        entity_type="person",
        scope=scope_a,
        provenance=provenance,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    org = Entity(
        name="Acme",
        entity_type="org",
        scope=scope_a,
        provenance=provenance,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    other_scope_person = Entity(
        name="OtherUser",
        entity_type="person",
        scope=scope_b,
        provenance=provenance,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    memory = Memory(
        content="mixed entities",
        entities=[person, org, other_scope_person],
        scope=scope_a,
        provenance=provenance,
    )
    await store.write_memory(memory)
    await store.write_memory(
        Memory(
            content="other user entity",
            entities=[],
            scope=scope_b,
            provenance=provenance,
        )
    )

    by_type = await store.find_similar_entities(scope_a, [1.0, 0.0, 0.0, 0.0], "person", 5)
    assert [e.id for e, _ in by_type] == [person.id]

    by_scope = await store.find_similar_entities(scope_a, [1.0, 0.0, 0.0, 0.0], None, 5)
    assert other_scope_person.id not in [e.id for e, _ in by_scope]


async def test_ensure_graph_initialized_is_idempotent(store, tenant_id):
    await store.ensure_graph_initialized(tenant_id, embedding_dimension=4)
    await store.ensure_graph_initialized(tenant_id, embedding_dimension=4)


# --- graph_stats / drop_graph -------------------------------------------------


async def test_find_disputed_claims_returns_disputed_excludes_resolved(store, tenant_id):
    scope = make_scope(tenant_id)
    provenance = make_provenance()
    pedro = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    asml = Entity(name="ASML", entity_type="org", scope=scope, provenance=provenance)
    booking = Entity(name="Booking", entity_type="org", scope=scope, provenance=provenance)
    superseded_obj = Entity(name="Old", entity_type="org", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(
            content="seed",
            entities=[pedro, asml, booking, superseded_obj],
            scope=scope,
            provenance=provenance,
        )
    )
    await store.upsert_claim(
        make_claim(scope, pedro.id, asml.id, status="disputed", asserted_by=["agent_a"]), "m1"
    )
    await store.upsert_claim(
        make_claim(scope, pedro.id, booking.id, status="disputed", asserted_by=["agent_b"]), "m1"
    )
    # A superseded (resolved) fact must NOT surface as a live conflict.
    await store.upsert_claim(
        make_claim(scope, pedro.id, superseded_obj.id, status="superseded"), "m1"
    )

    disputed = await store.find_disputed_claims(scope)
    objects = {c.object_id for c in disputed}
    assert objects == {asml.id, booking.id}


async def test_graph_stats_zero_for_unwritten_tenant(store, tenant_id):
    # A tenant whose graph was never written to reports (0, 0), not an error.
    assert await store.graph_stats(tenant_id) == (0, 0)


async def test_graph_stats_counts_entities_and_live_claims(store, tenant_id):
    scope = make_scope(tenant_id)
    provenance = make_provenance()
    a = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    b = Entity(name="ASML", entity_type="org", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(content="seed", entities=[a, b], scope=scope, provenance=provenance)
    )
    await store.upsert_claim(make_claim(scope, a.id, b.id), memory_id="m1")

    # 2 entities + 1 live claim.
    assert await store.graph_stats(tenant_id) == (2, 1)


async def test_drop_graph_wipes_tenant_and_is_idempotent(store, tenant_id):
    scope = make_scope(tenant_id)
    provenance = make_provenance()
    entity = Entity(name="Pedro", entity_type="person", scope=scope, provenance=provenance)
    await store.write_memory(
        Memory(content="Pedro exists.", entities=[entity], scope=scope, provenance=provenance)
    )
    assert await store.graph_stats(tenant_id) == (1, 0)

    await store.drop_graph(tenant_id)
    assert await store.graph_stats(tenant_id) == (0, 0)
    # Dropping an already-gone graph is a no-op, not an error.
    await store.drop_graph(tenant_id)
