"""traverse_from_seeds applies the caller's scope to the connecting claim as
well as to the neighbour. The Neo4j driver is faked -- no database."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from graphrag_lab.graph import queries
from graphrag_lab.graph.neo4j_store import Neo4jGraphStore, tenant_query
from graphrag_lab.graph.schema import claim_to_node_props, entity_to_node_props
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import FactClaim
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope

TENANT = "acme"
ALPHA = Scope.from_dict({"tenant_id": TENANT, "project": "alpha"})
TENANT_WIDE = Scope.from_dict({"tenant_id": TENANT})


def _entity_node(entity: Entity) -> dict:
    return {"id": str(entity.id), **entity_to_node_props(entity, "m1")}


def _claim_node(claim: FactClaim) -> dict:
    return {"id": str(claim.id), **claim_to_node_props(claim, "m1")}


async def test_claim_outside_the_callers_scope_is_not_walked():
    provenance = Provenance(source="a")
    seed = Entity(name="Seed", entity_type="org", scope=ALPHA, provenance=provenance)
    near = Entity(name="Near", entity_type="org", scope=ALPHA, provenance=provenance)
    far = Entity(name="Far", entity_type="org", scope=ALPHA, provenance=provenance)
    in_scope = FactClaim(
        predicate="owns", raw_predicate="owns", subject_id=seed.id, object_id=near.id, scope=ALPHA
    )
    out_of_scope = FactClaim(
        predicate="owns",
        raw_predicate="owns",
        subject_id=seed.id,
        object_id=far.id,
        scope=TENANT_WIDE,
    )

    async def execute_query(query, parameters_, database_):
        if query == tenant_query(queries.GET_ENTITIES, TENANT):
            return SimpleNamespace(records=[{"n": _entity_node(seed)}])
        assert query == tenant_query(queries.FETCH_CLAIM_NEIGHBOURS, TENANT)
        return SimpleNamespace(
            records=[
                {"n": _entity_node(near), "c": _claim_node(in_scope)},
                {"n": _entity_node(far), "c": _claim_node(out_of_scope)},
            ]
        )

    driver = AsyncMock()
    driver.execute_query.side_effect = execute_query
    store = Neo4jGraphStore(driver=driver)

    entities, relations, truncated = await store.traverse_from_seeds(ALPHA, [str(seed.id)], 1)

    assert {entity.id for entity in entities} == {seed.id, near.id}
    assert [relation.id for relation in relations] == [in_scope.id]
    assert truncated is False
