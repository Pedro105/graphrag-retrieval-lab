"""Breadth-first traversal, kept free of any database code.

The store passes in two fetch functions (one batched Cypher query per hop)
and tests pass in-memory ones. Traversal is undirected; each relation still
records its real direction.
"""

from collections.abc import Awaitable, Callable
from uuid import UUID

import structlog

from graphrag_lab.models.entity import Entity
from graphrag_lab.models.relation import Relation

logger = structlog.get_logger()

SeedFetcher = Callable[[list[str]], Awaitable[list[Entity]]]

# Returns (neighbour, relation) for every edge touching the frontier, including
# edges to entities already visited. Leaving those out would lose the edges
# between two seeds, since seeds are visited before the first hop.
NeighbourFetcher = Callable[[list[str]], Awaitable[list[tuple[Entity, Relation]]]]


async def breadth_first_traverse(
    seed_ids: list[str],
    depth: int,
    max_entities: int,
    fetch_seeds: SeedFetcher,
    fetch_neighbours: NeighbourFetcher,
    *,
    tenant_id: str,
) -> tuple[list[Entity], list[Relation], bool]:
    """Expand `depth` hops out from the seeds.

    Returns (entities, relations, truncated). The seeds are included, at most
    `max_entities` entities come back, and only relations with both ends in
    the result are kept.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if max_entities < 1:
        raise ValueError("max_entities must be >= 1")

    entities_by_id: dict[str, Entity] = {}
    relations_by_id: dict[UUID, Relation] = {}
    truncated = False

    def _add_entity(entity: Entity) -> bool:
        """False once the cap is reached and the entity could not be added."""
        nonlocal truncated
        entity_id = str(entity.id)
        if entity_id in entities_by_id:
            return True
        if len(entities_by_id) >= max_entities:
            truncated = True
            return False
        entities_by_id[entity_id] = entity
        return True

    frontier: list[str] = []
    for seed in await fetch_seeds(seed_ids):
        if _add_entity(seed):
            frontier.append(str(seed.id))
        else:
            break

    hops_completed = 0
    while frontier and hops_completed < depth and not truncated:
        rows = await fetch_neighbours(frontier)
        next_frontier: list[str] = []
        for neighbour, relation in rows:
            relations_by_id[relation.id] = relation
            neighbour_id = str(neighbour.id)
            already_known = neighbour_id in entities_by_id
            if not _add_entity(neighbour):
                break
            if not already_known:
                next_frontier.append(neighbour_id)
        frontier = next_frontier
        hops_completed += 1

    if truncated:
        logger.warning(
            "traversal.truncated",
            tenant_id=tenant_id,
            depth_reached=hops_completed,
            requested_depth=depth,
            entity_count=len(entities_by_id),
            max_entities=max_entities,
        )

    present_ids = {UUID(entity_id) for entity_id in entities_by_id}
    relations = [
        relation
        for relation in relations_by_id.values()
        if relation.source_entity_id in present_ids and relation.target_entity_id in present_ids
    ]
    return list(entities_by_id.values()), relations, truncated
