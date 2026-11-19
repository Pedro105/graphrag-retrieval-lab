"""Cypher for the reified-claim model on Neo4j.

Every node carries its type label (`:Entity` / `:Claim`) plus a per-tenant
label (`:Tenant_<id>`, see ADR 0004). Queries are written against the
placeholder `__T__`, which `neo4j_store.tenant_query` replaces with the
validated tenant label -- labels and index names cannot be Cypher
parameters. Everything else is parameterized.

Facts are reified: the predicate is a `:Claim` property, and the only
relationship types are the fixed `:SUBJECT` / `:OBJECT`.
"""

# --- schema (global, created once) -------------------------------------------

CONSTRAINTS = (
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
)

# --- writes ------------------------------------------------------------------

UPSERT_ENTITY = """
MERGE (n:Entity {id: $id})
SET n:__T__, n += $props
"""

SET_ENTITY_EMBEDDING = """
MATCH (n:Entity:__T__ {id: $id})
CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
"""

# Create-or-update a claim by id (status transitions reuse this) and ensure its
# structural edges exist. MERGE on the edges keeps re-upserts idempotent.
UPSERT_CLAIM = """
MERGE (c:Claim {id: $id})
SET c:__T__, c += $props
WITH c
MATCH (s:Entity:__T__ {id: $subject_id})
MATCH (o:Entity:__T__ {id: $object_id})
MERGE (c)-[:SUBJECT]->(s)
MERGE (c)-[:OBJECT]->(o)
"""

# Batched variants for bulk loading curated graphs.
BULK_UPSERT_ENTITIES = """
UNWIND $rows AS row
MERGE (n:Entity {id: row.id})
SET n:__T__, n += row.props
WITH n, row
CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
"""

BULK_UPSERT_CLAIMS = """
UNWIND $rows AS row
MERGE (c:Claim {id: row.id})
SET c:__T__, c += row.props
WITH c, row
MATCH (s:Entity:__T__ {id: row.subject_id})
MATCH (o:Entity:__T__ {id: row.object_id})
MERGE (c)-[:SUBJECT]->(s)
MERGE (c)-[:OBJECT]->(o)
"""

# --- claim reads -------------------------------------------------------------

# Conflict/corroboration candidates: same subject and normalized predicate,
# currently active. Disputed claims are not re-adjudicated.
FIND_ACTIVE_CLAIMS = """
MATCH (c:Claim:__T__)-[:SUBJECT]->(:Entity {id: $subject_id})
WHERE c.status = 'active' AND c.predicate = $predicate
RETURN c
"""

# Every live (active or disputed) claim: the single source of display edges.
PROJECT_LIVE_CLAIMS = """
MATCH (c:Claim:__T__)
WHERE c.status IN ['active', 'disputed']
RETURN c
"""

DISTINCT_PREDICATES = """
MATCH (c:Claim:__T__)
WHERE c.status IN ['active', 'disputed']
RETURN DISTINCT c.raw_predicate AS predicate
"""

DISPUTED_CLAIMS = "MATCH (c:Claim:__T__) WHERE c.status = 'disputed' RETURN c"

# --- entity reads ------------------------------------------------------------

GET_ENTITY = "MATCH (n:Entity:__T__ {id: $id}) RETURN n LIMIT 1"

GET_ENTITIES = "MATCH (n:Entity:__T__) WHERE n.id IN $ids RETURN n"

FIND_ENTITIES = """
MATCH (n:Entity:__T__)
WHERE ($name IS NULL OR n.name = $name)
  AND ($entity_type IS NULL OR n.entity_type = $entity_type)
RETURN n
"""

# Cypher 25 SEARCH over the tenant's vector index (index names, like labels,
# cannot be parameters: `__VINDEX__` is substituted after validation). Neo4j
# reports cosine as (1 + cos) / 2 in [0, 1]; the store converts back.
VECTOR_SEARCH = """
CYPHER 25
MATCH (node:__T__)
SEARCH node IN (VECTOR INDEX __VINDEX__ FOR $embedding LIMIT $k) SCORE AS score
RETURN node, score
ORDER BY score DESC
"""

FULLTEXT_SEARCH = """
CYPHER 25
MATCH (node:__T__)
SEARCH node IN (FULLTEXT INDEX __FINDEX__ FOR $q LIMIT $limit) SCORE AS score
RETURN node, score
ORDER BY score DESC
"""

# One BFS hop through live claims, a claim acting as an undirected logical edge
# between its two entities. Already-visited neighbours are returned on purpose
# so edges between known entities reach the result (see retrieval/traversal.py).
FETCH_CLAIM_NEIGHBOURS = """
MATCH (e:Entity:__T__)<-[:SUBJECT|OBJECT]-(c:Claim:__T__)-[:SUBJECT|OBJECT]->(n:Entity:__T__)
WHERE e.id IN $frontier_ids AND c.status IN ['active', 'disputed'] AND n.id <> e.id
RETURN DISTINCT n, c
"""

# --- deletes and stats -------------------------------------------------------

# Deleting an entity also deletes its incident claims, never leaving orphans.
DELETE_ENTITY_WITH_CLAIMS = """
MATCH (n:Entity:__T__ {id: $id})
OPTIONAL MATCH (c:Claim)-[:SUBJECT|OBJECT]->(n)
DETACH DELETE c, n
"""

DELETE_MEMORY_WITH_CLAIMS = """
MATCH (n:Entity:__T__ {memory_id: $memory_id})
OPTIONAL MATCH (c:Claim)-[:SUBJECT|OBJECT]->(n)
DETACH DELETE c, n
"""

DELETE_TENANT_BATCH = """
MATCH (n:__T__)
WITH n LIMIT $batch
DETACH DELETE n
RETURN count(*) AS deleted
"""

COUNT_ENTITIES = "MATCH (n:Entity:__T__) RETURN count(n) AS count"
COUNT_LIVE_CLAIMS = (
    "MATCH (c:Claim:__T__) WHERE c.status IN ['active', 'disputed'] RETURN count(c) AS count"
)
