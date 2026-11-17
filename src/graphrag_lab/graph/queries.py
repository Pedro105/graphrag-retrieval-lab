"""All Cypher.

`__T__` stands for the tenant label and is filled in by
`neo4j_store.tenant_query`, because labels and index names cannot be query
parameters. Everything else is parameterized.
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

# Also used for status changes, hence MERGE on the edges.
UPSERT_CLAIM = """
MERGE (c:Claim {id: $id})
SET c:__T__, c += $props
WITH c
MATCH (s:Entity:__T__ {id: $subject_id})
MATCH (o:Entity:__T__ {id: $object_id})
MERGE (c)-[:SUBJECT]->(s)
MERGE (c)-[:OBJECT]->(o)
"""

# --- claim reads -------------------------------------------------------------

# Disputed claims are left out: they are not adjudicated again.
FIND_ACTIVE_CLAIMS = """
MATCH (c:Claim:__T__)-[:SUBJECT]->(:Entity {id: $subject_id})
WHERE c.status = 'active' AND c.predicate = $predicate
RETURN c
"""

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

# `__VINDEX__` is the tenant's vector index name, substituted like `__T__`.
VECTOR_SEARCH = """
CYPHER 25
MATCH (node:__T__)
SEARCH node IN (VECTOR INDEX __VINDEX__ FOR $embedding LIMIT $k) SCORE AS score
RETURN node, score
ORDER BY score DESC
"""

FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes($index, $q, {limit: $limit}) YIELD node, score
RETURN node, score
ORDER BY score DESC
"""

# One hop: from each frontier entity, through a live claim, to the entity on
# its other end. Visited neighbours are returned too (see traversal.py).
FETCH_CLAIM_NEIGHBOURS = """
MATCH (e:Entity:__T__)<-[:SUBJECT|OBJECT]-(c:Claim:__T__)-[:SUBJECT|OBJECT]->(n:Entity:__T__)
WHERE e.id IN $frontier_ids AND c.status IN ['active', 'disputed'] AND n.id <> e.id
RETURN DISTINCT n, c
"""

# --- deletes and stats -------------------------------------------------------

# A claim cannot outlive either of its entities.
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
