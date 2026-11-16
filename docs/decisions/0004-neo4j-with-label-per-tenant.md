# 4. Move to Neo4j, isolating tenants by label and per-tenant indexes

## Status

Accepted. Supersedes [1](0001-falkordb-native-vector-index.md) and
[2](0002-graph-per-tenant-isolation.md).

## Context

The project started on FalkorDB, chosen for its native vector index and one
graph per tenant. As the project became a GraphRAG research testbed, three
things changed the trade-off:

- The research plan needs graph algorithms (personalized PageRank, node
  embeddings) and comparison against the `neo4j-graphrag` reference
  retrievers; both live in the Neo4j ecosystem.
- Neo4j is the graph database most GraphRAG work and tooling targets, so
  results and code are easier for others to reproduce and reuse.
- FalkorDB changed procedure signatures between releases (the full-text
  index call broke silently), which is a reproducibility risk.

Neo4j Community Edition, however, has a single user database, so the
graph-per-tenant isolation from ADR 2 is not available. A shared vector
index with a tenant post-filter would reintroduce the problem ADR 2 solved:
other tenants' vectors can crowd a tenant out of the top-k.

## Decision

- Store everything in one Neo4j database. Every node owned by a tenant gets
  a `:Tenant_<id>` label in addition to `:Entity` or `:Claim`, and a
  `tenant_id` property.
- Give each tenant its own vector index (`entity_embedding_<id>`) and
  full-text index (`entity_name_<id>`) on that label. Vector search runs
  through the Cypher 25 `SEARCH` clause against the tenant's index, so it
  only ever ranks that tenant's entities.
- Validate `tenant_id` against `^[A-Za-z0-9_-]+$` before inlining it into
  labels and index names, which cannot be query parameters. Everything
  else is parameterized.
- Store lists (asserters, observed types, supersession links) and
  timestamps as native Neo4j values so they are queryable in Cypher; keep
  JSON only for nested maps, which Neo4j properties cannot hold.
- Convert Neo4j's cosine score, reported as `(1 + cos) / 2`, back to plain
  cosine so the entity-resolution thresholds keep their meaning (verified
  live: identical vectors 1.0, orthogonal 0.5, opposite 0.0).

## Consequences

- Isolation stays structural, with one index pair per tenant. That suits a
  handful of corpora and experiment runs, not thousands of tenants.
- Dropping a tenant deletes its nodes in batches and drops its indexes.
- A tenant's vector index fixes its dimension; switching embedding models
  requires dropping the tenant (the store raises a clear error otherwise).
- The FalkorDB implementation and its dependency are removed.
