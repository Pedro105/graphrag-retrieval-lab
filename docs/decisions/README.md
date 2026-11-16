# Architectural Decision Records

- [1. Use FalkorDB's native vector index instead of Postgres + pgvector](0001-falkordb-native-vector-index.md)
- [2. One FalkorDB graph per tenant, not a property filter](0002-graph-per-tenant-isolation.md)
- [3. Cache every LLM call on disk, keyed by its inputs](0003-cache-llm-calls-on-disk.md)
