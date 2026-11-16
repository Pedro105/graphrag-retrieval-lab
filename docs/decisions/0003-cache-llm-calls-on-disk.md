# 3. Cache every LLM call on disk, keyed by its inputs

## Status

Accepted.

## Context

Extraction, query classification, claim adjudication and synthesis all call
an LLM. Experiments re-run the same pipeline over the same corpus many times
(new retriever, new metric, a bug fix downstream of extraction). Without a
cache every rerun costs money, takes minutes to hours, and returns slightly
different graphs, which makes comparisons between runs noisy and forces
evaluation cases to chase the LLM's latest naming choices.

## Decision

Route every LLM call through one client (`llm/client.py`) that stores the
validated response as JSON under `LLM_CACHE_DIR`, keyed by the SHA-256 of
model, temperature, max_tokens, system prompt, user message and the response
model's JSON schema. Calls run at temperature 0 by default. The API client
is created lazily, so a fully cached run needs no API key.

## Consequences

- Reruns are free and produce identical graphs; extracted corpora can be
  shared as a cache directory so others reproduce results without keys.
- Any change to a prompt or schema invalidates exactly the affected entries,
  never stale results.
- To measure LLM nondeterminism itself, disable the cache
  (`LLM_CACHE_ENABLED=false`) or point it at a fresh directory.
- The cache grows without bound; it is plain files and safe to delete.
