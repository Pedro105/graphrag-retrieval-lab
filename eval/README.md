# Recall smoke eval

A small regression check for `POST /v1/recall`. It is **not** the research
benchmark: 11 hand-written cases over a synthetic graph of ~40 sentences
cannot show whether graph retrieval helps. The benchmark harness with real
datasets, baselines and confidence intervals arrives in milestone 2 (see
[`docs/research-plan.md`](../docs/research-plan.md)); this stays as a fast
end-to-end smoke test.

## The graph

`scripts/populate_graph.py` ingests three mutually isolated clusters into
the `eval_test` graph: a biotech startup, an open-source ML project, and an
EV supply chain. Each cluster contains an engineered two-hop chain whose
endpoints never appear in the same sentence, plus a property that two
sources state differently (for provenance checks).

## Checks

Each case in `dataset.yaml` sends its `query` to `/v1/recall` and can check:

| Field | Check |
|---|---|
| `expected_entities` | every named entity is returned (required) |
| `expected_properties` | an entity's property has the expected value |
| `expected_relations` | an edge (source, target, type) exists with the expected property value |
| `forbidden_entities` | none of these appear (cluster isolation) |
| `expected_claim_history` | an entity's claim history includes the expected superseded values |
| `bridge_target` | `bridging` cases run at depth 1 (target must be absent) and depth 2 (present) |

Categories: `exact_term`, `single_hop`, `bridging`, `provenance`, `isolation`.

## Running

```bash
make up && make api          # FalkorDB + API, with API keys in .env
make populate                # ingest the synthetic graph (LLM calls are cached)
make eval                    # or: uv run python eval/run_eval.py --help
```

Results print as a per-category hit-rate table and a timestamped markdown
report in `eval/reports/` (gitignored).

## Caveats

- Matching is by entity name, which is stable for this graph but not in
  general.
- Extraction chooses property and relation names itself. With the LLM cache
  (ADR 0003) a populated graph is reproducible, but changing the extraction
  prompt can rename properties; if a case fails only on a property or
  relation *name*, inspect `GET /v1/graph?tenant_id=eval_test` before
  assuming a regression.
- Some cases pin `limit` because FalkorDB's HNSW index under-returns for
  small `k` on tiny graphs.
