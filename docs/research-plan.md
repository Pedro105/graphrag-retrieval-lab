# Research plan

**Working title:** *From curated to extracted: do learned graph retrievers
survive noisy, LLM-built knowledge graphs?*

## Motivation

Learned graph retrievers (GNN-RAG, SubgraphRAG, G-Retriever, RoG) are
mostly trained and evaluated on curated knowledge graphs such as Freebase.
Practical GraphRAG systems build their graph from text with an LLM, which
produces duplicate entities, inconsistent relation names and missing edges.
How much of the reported benefit survives that noise, and whether graph
retrieval beats plain dense retrieval at equal context budgets, is under
studied. This repository's extraction → resolution → claims → retrieval
pipeline is the system under study.

## Research questions

- **RQ1: graph vs dense.** At matched context budgets (500 / 1k / 2k
  tokens), how much does graph retrieval improve multi-hop QA over dense
  passage retrieval, by hop count?
- **RQ2: learned vs heuristic.** Does a question-conditioned GNN subgraph
  scorer beat BFS expansion, personalized PageRank and vector seeding?
- **RQ3: robustness.** Starting from a gold graph, inject controlled noise
  (split X% of entities into duplicates, rename X% of predicates, drop X% of
  edges). How quickly does each retriever degrade?
- **RQ4 (stretch): evolving knowledge.** When sources disagree or facts
  change, does claim-level adjudication beat append-only storage and
  last-write-wins on current-state questions?

Each question is informative whatever the outcome: a learned retriever that
fails to beat PageRank on extracted graphs is a finding, not a failure.

## Data

- **2WikiMultiHopQA**: primary. Ships evidence triples per question, giving
  a gold graph and training labels alongside the graph we extract from the
  same passages, which separates extraction error from retrieval error.
  (Confirm the field layout when writing the loader.)
- **MuSiQue**: 2–4 hop questions built to resist shortcuts; tests
  generalization on extracted graphs only.
- **RoG-WebQSP / CWQ** (optional): sanity check of the GNN against
  published numbers on a curated graph.
- **Synthetic multi-writer benchmark** (RQ4): a seeded generator of a world
  state over time and documents emitted by simulated sources with
  configurable update, contradiction and noise rates. Ground truth is known
  by construction.

Extraction cost is piloted on ~100 passages before scaling to a ~1k
question subset. All LLM calls are cached (ADR 0003) and extracted graphs
are published so results reproduce without API keys.

## Systems compared

| Id | System |
|---|---|
| B0 | No retrieval (closed-book LLM) |
| B1 | Dense passage retrieval |
| B2 | BM25 + dense hybrid passage retrieval |
| B3 | Current pipeline: hybrid seeds + BFS (depth 1 / 2) |
| B4 | Personalized PageRank from seeds (HippoRAG-style) |
| B5 | Off-the-shelf `neo4j-graphrag` retriever |
| M1 | MLP triple scorer (SubgraphRAG-style, no message passing) |
| M2 | Question-conditioned GNN triple scorer |
| UB | Gold supporting facts (upper bound) |

All retrievers share one interface, `retrieve(question, token_budget)`, and
one answer generator, so only retrieval varies. The graph retrievers also
share the seeding step and a fixed snapshot of the graph, and a context is
always a prefix of the retriever's ranking cut at the budget, so a retriever
cannot gain by happening to rank short facts first. Budgets are counted in
words and punctuation marks rather than with a model tokenizer; the count
only needs to be the same for every system.

## The learned retriever (M1/M2)

- **Graph:** the candidate subgraph around entities linked from the question.
- **Features:** sentence-transformer embeddings of entity names (nodes) and
  predicate text (edges), plus distance to the question's topic entities.
- **Model:** PyTorch Geometric, edge-attribute-aware message passing
  (GINEConv or TransformerConv). R-GCN is unsuitable because LLM-extracted
  predicates are open vocabulary.
- **Output:** a relevance score per triple; the top triples that fit the
  token budget become the context.
- **Labels:** triples on shortest paths from question entities to answer
  entities (gold graph), used as weak supervision on extracted graphs.
- **Ablation:** M1 vs M2 isolates the value of message passing.

## Metrics and rigor

- Retrieval: supporting-fact recall at budget, answer-entity coverage,
  context tokens.
- QA: exact match and token F1.
- Cost and latency per question.
- 95% bootstrap confidence intervals, paired significance tests, 3+ seeds
  for trained models; per-question results stored as JSONL so every table
  and figure regenerates from raw outputs.
- Headline figure: answer F1 vs context tokens, one line per retriever,
  split by hop count.

## Milestones

Each milestone leaves the repository in a demonstrable state.

1. **Cleanup** (done): remove the SaaS layer, rename, compose stack, CI,
   cached LLM client, docs.
2. **Neo4j port and benchmark harness** (in progress): done are local
   embeddings, the Neo4j `GraphStore`, the retriever interface with BFS and
   personalized PageRank behind it, EM / F1 / coverage metrics, bootstrap
   confidence intervals and a paired permutation test. Remaining are the
   2WikiMultiHopQA loader and ingestion, the passage baselines, the runner
   that writes per-question JSONL, and the first results table (RQ1).
3. **Learned retriever**: M1 and M2, training pipeline, experiment
   tracking, model card (RQ2).
4. **Robustness study**: corruption operators and degradation curves (RQ3);
   MuSiQue for generalization.
5. **Evolving knowledge** (stretch): synthetic benchmark and adjudication
   comparison (RQ4).
6. **Report and demo**: static results page generated from `results/`, and
   an interactive demo comparing dense context with the learned subgraph.

## Reading list

- Edge et al., 2024. *From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization* (Microsoft GraphRAG).
- Gutiérrez et al., 2024. *HippoRAG*; and HippoRAG 2 (2025).
- He et al., 2024. *G-Retriever*.
- Mavromatis & Karypis, 2024. *GNN-RAG*.
- Li et al., 2025. *Simple is Effective: The Roles of Graphs and LLMs in
  Knowledge-Graph-Based RAG* (SubgraphRAG).
- Luo et al., 2025. *GFM-RAG*.
- Luo et al., 2024. *Reasoning on Graphs* (RoG).
- Rasmussen et al., 2025. *Zep: A Temporal Knowledge Graph Architecture for
  Agent Memory* (Graphiti).
- Trivedi et al., 2022. *MuSiQue*; Ho et al., 2020. *2WikiMultiHopQA*.
