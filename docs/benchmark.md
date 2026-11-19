# Benchmark protocol

How `python -m graphrag_lab.bench` measures retrieval, and what the numbers
do and do not show. Results live in [`results/`](../results/).

## Question

At an equal context budget, how often does each retriever put the facts
needed for a multi-hop question in front of the reader model? This is
retrieval quality only: no LLM reads the context, so a run needs no API key.

## Data

- **2WikiMultiHopQA** validation split, pinned to Hugging Face revision
  `fe713bf` of `framolfese/2WikiMultihopQA`. Each question has an answer,
  ten paragraphs (supporting ones plus distractors), sentence-level
  supporting facts, and its gold reasoning chain as evidence triples, e.g.
  `(Polish-Russian War, director, Xawery Żuławski)`,
  `(Xawery Żuławski, mother, Małgorzata Braunek)`.
- **Sample.** `n_corpus` questions are sampled with a fixed seed; the first
  `n_eval` of them are evaluated. All sampled questions' paragraphs and
  triples are pooled, so each question competes with every other question's
  distractors, not just its own ten paragraphs.
- **Background knowledge.** A graph made only of the sample's gold triples
  would be unrealistically clean, since every fact in it answers something.
  The corpus therefore also includes every *training* question that shares a
  specific entity with the sample. Hubs are excluded (entities in more than
  `hub_max` triples, such as "American") and so are dates. Their triples go
  into the graph and their supporting paragraphs into the text corpus, so
  both sides gain the same distractor knowledge. They are never evaluated.

## Conditions

This is the **gold-graph condition**: the graph holds curated Wikidata facts,
standing in for perfect LLM extraction, while text retrievers read raw
paragraphs. It therefore bounds what graph retrieval can gain *given perfect
extraction*. How fast that advantage erodes under extraction errors
(duplicate entities, inconsistent predicates, missing facts) is the planned
robustness study.

Comparisons *within* the graph (BFS vs PageRank vs dense triple search vs a
learned retriever) use the same graph and are fully like for like.

## Retrievers

| Retriever | Units | How it ranks |
|---|---|---|
| `dense_paragraph` | sentences | bge-small embedding of each paragraph, cosine to the question |
| `bm25_paragraph` | sentences | BM25 over paragraph text |
| `hybrid_paragraph` | sentences | reciprocal rank fusion of the two above |
| `dense_sentence` | sentences | bge-small embedding of each sentence |
| `dense_triple` | triples | bge-small embedding of each triple's text; no graph structure |
| `graph_bfs_d1`, `graph_bfs_d2` | triples | **the system as built**: `recall()` with hybrid seeding and breadth-first traversal in Neo4j; triples in discovery order, unranked |
| `graph_ppr` | triples | same seeds, then personalized PageRank over the Neo4j graph; a triple scores the sum of its endpoints' PageRank (HippoRAG-style) |
| `oracle_sentence`, `oracle_triple` | — | the gold supporting sentences / evidence triples: upper bounds and metric sanity checks |

Paragraph retrievers return their paragraphs' sentences in order. Graph
retrievers seed from the 5 best entities by fused vector + full-text search
over entity names. Queries to dense text indexes use bge's retrieval
instruction; entity seeding embeds the plain question, as the API does.

## Budgets and packing

Each ranking is packed into budgets of 64–1024 tokens (bge tokenizer). A
sentence costs its tokens plus its paragraph title the first time that title
appears; a triple costs its rendered text (`subject | predicate | object`).
Packing keeps the longest prefix of the ranking that fits.

## Metrics

All three are reported with percentile-bootstrap 95% confidence intervals
over questions (1,000 resamples).

- **Answer recall:** the normalized gold answer (SQuAD normalization, whole
  tokens) appears in the packed context. Only defined for bridge questions
  (compositional and inference), where the answer is reached through the
  chain. In comparison questions the answer is named in the question, and
  yes/no answers can't be matched. Checked: the gold triples contain the
  answer for 100% of bridge questions.
- **Full chain retrieved:** every gold unit is in the context. Gold units
  are evidence triples for graph retrievers and supporting sentences for
  text retrievers. Supporting sentences are matched by title and text, since
  about 2% of repeated titles in the dataset are split into sentences
  differently.
- **Gold units retrieved:** the fraction of gold units in the context.

The two support metrics count different units for text and graph retrievers,
so compare them within a modality. Answer recall is comparable across
modalities.

## Reproducing

```bash
make up                     # Neo4j
make bench-smoke            # 50 questions, a few minutes
make bench                  # configs/bench/2wiki-dev.yaml
uv run python -m graphrag_lab.bench report results/2wiki-dev   # rebuild tables
```

Embeddings are cached in `.cache/bench/embeddings/<corpus fingerprint>/`, and
the gold graph is reused when its Neo4j tenant (`bench_<fingerprint>`)
already holds it. Each run writes `manifest.json` with the config, corpus
statistics, git commit and package versions, and `per_question.jsonl.gz` with
one record per question, retriever and budget, so any table can be
recomputed or sliced (for example by question type) without rerunning.

## Limitations

- The gold graph is cleaner than any extracted graph, and sparser than full
  Wikidata (it only has the ~30 relation types 2WikiMultiHopQA uses).
- Entity seeding benefits from questions naming entities exactly as their
  Wikidata labels; the per-retriever seed hit rate is reported to make this
  visible.
- One embedding model (bge-small) for every dense retriever; stronger
  encoders would lift all dense baselines.
- Retrieval only. Whether a reader model turns better context into better
  answers needs an LLM and is a separate experiment.
