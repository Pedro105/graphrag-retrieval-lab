"""Benchmark graph retrievers against a real Neo4j: load a toy gold graph,
then check BFS (the system's recall) and personalized PageRank find the
reasoning chain. A hashing encoder stands in for the embedding model."""

import hashlib
import re
from types import SimpleNamespace

import numpy as np

from graphrag_lab.bench.data import build_corpus
from graphrag_lab.bench.encoding import Encoder, Matrix
from graphrag_lab.bench.graph import GraphBFSRetriever, GraphPPRRetriever, load_gold_graph
from tests.unit.bench.helpers import FILM_ROW, make_row

DIM = 512


class HashingEncoder(Encoder):
    """Deterministic bag-of-words vectors (word overlap drives similarity):
    no model download."""

    def __init__(self) -> None:
        super().__init__("hashing", cache_dir=None)

    @property
    def model(self):  # type: ignore[override]
        return SimpleNamespace(get_embedding_dimension=lambda: DIM)

    def _encode(self, texts: list[str], show_progress: bool = False) -> Matrix:
        matrix = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                matrix[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)


DISTRACTOR = make_row(
    "q2",
    "When was Snow White released?",
    "1937",
    [["Snow White", "publication date", "1937"], ["Snow White", "director", "David Hand"]],
    {"Snow White": ["Snow White was released in 1937."]},
    [("Snow White", 0)],
)


async def test_bfs_and_ppr_retrieve_the_chain_from_neo4j(store, tenant_id):
    corpus = build_corpus([FILM_ROW, DISTRACTOR], n_eval=1)
    encoder = HashingEncoder()
    await load_gold_graph(store, corpus, encoder, tenant_id)
    assert await store.graph_stats(tenant_id) == (len(corpus.entities), len(corpus.triples))
    await load_gold_graph(store, corpus, encoder, tenant_id)  # reused, not duplicated
    assert await store.graph_stats(tenant_id) == (len(corpus.entities), len(corpus.triples))

    question = corpus.eval_questions[0]
    chain = set(question.evidence)

    bfs = await GraphBFSRetriever(store, tenant_id, encoder, depth=2, n_seeds=2).rank(question)
    assert chain <= {unit.key for unit in bfs.units}
    assert bfs.diagnostics["seed_hit"] is True

    # One seed: the question's own entity. The distractor component is
    # unreachable from it, so it scores zero and is not returned at all.
    ppr_retriever = await GraphPPRRetriever(store, tenant_id, encoder, n_seeds=1).build()
    ppr = await ppr_retriever.rank(question)
    assert {unit.key for unit in ppr.units} == chain
    assert ppr.diagnostics["seed_hit"] is True


async def test_same_corpus_in_two_tenants_shares_no_nodes(store, tenant_id):
    """Regression: deterministic ids used to be name-only, so a second tenant
    merged into the first one's nodes and dropping it deleted them."""
    other = f"{tenant_id}_b"
    corpus = build_corpus([FILM_ROW, DISTRACTOR], n_eval=1)
    encoder = HashingEncoder()
    expected = (len(corpus.entities), len(corpus.triples))
    try:
        await load_gold_graph(store, corpus, encoder, tenant_id)
        await load_gold_graph(store, corpus, encoder, other)
        await store.drop_graph(other)
        assert await store.graph_stats(tenant_id) == expected
    finally:
        await store.drop_graph(other)
