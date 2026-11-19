"""Retriever interface, text baselines, and oracles.

Every retriever ranks units for a question; the runner packs that ranking
into each token budget. Text retrievers return sentences (grouped under
their paragraph title), graph retrievers (bench/graph.py) return triples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from graphrag_lab.bench.context import Unit, UnitKind
from graphrag_lab.bench.data import Corpus, Paragraph, Question, Triple
from graphrag_lab.bench.encoding import Encoder, top_k
from graphrag_lab.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class Ranking:
    units: list[Unit]
    # Per-question diagnostics recorded alongside metrics (e.g. seed hits).
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    name: str
    unit_kind: UnitKind

    @abstractmethod
    async def rank(self, question: Question) -> Ranking:
        """Units for `question`, best first, enough to fill the largest budget."""


def sentence_units(paragraph: Paragraph) -> list[Unit]:
    return [
        Unit("sentence", (paragraph.title, sentence), sentence, header=paragraph.title)
        for sentence in paragraph.sentences
    ]


def triple_unit(triple: Triple) -> Unit:
    return Unit("triple", triple, triple.text)


def paragraph_text(paragraph: Paragraph) -> str:
    return f"{paragraph.title}: {' '.join(paragraph.sentences)}"


class DenseParagraphRetriever(Retriever):
    """Classic RAG: embed whole paragraphs, return the nearest ones."""

    name = "dense_paragraph"
    unit_kind: UnitKind = "sentence"

    def __init__(self, corpus: Corpus, encoder: Encoder, n_paragraphs: int = 20) -> None:
        self._paragraphs = list(corpus.paragraphs.values())
        self._encoder = encoder
        self._matrix = encoder.documents(
            [paragraph_text(p) for p in self._paragraphs], "paragraphs"
        )
        self._n = n_paragraphs

    def ranked_indices(self, question: Question, n: int) -> list[int]:
        query = self._encoder.queries([question.text])[0]
        return top_k(self._matrix @ query, n)

    async def rank(self, question: Question) -> Ranking:
        units = [
            unit
            for index in self.ranked_indices(question, self._n)
            for unit in sentence_units(self._paragraphs[index])
        ]
        return Ranking(units)


class BM25ParagraphRetriever(Retriever):
    """Lexical baseline: BM25 over paragraphs (title included)."""

    name = "bm25_paragraph"
    unit_kind: UnitKind = "sentence"

    def __init__(self, corpus: Corpus, n_paragraphs: int = 20) -> None:
        import bm25s

        self._paragraphs = list(corpus.paragraphs.values())
        self._bm25s = bm25s
        self._index = bm25s.BM25()
        self._index.index(
            bm25s.tokenize(
                [paragraph_text(p) for p in self._paragraphs], stopwords="en", show_progress=False
            ),
            show_progress=False,
        )
        self._n = n_paragraphs

    def ranked_indices(self, question: Question, n: int) -> list[int]:
        tokens = self._bm25s.tokenize([question.text], stopwords="en", show_progress=False)
        indices, _scores = self._index.retrieve(tokens, k=n, show_progress=False)
        return [int(i) for i in indices[0]]

    async def rank(self, question: Question) -> Ranking:
        units = [
            unit
            for index in self.ranked_indices(question, self._n)
            for unit in sentence_units(self._paragraphs[index])
        ]
        return Ranking(units)


class HybridParagraphRetriever(Retriever):
    """Dense + BM25 paragraph rankings fused with reciprocal rank fusion (the
    same fusion the system uses to seed graph traversal)."""

    name = "hybrid_paragraph"
    unit_kind: UnitKind = "sentence"

    def __init__(
        self,
        dense: DenseParagraphRetriever,
        bm25: BM25ParagraphRetriever,
        corpus: Corpus,
        n_paragraphs: int = 20,
        candidates: int = 50,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._paragraphs = list(corpus.paragraphs.values())
        self._n = n_paragraphs
        self._candidates = candidates

    async def rank(self, question: Question) -> Ranking:
        dense = self._dense.ranked_indices(question, self._candidates)
        lexical = self._bm25.ranked_indices(question, self._candidates)
        fused = reciprocal_rank_fusion(
            [(i, rank) for rank, i in enumerate(dense, start=1)],
            [(i, rank) for rank, i in enumerate(lexical, start=1)],
        )
        units = [
            unit
            for index, _ in fused[: self._n]
            for unit in sentence_units(self._paragraphs[index])
        ]
        return Ranking(units)


class DenseSentenceRetriever(Retriever):
    """Fine-grained text baseline: rank individual sentences, which are close
    to triples in size."""

    name = "dense_sentence"
    unit_kind: UnitKind = "sentence"

    def __init__(self, corpus: Corpus, encoder: Encoder, n_sentences: int = 150) -> None:
        self._units = [u for p in corpus.paragraphs.values() for u in sentence_units(p)]
        self._encoder = encoder
        self._matrix = encoder.documents([u.text for u in self._units], "sentences")
        self._n = n_sentences

    async def rank(self, question: Question) -> Ranking:
        query = self._encoder.queries([question.text])[0]
        return Ranking([self._units[i] for i in top_k(self._matrix @ query, self._n)])


class DenseTripleRetriever(Retriever):
    """Vector search over facts: embed each triple's text and rank by
    similarity, ignoring graph structure. The baseline a structure-aware
    retriever has to beat."""

    name = "dense_triple"
    unit_kind: UnitKind = "triple"

    def __init__(self, corpus: Corpus, encoder: Encoder, n_triples: int = 150) -> None:
        self._triples = corpus.triples
        self._encoder = encoder
        self._matrix = encoder.documents([t.text for t in self._triples], "triples")
        self._n = n_triples

    async def rank(self, question: Question) -> Ranking:
        query = self._encoder.queries([question.text])[0]
        return Ranking(
            [triple_unit(self._triples[i]) for i in top_k(self._matrix @ query, self._n)]
        )


class OracleTripleRetriever(Retriever):
    """Upper bound: exactly the question's gold evidence triples."""

    name = "oracle_triple"
    unit_kind: UnitKind = "triple"

    async def rank(self, question: Question) -> Ranking:
        return Ranking([triple_unit(t) for t in question.evidence])


class OracleSentenceRetriever(Retriever):
    """Upper bound: exactly the question's gold supporting sentences."""

    name = "oracle_sentence"
    unit_kind: UnitKind = "sentence"

    async def rank(self, question: Question) -> Ranking:
        return Ranking(
            [Unit("sentence", (title, s), s, header=title) for title, s in question.supporting]
        )
