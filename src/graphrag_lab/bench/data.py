"""2WikiMultiHopQA loading and corpus construction.

Each 2WikiMultiHopQA question ships its answer, ten context paragraphs (the
supporting ones plus distractors), sentence-level supporting facts, and the
gold reasoning chain as Wikidata-style evidence triples. That last field is
what makes the benchmark usable without an LLM: the union of evidence triples
is a gold knowledge graph, and each question's own triples are its
retrieval labels.

The benchmark pools a seeded sample of questions into one shared corpus
(every sampled question's paragraphs and triples), so each question is
answered against distractors from all the others rather than only its own
ten paragraphs.

Background knowledge. A graph built only from the sample's gold triples
would be unrealistically clean: every fact in it answers some question, so
a question entity's neighbourhood is almost all signal. To give the graph
realistic neighbourhoods, the corpus also pulls in every *training*
question that shares a specific entity with the sample (excluding hubs such
as "American" and date literals). Their gold triples go into the graph and
their supporting paragraphs -- the text those facts come from -- go into
the text corpus, so both modalities gain the same distractor knowledge.
Background questions are never evaluated.
"""

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

QuestionType = Literal["compositional", "inference", "comparison", "bridge_comparison"]

# Question types whose answer is an entity reached through the chain (not
# one of the entities named in the question, and never yes/no).
BRIDGE_TYPES: frozenset[str] = frozenset({"compositional", "inference"})

TWO_WIKI_REPO = "framolfese/2WikiMultihopQA"
# Pinned dataset revision, so every run reads byte-identical data.
TWO_WIKI_REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"
TWO_WIKI_FILES = {
    "validation": ["data/validation-00000-of-00001.parquet"],
    "train": [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    ],
}

# Entities containing a 3-4 digit number are dates or years: shared by many
# unrelated facts, so they never make two questions "related".
_LITERAL = re.compile(r"\d{3,4}")


@dataclass(frozen=True, order=True)
class Triple:
    subject: str
    predicate: str
    object: str

    @property
    def text(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"


@dataclass(frozen=True)
class Paragraph:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    answer: str
    type: str
    evidence: tuple[Triple, ...]
    # (paragraph title, sentence text) pairs that support the answer. Keyed by
    # text rather than index: ~2% of repeated titles in the dataset carry
    # differently split text, so an index into the pooled copy could point at
    # the wrong sentence.
    supporting: tuple[tuple[str, str], ...]

    @property
    def is_bridge(self) -> bool:
        return self.type in BRIDGE_TYPES

    @property
    def topic_entity(self) -> str:
        """The chain's starting entity: the subject of the first evidence triple."""
        return self.evidence[0].subject


@dataclass
class Corpus:
    """Pooled corpus over a sample of questions, plus the evaluation subset."""

    questions: list[Question]
    eval_questions: list[Question]
    paragraphs: dict[str, Paragraph]
    triples: list[Triple]
    background_questions: int = 0
    fingerprint: str = field(default="")

    @property
    def entities(self) -> list[str]:
        names = {t.subject for t in self.triples} | {t.object for t in self.triples}
        return sorted(names)


def download_two_wiki(split: str = "validation", cache_dir: Path | None = None) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(
                repo_id=TWO_WIKI_REPO,
                filename=filename,
                repo_type="dataset",
                revision=TWO_WIKI_REVISION,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        for filename in TWO_WIKI_FILES[split]
    ]


def parse_row(row: dict[str, Any]) -> tuple[Question, list[Paragraph]]:
    """One parquet row -> its Question and its ten context paragraphs."""
    context = row["context"]
    paragraphs = [
        Paragraph(title=title, sentences=tuple(sentences))
        for title, sentences in zip(context["title"], context["sentences"], strict=True)
    ]
    by_title = {paragraph.title: paragraph.sentences for paragraph in paragraphs}
    supporting = tuple(
        (title, by_title[title][index])
        for title, index in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True
        )
        if title in by_title and index < len(by_title[title])
    )
    question = Question(
        id=row["id"],
        text=row["question"],
        answer=row["answer"],
        type=row["type"],
        evidence=tuple(Triple(*triple) for triple in row["evidences"]),
        supporting=supporting,
    )
    return question, paragraphs


def load_rows(paths: Sequence[Path], columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def load_rows_by_id(paths: Sequence[Path], ids: set[str]) -> list[dict[str, Any]]:
    """Full rows for just `ids`, without materializing the rest of the split."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = pa.array(sorted(ids))
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        rows.extend(table.filter(pc.is_in(table["id"], value_set=wanted)).to_pylist())
    return rows


def key_entities(evidence: Iterable[Sequence[str]], degree: Counter[str], hub_max: int) -> set[str]:
    """Entities specific enough to link two questions: not hubs, not dates."""
    return {
        entity
        for subject, _, obj in evidence
        for entity in (subject, obj)
        if degree[entity] <= hub_max and not _LITERAL.search(entity)
    }


def related_ids(
    sample: list[dict[str, Any]], background: list[dict[str, Any]], hub_max: int
) -> set[str]:
    """Ids of background questions sharing a key entity with the sample.

    `background` rows only need `id` and `evidences`. Entity degree is
    counted over sample and background together.
    """
    degree: Counter[str] = Counter()
    for row in (*sample, *background):
        for subject, _, obj in row["evidences"]:
            degree[subject] += 1
            degree[obj] += 1
    anchors = set().union(*(key_entities(r["evidences"], degree, hub_max) for r in sample))
    return {
        row["id"] for row in background if key_entities(row["evidences"], degree, hub_max) & anchors
    }


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded sample of questions that have evidence triples (no labels
    otherwise), in sampled order."""
    usable = [row for row in rows if row["evidences"]]
    return random.Random(seed).sample(usable, min(n, len(usable)))


def build_corpus(
    sample: list[dict[str, Any]],
    n_eval: int,
    background: list[dict[str, Any]] | None = None,
) -> Corpus:
    """Pool the sampled questions' paragraphs and evidence triples, and take
    the first `n_eval` of the sample for evaluation.

    `background` rows (full rows, selected with `related_ids`) add their
    evidence triples and supporting paragraphs, and are never evaluated.
    Paragraphs are deduplicated by title (first occurrence wins).
    """
    if n_eval > len(sample):
        raise ValueError("n_eval cannot exceed the sample size")

    questions: list[Question] = []
    paragraphs: dict[str, Paragraph] = {}
    triples: set[Triple] = set()
    for row in sample:
        question, row_paragraphs = parse_row(row)
        questions.append(question)
        triples.update(question.evidence)
        for paragraph in row_paragraphs:
            paragraphs.setdefault(paragraph.title, paragraph)

    background_ids: list[str] = []
    for row in sorted(background or [], key=lambda r: str(r["id"])):
        question, row_paragraphs = parse_row(row)
        background_ids.append(question.id)
        triples.update(question.evidence)
        supporting_titles = {title for title, _ in question.supporting}
        for paragraph in row_paragraphs:
            if paragraph.title in supporting_titles:
                paragraphs.setdefault(paragraph.title, paragraph)

    corpus = Corpus(
        questions=questions,
        eval_questions=questions[:n_eval],
        paragraphs=paragraphs,
        triples=sorted(triples),
        background_questions=len(background_ids),
    )
    corpus.fingerprint = fingerprint(corpus, background_ids)
    return corpus


def fingerprint(corpus: Corpus, background_ids: Sequence[str] = ()) -> str:
    """Stable short hash of the corpus content, used to key caches and the
    Neo4j tenant so a different sample never reuses stale indexes."""
    digest = hashlib.sha256()
    digest.update(json.dumps([q.id for q in corpus.questions]).encode())
    digest.update(json.dumps([q.id for q in corpus.eval_questions]).encode())
    digest.update(json.dumps(list(background_ids)).encode())
    return digest.hexdigest()[:12]
