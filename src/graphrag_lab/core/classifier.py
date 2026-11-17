"""Query classifier: picks the seeding mode and traversal depth for a recall.

Cheap rules handle the queries with an obvious shape. Only when they are
unsure is the LLM asked. Background and tuning notes are in
docs/classifier_notes.md.
"""

import re

import structlog
from pydantic import BaseModel, Field

from graphrag_lab.core.config import get_settings
from graphrag_lab.llm.client import StructuredLLM, get_llm
from graphrag_lab.models.classification import (
    ClassifierResult,
    QueryClass,
    RetrievalStrategy,
)

logger = structlog.get_logger()

# Rules scoring at or above this are trusted; anything lower goes to the LLM.
# The scores below are hand-set relative to this cutoff, not measured.
CONFIDENCE_THRESHOLD = 0.85

# The LLM returns a class, not a probability, so its results get a fixed value.
_LLM_RESULT_CONFIDENCE = 0.7

# Full-text finds identifiers like "RG-88" that vector search tends to miss.
CLASS_TO_STRATEGY: dict[QueryClass, RetrievalStrategy] = {
    "exact_lookup": RetrievalStrategy(retrieval_mode="fulltext", traversal_depth=1),
    "single_hop": RetrievalStrategy(retrieval_mode="hybrid", traversal_depth=1),
    "relational": RetrievalStrategy(retrieval_mode="hybrid", traversal_depth=2),
}

_QUESTION_WORDS = {"who", "what", "where", "which", "when", "whom", "whose", "how", "why"}

# "How many cells does X produce" asks for one attribute, so it must not be
# read as a relational "how" question.
_QUANTITATIVE_RE = re.compile(
    r"\bhow\s+(many|much|long|old|big|far|tall|heavy|often|fast|wide|deep)\b"
)

# Phrases that ask for a connection or a cause. Matched as substrings of the
# lowercased query, so stems cover inflections (connect, connected, connection).
_RELATIONAL_CUES = (
    "root cause",
    "connect",  # connected to, connection between
    "related",
    "relationship",
    "relate to",
    "relates to",
    "led to",
    "caused by",
    "what caused",
    "which caused",
    "result of",
    "due to",
    "path between",
    "path from",
    "linked",
    "link between",
    "influence",  # influence(d) on/by
    "impact of",
    "stem from",
    "stems from",
    "arise from",
    "arises from",
    "trace",
    "downstream",
    "upstream",
    "depend on",
    "depends on",
)

# Verbs that, after who/what/where/which, mark a one-hop question.
_ATTRIBUTE_VERBS = (
    "lead",
    "manage",
    "manufacture",
    "make",
    "produce",
    "use",
    "work",
    "build",
    "own",
    "sponsor",
    "grant",
    "sell",
    "supply",
    "develop",
    "head",
    "run",
    "employ",
    "found",
    "create",
    "provide",
    "offer",
    "report to",
    "belong",
)
_ATTRIBUTE_VERB_RE = re.compile(
    r"\b(" + "|".join(v.replace(" ", r"\s+") for v in _ATTRIBUTE_VERBS) + r")(s|es|ed|ing)?\b"
)

# A code mixes letters and digits ("RG-88", "R-2025-014"); "2025" alone is not one.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'./-]*")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
_HAS_DIGIT_RE = re.compile(r"\d")


def _is_id_code(token: str) -> bool:
    return bool(_HAS_LETTER_RE.search(token) and _HAS_DIGIT_RE.search(token))


def _is_proper_noun_phrase(tokens: list[str]) -> bool:
    """A bare name such as "Korrigan Cells Ltd" or "Dr. Sarah Chen": starts
    with a capital and most of its words are capitalised."""
    if not tokens:
        return False
    alpha_tokens = [t for t in tokens if _HAS_LETTER_RE.search(t)]
    if not alpha_tokens or not alpha_tokens[0][0].isupper():
        return False
    capitalized = sum(1 for t in alpha_tokens if t[0].isupper())
    return capitalized * 2 > len(alpha_tokens)


class HeuristicClassifier:
    """Rule-based classification. Returns (class, confidence in [0, 1]).

    The rules are tried in order and the first match wins. Relational phrasing
    is checked before codes so that "how is X connected to RG-88" is not
    mistaken for a lookup of RG-88.
    """

    def classify(self, query: str) -> tuple[QueryClass, float]:
        raw = query.strip()
        lower = raw.lower()
        tokens = _TOKEN_RE.findall(raw)
        n = len(tokens)
        first = tokens[0].lower() if tokens else ""
        qword = first if first in _QUESTION_WORDS else None
        quantitative = bool(_QUANTITATIVE_RE.search(lower))

        if any(cue in lower for cue in _RELATIONAL_CUES):
            return "relational", 0.9

        # A bare how/why leans relational, but not enough to skip the LLM.
        if qword in {"how", "why"} and not quantitative:
            return "relational", 0.78

        # A code or a bare name, not phrased as a question.
        if qword is None and any(_is_id_code(t) for t in tokens):
            return "exact_lookup", 0.95 if n <= 4 else 0.7
        if qword is None and _is_proper_noun_phrase(tokens) and n <= 5:
            return "exact_lookup", 0.88

        if quantitative:
            return "single_hop", 0.86
        if qword in {"who", "what", "where", "which"} and _ATTRIBUTE_VERB_RE.search(lower):
            return "single_hop", 0.88
        if qword in {"who", "what", "where", "which"}:
            return "single_hop", 0.6

        # No rule matched; the low score sends it to the LLM.
        return "single_hop", 0.3


CLASSIFIER_SYSTEM_PROMPT = """\
You classify a search query for a knowledge-graph memory system into exactly \
one of three retrieval classes. Respond only with the structured fields.

- exact_lookup: the query names a single specific entity by identifier, \
proper noun, product code, or SKU, and the answer is that entity itself or \
its own properties. Examples: "RG-88", "Recall R-2025-014", "Korrigan Cells \
Ltd", "LFP-9 Cell".
- single_hop: the query asks about a direct relationship of one clear \
subject entity, or an attribute reachable one hop away. Examples: "What does \
Vanta Battery Co manufacture?", "Who leads supply chain at Solstice \
Motors?", "How many cells does Vanta produce?".
- relational: the query asks to connect entities that do not co-occur \
directly, or asks for a cause/influence/path, requiring multiple hops. \
Examples: "What is the root cause of the Solstice recall?", "Which \
institution influenced AuroraML's architecture?", "How is Dr. Chen connected \
to RG-88?".

Pick the single best class. If a query mentions an identifier but asks how it \
connects to or is caused by something else, it is relational, not \
exact_lookup. Put a one-sentence justification in `reasoning`.\
"""


class _LLMClassification(BaseModel):
    # `reasoning` is only logged.
    query_class: QueryClass
    reasoning: str = Field(min_length=1)


class LLMClassifier:
    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self._llm = llm

    async def classify(self, query: str) -> _LLMClassification:
        settings = get_settings()
        response = await (self._llm or get_llm()).generate(
            model=settings.classifier_model,
            system=CLASSIFIER_SYSTEM_PROMPT,
            user=query,
            response_model=_LLMClassification,
            max_tokens=512,
        )
        return response.value


class QueryClassifier:
    """Rules first, LLM only when the rules are below `threshold`."""

    def __init__(
        self,
        heuristic: HeuristicClassifier | None = None,
        llm: LLMClassifier | None = None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._heuristic = heuristic or HeuristicClassifier()
        self._llm = llm or LLMClassifier()
        self._threshold = threshold

    async def classify(self, query: str) -> ClassifierResult:
        query_class, confidence = self._heuristic.classify(query)
        used_llm = False

        if confidence < self._threshold:
            llm_result = await self._llm.classify(query)
            logger.debug(
                "classifier.llm_fallback",
                query=query,
                heuristic_class=query_class,
                heuristic_confidence=confidence,
                llm_class=llm_result.query_class,
                reasoning=llm_result.reasoning,
            )
            query_class = llm_result.query_class
            confidence = _LLM_RESULT_CONFIDENCE
            used_llm = True

        return ClassifierResult(
            query_class=query_class,
            confidence=confidence,
            strategy=CLASS_TO_STRATEGY[query_class],
            used_llm=used_llm,
        )
