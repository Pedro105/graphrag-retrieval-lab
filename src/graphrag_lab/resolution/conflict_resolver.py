"""Claim adjudication: what a new assertion does to the active claims about
the same subject and predicate.

Most cases are decided by rule (see the table in docs/architecture.md). The
LLM judge is asked only when the new assertion names a different object and
the text gives no hint that it replaces the old one. Whether a predicate is
single-valued is never configured; `works_at A` vs `works_at B` and
`collaborated_with X` vs `collaborated_with Y` are told apart per case.

`adjudicate` returns the claims to upsert: the new one and any existing ones
whose status changed. Nothing is ever deleted.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from graphrag_lab.core.config import get_settings
from graphrag_lab.llm.client import StructuredLLM, get_llm
from graphrag_lab.models.claim import apply_claim, derive_active_view
from graphrag_lab.models.fact_claim import ClaimAssertion, ClaimStatus, FactClaim

logger = structlog.get_logger()

# A judge verdict less confident than this becomes a dispute, not a supersede.
JUDGE_CONFIDENCE_THRESHOLD = 0.6


def _now() -> datetime:
    return datetime.now(UTC)


def _new_claim(
    assertion: ClaimAssertion,
    status: ClaimStatus,
    *,
    valid_to: datetime | None = None,
    supersedes: list[UUID] | None = None,
) -> FactClaim:
    claims = apply_claim([], assertion.provenance, assertion.properties)
    properties, _provenance, _sources = derive_active_view(claims)
    return FactClaim(
        predicate=assertion.predicate,
        raw_predicate=assertion.raw_predicate,
        subject_id=assertion.subject_id,
        object_id=assertion.object_id,
        status=status,
        asserted_by=[assertion.asserter],
        support_count=1,
        confidence=assertion.provenance.confidence,
        valid_to=valid_to,
        supersedes=list(supersedes or []),
        properties=properties,
        claims=claims,
        scope=assertion.scope,
    )


def _corroborate(claim: FactClaim, assertion: ClaimAssertion) -> FactClaim:
    """The same fact asserted again: add support and any revised properties."""
    merged_claims = apply_claim(claim.claims, assertion.provenance, assertion.properties)
    properties, _provenance, _sources = derive_active_view(merged_claims)
    asserted_by = (
        claim.asserted_by
        if assertion.asserter in claim.asserted_by
        else [*claim.asserted_by, assertion.asserter]
    )
    return claim.model_copy(
        update={
            "support_count": claim.support_count + 1,
            "asserted_by": asserted_by,
            "properties": properties,
            "claims": merged_claims,
            "last_seen": _now(),
        }
    )


def _retract(claim: FactClaim) -> FactClaim:
    return claim.model_copy(update={"status": "retracted", "valid_to": _now()})


def _supersede(assertion: ClaimAssertion, candidates: list[FactClaim]) -> list[FactClaim]:
    now = _now()
    superseded = [
        c.model_copy(update={"status": "superseded", "valid_to": now}) for c in candidates
    ]
    new = _new_claim(assertion, "active", supersedes=[c.id for c in candidates])
    return [*superseded, new]


def _dispute(assertion: ClaimAssertion, candidates: list[FactClaim]) -> list[FactClaim]:
    new = _new_claim(assertion, "disputed")
    new = new.model_copy(update={"disputed_with": [c.id for c in candidates]})
    disputed_candidates = [
        c.model_copy(
            update={
                "status": "disputed",
                "disputed_with": (
                    c.disputed_with if new.id in c.disputed_with else [*c.disputed_with, new.id]
                ),
            }
        )
        for c in candidates
    ]
    return [new, *disputed_candidates]


def _has_other_asserter(candidates: list[FactClaim], asserter: str) -> bool:
    return any(source != asserter for c in candidates for source in c.asserted_by)


class AdjudicationVerdict(BaseModel):
    verdict: Literal["independent", "supersede", "dispute"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


_JUDGE_SYSTEM_PROMPT = """\
You adjudicate a potential conflict between two facts in a knowledge graph. A \
subject already has one or more ACTIVE relationships of a given type, and a new \
write asserts the same relationship type from the same subject toward a \
DIFFERENT object. Decide which of three things is true:

- independent: the subject can genuinely hold BOTH at once -- the relationship \
type is naturally multi-valued (e.g. collaborated_with, knows, mentions, \
authored). Both facts stay true. Choose this whenever the objects are not \
mutually exclusive.
- supersede: the new fact REPLACES the old one -- the relationship type is \
single-valued for a subject at a time (e.g. current employer, current status, \
capital_of) and the new assertion is an update to the same slot.
- dispute: the relationship is single-valued AND the two facts genuinely \
contradict each other with no signal that one replaces the other (so they \
should be flagged as conflicting, not silently merged).

Judge primarily from the relationship type's natural cardinality. Put a \
one-sentence justification in `reason` and a calibrated confidence in [0,1].\
"""


class ClaimJudge:
    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self._llm = llm

    async def judge(
        self, assertion: ClaimAssertion, candidates: list[FactClaim]
    ) -> AdjudicationVerdict:
        settings = get_settings()
        existing = (
            ", ".join(sorted({c.raw_predicate for c in candidates})) or assertion.raw_predicate
        )
        user = (
            f"Subject: {assertion.subject_name}\n"
            f"Relationship type: {assertion.raw_predicate}\n"
            f"Existing active object(s) for this subject+type: "
            f"{len(candidates)} (type {existing})\n"
            f"New asserted object: {assertion.object_name}\n"
            "Does the new fact stand independently, supersede the existing one(s), "
            "or dispute them?"
        )
        response = await (self._llm or get_llm()).generate(
            model=settings.classifier_model,
            system=_JUDGE_SYSTEM_PROMPT,
            user=user,
            response_model=AdjudicationVerdict,
            max_tokens=512,
        )
        return response.value


async def adjudicate(
    assertion: ClaimAssertion,
    candidates: list[FactClaim],
    judge: ClaimJudge | None = None,
) -> list[FactClaim]:
    """Weigh `assertion` against the active claims with the same subject and
    predicate, and return the claims to upsert."""
    same_object = [c for c in candidates if c.object_id == assertion.object_id]
    other_object = [c for c in candidates if c.object_id != assertion.object_id]

    if assertion.assertion_type in ("terminated", "negated"):
        if same_object:
            return [_retract(c) for c in same_object]
        # Nothing to retract. Store the denial itself, as a retracted claim.
        return [_new_claim(assertion, "retracted", valid_to=_now())]

    if same_object:
        return [_corroborate(same_object[0], assertion)]

    if not other_object:
        return [_new_claim(assertion, "active")]

    # A different object for the same subject and predicate.
    cross_asserter = _has_other_asserter(other_object, assertion.asserter)

    if assertion.has_recency_signal:
        # The text presents this as the new state. A source may update its
        # own claim, but never overwrite someone else's.
        decision: Literal["supersede", "dispute"] = "dispute" if cross_asserter else "supersede"
    else:
        verdict = await (judge or ClaimJudge()).judge(assertion, other_object)
        logger.debug(
            "adjudicate.llm",
            subject=str(assertion.subject_id),
            predicate=assertion.predicate,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            reason=verdict.reason,
        )
        if verdict.verdict == "independent":
            return [_new_claim(assertion, "active")]
        if cross_asserter:
            decision = "dispute"
        elif verdict.confidence < JUDGE_CONFIDENCE_THRESHOLD:
            decision = "dispute"
        else:
            decision = verdict.verdict

    if decision == "supersede":
        return _supersede(assertion, other_object)
    return _dispute(assertion, other_object)
