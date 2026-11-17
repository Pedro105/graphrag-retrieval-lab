"""The prose answer optionally attached to a RecallResult."""

from typing import Literal

from pydantic import BaseModel, Field

# How well the subgraph supports the answer. insufficient_data means it does
# not answer the query at all, and the model must say so rather than guess.
SynthesisConfidence = Literal["high", "medium", "low", "insufficient_data"]


class SynthesisResult(BaseModel):
    answer: str = Field(min_length=1)
    # Ids of the entities the answer rests on, limited to retrieved ones.
    grounded_entity_ids: list[str] = Field(default_factory=list)
    confidence: SynthesisConfidence
    # What is missing or contested. Expected whenever confidence is not high.
    caveat: str | None = None
