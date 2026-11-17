"""The result of a recall() call."""

from pydantic import BaseModel, Field

from graphrag_lab.models.classification import ClassifierResult, RetrievalMode
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope
from graphrag_lab.models.synthesis import SynthesisResult

__all__ = ["RecallResult", "RetrievalMode", "RetrievalStats"]


class RetrievalStats(BaseModel):
    """Timings and sizes for one recall."""

    total_ms: float
    retrieval_ms: float  # up to the end of traversal, so without synthesis
    synthesis_ms: float | None  # None when synthesis was not requested
    seeds_found: int
    nodes_traversed: int  # entities in the returned subgraph
    relations_found: int
    depth_reached: int  # the traversal depth that was used
    query_class: str  # "manual" when the caller pinned mode or depth
    used_llm_classifier: bool


class RecallResult(BaseModel):
    query: str = Field(min_length=1)
    scope: Scope
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    # None when the caller pinned mode or depth, since nothing was classified.
    classifier_result: ClassifierResult | None = None
    # The max_entities cap cut the subgraph short.
    truncated: bool = False
    # None unless synthesis was requested and succeeded.
    synthesis: SynthesisResult | None = None
    stats: RetrievalStats
