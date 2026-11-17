"""How a recall query is routed. Kept apart from recall.py, which imports it,
to avoid a circular import."""

from typing import Literal

from pydantic import BaseModel, Field

# How seeds are found: vector search, full-text search, or both fused.
RetrievalMode = Literal["vector", "fulltext", "hybrid"]

# exact_lookup: names one entity by id, code or proper noun.
# single_hop:   asks for something directly attached to one entity.
# relational:   asks how entities connect, which needs more than one hop.
QueryClass = Literal["exact_lookup", "single_hop", "relational"]


class RetrievalStrategy(BaseModel):
    retrieval_mode: RetrievalMode
    traversal_depth: int = Field(ge=1)


class ClassifierResult(BaseModel):
    """How one query was classified. `confidence` is the rule score, or a
    fixed constant when the LLM decided (it returns no probability)."""

    query_class: QueryClass
    confidence: float = Field(ge=0.0, le=1.0)
    strategy: RetrievalStrategy
    used_llm: bool
