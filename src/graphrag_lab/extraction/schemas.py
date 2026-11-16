"""Extraction-time intermediate representations.

These are deliberately not the full Entity/Relation models: scope and
provenance are attached later by the orchestrating service
(core/service.py), once entities have been resolved against the graph.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

AssertionType = Literal["asserted", "terminated", "negated"]


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    source_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    # Assertion semantics: let extraction represent a relationship that has
    # ended or never held instead of dropping that signal. Consumed by claim
    # adjudication (resolution/conflict_resolver.py), not by the display
    # Relation model.
    assertion_type: AssertionType = Field(
        default="asserted",
        description=(
            "How the text frames this relation: 'asserted' = it currently holds "
            "(default); 'terminated' = the text says it ended / no longer holds "
            "(left, quit, former); 'negated' = the text explicitly denies it "
            "(does not work at, never)."
        ),
    )
    as_of: str | None = Field(
        default=None,
        description=(
            "Effective-time marker the text gives for this assertion ('now', "
            "'since March 2024', a date), or null if none is stated."
        ),
    )
    replaces_hint: bool = Field(
        default=False,
        description=(
            "True when the text presents this relationship as the current state "
            "replacing a prior one (e.g. 'now works at ...'). A hint for later "
            "supersession; it does not by itself change anything."
        ),
    )
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Wrapper used as the Instructor response_model (a single structured call)."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
