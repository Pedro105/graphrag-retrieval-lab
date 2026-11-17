"""What the extraction call returns. Scope and provenance are added later,
once the entities have been resolved against the graph."""

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
    # The field descriptions below are sent to the model as part of the schema.
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
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
