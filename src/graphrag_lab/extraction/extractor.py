"""Entity and relation extraction with one structured LLM call."""

from graphrag_lab.core.config import get_settings
from graphrag_lab.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from graphrag_lab.extraction.schemas import ExtractedEntity, ExtractedRelation, ExtractionResult
from graphrag_lab.llm.client import StructuredLLM, get_llm


def _build_user_message(content: str, existing_relation_types: list[str] | None) -> str:
    """Prepend the relation types already in the graph.

    Claims are matched on the exact predicate, so if the model calls the same
    relationship "owns" in one text and "has" in the next, an update becomes
    a second, parallel fact. Showing it the existing vocabulary makes reuse
    much more likely.
    """
    if not existing_relation_types:
        return content
    types_list = ", ".join(sorted(existing_relation_types))
    context = (
        f"Relation types already used elsewhere in this knowledge graph: {types_list}.\n"
        "If the text below describes an update to a relationship that already exists "
        "between the same two entities (e.g. a revised price, a changed status), reuse "
        "the EXACT existing relation_type string rather than inventing a new but "
        "similar one. Only use a different relation_type if the text describes a "
        "genuinely different kind of relationship.\n\n---\n\n"
    )
    return context + content


async def extract(
    content: str,
    existing_relation_types: list[str] | None = None,
    llm: StructuredLLM | None = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation], int]:
    """Returns (entities, relations, tokens used by the original LLM call)."""
    settings = get_settings()
    response = await (llm or get_llm()).generate(
        model=settings.extraction_model,
        system=EXTRACTION_SYSTEM_PROMPT,
        user=_build_user_message(content, existing_relation_types),
        response_model=ExtractionResult,
        max_tokens=4096,
    )
    return response.value.entities, response.value.relations, response.tokens
