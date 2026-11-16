"""LLM-based entity/relation extraction (structured output via llm/client.py)."""

from graphrag_lab.core.config import get_settings
from graphrag_lab.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from graphrag_lab.extraction.schemas import ExtractedEntity, ExtractedRelation, ExtractionResult
from graphrag_lab.llm.client import StructuredLLM, get_llm


def _build_user_message(content: str, existing_relation_types: list[str] | None) -> str:
    """Bias extraction toward reusing an existing relation_type for the same
    conceptual fact, rather than inventing a new-but-synonymous string on
    every write.

    Why this exists: claim adjudication (resolution/conflict_resolver.py)
    matches facts on an exact normalized predicate. relation_type is otherwise a free
    string (extraction/schemas.py), so nothing previously stopped the model
    from extracting "owns" one call and "relates_to" another for what's
    really an update to the same relationship -- the second write then
    silently created a parallel edge instead of updating the first (see the
    eval harness's globex_current_price_single_hop case). Telling the model
    what's already in the graph lets it make the same call a human curator
    would: reuse the existing string for an update to the same kind of
    relationship, but still mint a new one when the text genuinely
    describes something different. Tenant-wide rather than entity-pair-
    specific, since which pair is involved isn't known until after this
    same extraction call resolves entity names.
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
    """Extract entities and relations from `content`. Also returns the tokens
    the (original, uncached) LLM call consumed, for cost accounting."""
    settings = get_settings()
    response = await (llm or get_llm()).generate(
        model=settings.extraction_model,
        system=EXTRACTION_SYSTEM_PROMPT,
        user=_build_user_message(content, existing_relation_types),
        response_model=ExtractionResult,
        max_tokens=4096,
    )
    return response.value.entities, response.value.relations, response.tokens
