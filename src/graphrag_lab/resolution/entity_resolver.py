"""Entity resolution: is an extracted entity one we already have, a possible
duplicate of one, or new?"""

from collections.abc import Sequence
from typing import Any

from graphrag_lab.core.config import get_settings
from graphrag_lab.extraction.schemas import ExtractedEntity
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.claim import apply_claim, derive_active_view
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.scope import Scope
from graphrag_lab.vector.embeddings import EmbeddingProvider

# A few more than the one best match we need, so an exact-name match is still
# in the list when near-synonyms outrank it.
_CANDIDATE_SEARCH_K = 5

# Subtracted from the similarity when the types disagree. Small on purpose:
# two agents often type the same thing differently (Organization vs Company),
# so type should break ties, not block a merge.
_TYPE_MISMATCH_PENALTY = 0.05


def _normalize_name(name: str) -> str:
    """Case- and whitespace-insensitive key for a name.

    Embeddings are too sensitive to casing to rely on here: "Ada Lovelace"
    and "Ada lovelace" score about 0.90 with text-embedding-3-small, under
    the 0.92 merge threshold.
    """
    return " ".join(name.split()).casefold()


async def resolve_entity(
    extracted: ExtractedEntity,
    scope: Scope,
    provenance: Provenance,
    graph_store: GraphStore,
    embedding_provider: EmbeddingProvider,
    pending: Sequence[Entity] = (),
) -> Entity:
    """Return the entity `extracted` refers to: an existing one with this
    write's properties merged in, or a new one.

    `pending` holds entities resolved earlier in the same write. They are
    not in the store yet, so without it "Acme" and "ACME" in one text would
    become two nodes.
    """
    settings = get_settings()
    normalized = _normalize_name(extracted.name)
    for entity in pending:
        if _normalize_name(entity.name) == normalized:
            return _merge_into(entity, provenance, extracted.properties, extracted.entity_type)

    # The embedding is of the name alone. Prefixing the type pushes the same
    # name apart ("Organization: Acme" vs "Product: Acme" score about 0.85).
    embedding = await embedding_provider.embed(extracted.name)
    candidates = await graph_store.find_similar_entities(
        scope, embedding, None, _CANDIDATE_SEARCH_K
    )
    # If this write already merged into a candidate, build on that version.
    pending_by_id = {entity.id: entity for entity in pending}
    candidates = [(pending_by_id.get(entity.id, entity), score) for entity, score in candidates]

    merge_candidates = []
    if candidates:
        # Same name means same entity, whatever the similarity score or type.
        for candidate, _score in candidates:
            if _normalize_name(candidate.name) == normalized:
                return _merge_into(
                    candidate, provenance, extracted.properties, extracted.entity_type
                )

        best, best_score = candidates[0]
        score = _type_adjusted_score(best, best_score, extracted.entity_type)
        if score >= settings.vector_merge_threshold:
            return _merge_into(best, provenance, extracted.properties, extracted.entity_type)
        if score >= settings.vector_candidate_threshold:
            merge_candidates = [best.id]

    claims = apply_claim([], provenance, extracted.properties)
    properties, active_provenance, contributing_sources = derive_active_view(claims)
    return Entity(
        name=extracted.name,
        entity_type=extracted.entity_type,
        observed_types=[extracted.entity_type],
        properties=properties,
        scope=scope,
        provenance=active_provenance,
        embedding=embedding,
        merge_candidates=merge_candidates,
        claims=claims,
        contributing_sources=contributing_sources,
    )


def _type_adjusted_score(entity: Entity, score: float, extracted_type: str) -> float:
    observed = entity.observed_types or [entity.entity_type]
    if extracted_type in observed:
        return score
    return score - _TYPE_MISMATCH_PENALTY


def _merge_into(
    entity: Entity, provenance: Provenance, properties: dict[str, Any], new_type: str
) -> Entity:
    """Add this write's claims to an existing entity. Its id, name, type and
    embedding stay as they were; a differing type is recorded in
    `observed_types`."""
    claims = apply_claim(entity.claims, provenance, properties)
    merged_properties, active_provenance, contributing_sources = derive_active_view(claims)
    observed = entity.observed_types or [entity.entity_type]
    observed_types = observed if new_type in observed else [*observed, new_type]
    return entity.model_copy(
        update={
            "properties": merged_properties,
            "provenance": active_provenance,
            "claims": claims,
            "contributing_sources": contributing_sources,
            "observed_types": observed_types,
        }
    )
