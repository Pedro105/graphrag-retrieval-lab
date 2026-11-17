"""Reciprocal rank fusion: merge two ranked lists without comparing their
scores, which are on different scales."""

from uuid import UUID

DEFAULT_K_RRF = 60


def reciprocal_rank_fusion(
    vector_results: list[tuple[UUID, int]],
    fulltext_results: list[tuple[UUID, int]],
    k_rrf: int = DEFAULT_K_RRF,
) -> list[tuple[UUID, float]]:
    """Ranks are 1-based. An entity scores the sum of `1 / (rank + k_rrf)`
    over the lists it appears in. Returned best first."""
    scores: dict[UUID, float] = {}
    for entity_id, rank in vector_results:
        scores[entity_id] = scores.get(entity_id, 0.0) + 1.0 / (rank + k_rrf)
    for entity_id, rank in fulltext_results:
        scores[entity_id] = scores.get(entity_id, 0.0) + 1.0 / (rank + k_rrf)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
