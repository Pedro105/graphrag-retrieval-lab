"""The write path (remember) and the read path (recall).

remember: extract -> resolve entities -> adjudicate claims -> write.
recall:   pick a strategy -> seed -> traverse -> optionally synthesise.
"""

import asyncio
import re
import time
from uuid import UUID

import structlog

from graphrag_lab.core.classifier import QueryClassifier
from graphrag_lab.core.synthesiser import RecallSynthesiser
from graphrag_lab.extraction.extractor import extract
from graphrag_lab.graph.store import GraphStore
from graphrag_lab.models.classification import ClassifierResult
from graphrag_lab.models.entity import Entity
from graphrag_lab.models.fact_claim import ClaimAssertion
from graphrag_lab.models.memory import Memory
from graphrag_lab.models.provenance import Provenance
from graphrag_lab.models.recall import RecallResult, RetrievalMode, RetrievalStats
from graphrag_lab.models.relation import Relation
from graphrag_lab.models.scope import Scope
from graphrag_lab.resolution.conflict_resolver import adjudicate
from graphrag_lab.resolution.entity_resolver import resolve_entity
from graphrag_lab.retrieval.fusion import reciprocal_rank_fusion
from graphrag_lab.vector.embeddings import EmbeddingProvider

logger = structlog.get_logger()

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]+")

# "Left A" has to retract A's claim before "joined B" is weighed against it.
_ASSERTION_ORDER = {"terminated": 0, "negated": 0, "asserted": 1}


def _normalize_relation_type(relation_type: str) -> str:
    """Turn an extracted relation type ("works at", "Works-At") into the
    snake_case predicate claims are matched on."""
    normalized = _UNSAFE_CHARS.sub("_", relation_type.strip().lower()).strip("_")
    if not normalized:
        return "relates_to"
    if not (normalized[0].isalpha() or normalized[0] == "_"):
        normalized = f"rel_{normalized}"
    return normalized


async def remember(
    content: str,
    scope: Scope,
    source: str,
    graph_store: GraphStore,
    embedding_provider: EmbeddingProvider,
    confidence: float = 1.0,
    evidence: list[str] | None = None,
) -> tuple[Memory, int]:
    """Ingest one piece of text. Returns the Memory and the tokens the
    extraction call used (0 on a cache hit that recorded none)."""
    await graph_store.ensure_graph_initialized(scope.tenant_id, embedding_provider.dimension)

    existing_predicates = await graph_store.find_predicates(scope)
    extracted_entities, extracted_relations, tokens_used = await extract(
        content, existing_predicates
    )
    provenance = Provenance(source=source, confidence=confidence, evidence=evidence)

    # Several mentions can resolve to the same entity, so entities are kept by
    # id and each resolution sees the ones before it (`pending`).
    resolved_by_id: dict[UUID, Entity] = {}
    id_by_name: dict[str, UUID] = {}
    for extracted_entity in extracted_entities:
        resolved = await resolve_entity(
            extracted_entity,
            scope,
            provenance,
            graph_store,
            embedding_provider,
            pending=list(resolved_by_id.values()),
        )
        resolved_by_id[resolved.id] = resolved
        id_by_name[extracted_entity.name] = resolved.id

    relations: list[Relation] = []
    assertions: list[ClaimAssertion] = []
    for extracted_relation in extracted_relations:
        source_id = id_by_name.get(extracted_relation.source_name)
        target_id = id_by_name.get(extracted_relation.target_name)
        if source_id is None or target_id is None:
            logger.warning(
                "remember.skipped_relation_unknown_entity",
                source_name=extracted_relation.source_name,
                target_name=extracted_relation.target_name,
            )
            continue
        source_entity = resolved_by_id[source_id]
        target_entity = resolved_by_id[target_id]
        # The Memory reports what this text said, with the raw relation type.
        # What gets stored is the adjudicated claim, which may differ.
        relations.append(
            Relation(
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=extracted_relation.relation_type,
                properties=extracted_relation.properties,
                scope=scope,
                provenance=provenance,
            )
        )
        assertions.append(
            ClaimAssertion(
                subject_id=source_entity.id,
                object_id=target_entity.id,
                subject_name=source_entity.name,
                object_name=target_entity.name,
                predicate=_normalize_relation_type(extracted_relation.relation_type),
                raw_predicate=extracted_relation.relation_type,
                assertion_type=extracted_relation.assertion_type,
                as_of=extracted_relation.as_of,
                replaces_hint=extracted_relation.replaces_hint,
                properties=extracted_relation.properties,
                provenance=provenance,
                scope=scope,
            )
        )

    memory = Memory(
        content=content,
        entities=list(resolved_by_id.values()),
        relations=relations,
        scope=scope,
        provenance=provenance,
    )
    # Entities go in first so the claims have endpoints to attach to.
    await graph_store.write_memory(memory)
    memory_id = str(memory.id)
    for assertion in sorted(assertions, key=lambda a: _ASSERTION_ORDER[a.assertion_type]):
        candidates = await graph_store.find_active_claims(
            scope, assertion.subject_id, assertion.predicate
        )
        for claim in await adjudicate(assertion, candidates):
            await graph_store.upsert_claim(claim, memory_id)

    return memory, tokens_used


async def seed_entities(
    query: str,
    scope: Scope,
    graph_store: GraphStore,
    embedding_provider: EmbeddingProvider,
    limit: int,
    retrieval_mode: RetrievalMode,
) -> list[Entity]:
    """Entry points into the graph for a query: vector search, full-text
    search, or both fused with reciprocal rank fusion. Best match first."""
    vector_seeds: list[tuple[Entity, float]] = []
    fulltext_ranked: list[tuple[UUID, int]] = []

    if retrieval_mode in ("vector", "hybrid"):
        embedding = await embedding_provider.embed(query)
        vector_seeds = await graph_store.find_similar_entities(scope, embedding, None, limit)

    if retrieval_mode in ("fulltext", "hybrid"):
        fulltext_ranked = await graph_store.find_by_fulltext(scope, query, limit)

    if retrieval_mode == "vector":
        return [entity for entity, _ in vector_seeds]

    # Full-text search returns ids only, so those entities still need fetching.
    if retrieval_mode == "fulltext":
        fetched = await asyncio.gather(
            *(
                graph_store.get_entity(entity_id, scope.tenant_id)
                for entity_id, _ in fulltext_ranked
            )
        )
        return [entity for entity in fetched if entity is not None]

    vector_ranked = [(entity.id, rank) for rank, (entity, _) in enumerate(vector_seeds, start=1)]
    fused = reciprocal_rank_fusion(vector_ranked, fulltext_ranked)
    top_ids = [entity_id for entity_id, _ in fused[:limit]]

    entities_by_id = {entity.id: entity for entity, _ in vector_seeds}
    missing_ids = [entity_id for entity_id in top_ids if entity_id not in entities_by_id]
    if missing_ids:
        fetched = await asyncio.gather(
            *(graph_store.get_entity(entity_id, scope.tenant_id) for entity_id in missing_ids)
        )
        for entity_id, entity in zip(missing_ids, fetched, strict=True):
            if entity is not None:
                entities_by_id[entity_id] = entity

    return [entities_by_id[entity_id] for entity_id in top_ids if entity_id in entities_by_id]


async def _resolve_strategy(
    query: str,
    traversal_depth: int | None,
    retrieval_mode: RetrievalMode | None,
    classifier: QueryClassifier | None,
) -> tuple[RetrievalMode, int, ClassifierResult | None]:
    """Work out the retrieval mode and depth for this recall.

    If the caller set either one, the classifier is skipped and the other
    falls back to hybrid / depth 1. Experiments rely on this: pinning a knob
    must never trigger an LLM call that could change the routing.
    """
    if traversal_depth is not None or retrieval_mode is not None:
        return retrieval_mode or "hybrid", traversal_depth or 1, None

    classifier = classifier or QueryClassifier()
    result = await classifier.classify(query)
    logger.debug(
        "recall.classified",
        query=query,
        query_class=result.query_class,
        confidence=result.confidence,
        used_llm=result.used_llm,
    )
    return result.strategy.retrieval_mode, result.strategy.traversal_depth, result


async def recall(
    query: str,
    scope: Scope,
    graph_store: GraphStore,
    embedding_provider: EmbeddingProvider,
    limit: int = 10,
    traversal_depth: int | None = None,
    retrieval_mode: RetrievalMode | None = None,
    classifier: QueryClassifier | None = None,
    max_entities: int = 200,
    synthesise: bool = False,
    synthesiser: RecallSynthesiser | None = None,
) -> RecallResult:
    """Retrieve the subgraph relevant to `query`.

    `limit` bounds the seeds only: the result usually holds more entities
    than that once their neighbours are added. `max_entities` bounds the
    whole result, and `truncated` is set when it was hit.

    Synthesis is best effort. If the LLM call fails the structured result is
    still returned, with `synthesis` left as None.
    """
    start = time.perf_counter()

    effective_mode, effective_depth, classifier_result = await _resolve_strategy(
        query, traversal_depth, retrieval_mode, classifier
    )

    seeds = await seed_entities(
        query, scope, graph_store, embedding_provider, limit, effective_mode
    )
    truncated = len(seeds) > max_entities
    seeds = seeds[:max_entities]

    # Seeds are kept as fetched because they carry their embedding; the
    # traversal returns them again without it.
    entities_by_id: dict[UUID, Entity] = {entity.id: entity for entity in seeds}
    relations: list[Relation] = []

    if seeds:
        reached, relations, hit_cap = await graph_store.traverse_from_seeds(
            scope, [str(entity.id) for entity in seeds], effective_depth, max_entities
        )
        truncated = truncated or hit_cap
        for entity in reached:
            entities_by_id.setdefault(entity.id, entity)
        if truncated:
            logger.warning(
                "recall.truncated",
                tenant_id=scope.tenant_id,
                query=query,
                depth=effective_depth,
                entity_count=len(entities_by_id),
                max_entities=max_entities,
            )

    entities = list(entities_by_id.values())
    relations = [
        relation
        for relation in {relation.id: relation for relation in relations}.values()
        if relation.source_entity_id in entities_by_id
        and relation.target_entity_id in entities_by_id
    ]

    retrieval_end = time.perf_counter()

    result = RecallResult(
        query=query,
        scope=scope,
        entities=entities,
        relations=relations,
        classifier_result=classifier_result,
        truncated=truncated,
        stats=RetrievalStats(
            total_ms=0.0,
            retrieval_ms=(retrieval_end - start) * 1000,
            synthesis_ms=None,
            seeds_found=len(seeds),
            nodes_traversed=len(entities),
            relations_found=len(relations),
            depth_reached=effective_depth,
            query_class=classifier_result.query_class if classifier_result else "manual",
            used_llm_classifier=classifier_result.used_llm if classifier_result else False,
        ),
    )

    if synthesise:
        try:
            result.synthesis = await (synthesiser or RecallSynthesiser()).synthesise(
                query, result, scope
            )
        except Exception:
            logger.exception("recall.synthesis_failed", tenant_id=scope.tenant_id, query=query)
        result.stats.synthesis_ms = (time.perf_counter() - retrieval_end) * 1000

    result.stats.total_ms = (time.perf_counter() - start) * 1000
    return result
