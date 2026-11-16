"""FastAPI application: wiring, logging, and health endpoints."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response, status

from graphrag_lab.api.routes import router
from graphrag_lab.core.config import get_settings
from graphrag_lab.graph.neo4j_store import Neo4jGraphStore
from graphrag_lab.vector.embeddings import build_embedding_provider

APP_VERSION = "0.2.0"

settings = get_settings()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelNamesMapping()[settings.log_level.upper()]
    ),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    app.state.embedding_provider = build_embedding_provider(settings)
    logger.info("app_startup", neo4j_uri=settings.neo4j_uri)
    yield
    await app.state.graph_store.aclose()
    logger.info("app_shutdown")


app = FastAPI(
    title="GraphRAG Lab API",
    version=APP_VERSION,
    description="Ingest text into a knowledge graph and query it with hybrid GraphRAG retrieval.",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
async def health(response: Response) -> dict[str, str]:
    """200 when the graph store is reachable, 503 otherwise."""
    if await app.state.graph_store.health_check():
        return {"status": "ok", "version": APP_VERSION, "graph_store": "ok"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "degraded", "version": APP_VERSION, "graph_store": "unreachable"}
