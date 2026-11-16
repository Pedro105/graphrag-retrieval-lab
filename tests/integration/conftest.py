"""Shared fixtures for the integration suite, which runs against a live Neo4j
(`make up`). Tests that also need real LLM calls skip themselves when no API
key is configured.

If Neo4j isn't reachable the whole directory is skipped rather than erroring.
"""

import os
import socket
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest

from graphrag_lab.core.config import get_settings
from graphrag_lab.graph.neo4j_store import Neo4jGraphStore


def _neo4j_reachable() -> bool:
    parsed = urlparse(os.getenv("NEO4J_URI", get_settings().neo4j_uri))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 7687), 1.0):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _neo4j_reachable():
        return
    skip = pytest.mark.skip(reason="Neo4j not reachable (run `make up`)")
    for item in items:
        if "tests/integration" in str(item.path):
            item.add_marker(skip)


@pytest.fixture
async def store() -> AsyncIterator[Neo4jGraphStore]:
    settings = get_settings()
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    yield graph_store
    await graph_store.aclose()


@pytest.fixture
def graph_store(store: Neo4jGraphStore) -> Neo4jGraphStore:
    return store


@pytest.fixture
async def tenant_id(store: Neo4jGraphStore) -> AsyncIterator[str]:
    """A fresh tenant per test, dropped (nodes and indexes) afterwards."""
    tid = f"test_{uuid.uuid4().hex[:16]}"
    yield tid
    await store.drop_graph(tid)
