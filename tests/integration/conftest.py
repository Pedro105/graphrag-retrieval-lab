"""Integration tests need a live FalkorDB (and most also real LLM/embedding
API keys). Skip the whole directory cleanly when FalkorDB isn't reachable:
the falkordb client connects eagerly on construction, so per-test probes would
otherwise error instead of skipping."""

import os
import socket

import pytest


def _falkordb_reachable() -> bool:
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _falkordb_reachable():
        return
    skip = pytest.mark.skip(reason="FalkorDB not reachable (run `make up`)")
    for item in items:
        if "tests/integration" in str(item.path):
            item.add_marker(skip)
