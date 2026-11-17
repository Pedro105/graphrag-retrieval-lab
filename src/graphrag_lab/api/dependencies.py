"""The shared graph store and embedding provider, built once in the app's
lifespan and handed to routes."""

from typing import Annotated

from fastapi import Depends, Request

from graphrag_lab.graph.store import GraphStore
from graphrag_lab.vector.embeddings import EmbeddingProvider


def get_graph_store(request: Request) -> GraphStore:
    graph_store: GraphStore = request.app.state.graph_store
    return graph_store


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    embedding_provider: EmbeddingProvider = request.app.state.embedding_provider
    return embedding_provider


GraphStoreDep = Annotated[GraphStore, Depends(get_graph_store)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
