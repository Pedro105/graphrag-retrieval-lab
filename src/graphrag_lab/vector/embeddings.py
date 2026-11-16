"""Embedding providers. The graph store holds the vectors; this module only
produces them.

The default is a local sentence-transformers model: free, deterministic,
runs offline after the first download, and needs no API key. The OpenAI
provider remains available for comparison.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from graphrag_lab.core.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this provider returns."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single piece of text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call. Order matches input order."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers model run in-process on CPU (or GPU if present).

    The model loads lazily on first use; encoding runs in a worker thread so
    it doesn't block the event loop. Vectors are L2-normalized, so cosine
    similarity equals the dot product.
    """

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        dim = self._get_model().get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"model {self._model_name} does not report an embedding dimension")
        return int(dim)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    async def embed(self, text: str) -> list[float]:
        (embedding,) = await self.embed_batch([text])
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimension(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        (embedding,) = await self.embed_batch([text])
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the provider selected by `EMBEDDING_PROVIDER`."""
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings.local_embedding_model)
    if settings.openai_api_key is None:
        raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimension,
    )
