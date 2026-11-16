"""Embedding providers and the factory, with the model stubbed out (no
download, no network)."""

import numpy as np
import pytest

from graphrag_lab.core.config import Settings
from graphrag_lab.vector.embeddings import (
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"texts": texts, **kwargs})
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


def _provider_with_fake() -> tuple[LocalEmbeddingProvider, FakeModel]:
    provider = LocalEmbeddingProvider("fake-model")
    fake = FakeModel()
    provider._model = fake  # type: ignore[assignment]
    return provider, fake


async def test_local_embed_batch_preserves_order_and_normalizes() -> None:
    provider, fake = _provider_with_fake()

    vectors = await provider.embed_batch(["a", "abc"])

    assert vectors == [[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    assert fake.calls[0]["normalize_embeddings"] is True


async def test_local_embed_single_and_empty_batch() -> None:
    provider, fake = _provider_with_fake()

    assert await provider.embed("ab") == [2.0, 0.0, 1.0]
    assert await provider.embed_batch([]) == []
    assert len(fake.calls) == 1  # the empty batch never reaches the model


def test_local_dimension_comes_from_the_model() -> None:
    provider, _ = _provider_with_fake()
    assert provider.dimension == 3


def test_factory_defaults_to_local() -> None:
    provider = build_embedding_provider(Settings(_env_file=None))  # type: ignore[call-arg]
    assert isinstance(provider, LocalEmbeddingProvider)


def test_factory_openai_requires_a_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedding_provider(
            Settings(_env_file=None, embedding_provider="openai", openai_api_key=None)  # type: ignore[call-arg]
        )


def test_factory_openai_with_key() -> None:
    settings = Settings(_env_file=None, embedding_provider="openai", openai_api_key="sk-test")  # type: ignore[call-arg]
    provider = build_embedding_provider(settings)
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.dimension == settings.openai_embedding_dimension
