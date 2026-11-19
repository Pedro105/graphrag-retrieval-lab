"""Batch embedding for the benchmark, with an on-disk cache.

The API path embeds one string at a time through the async EmbeddingProvider;
the benchmark embeds tens of thousands of sentences, so it calls
sentence-transformers directly in large batches and caches the resulting
matrices as .npy files keyed by model and a hash of the exact texts.
"""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from graphrag_lab.vector.embeddings import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

Matrix = npt.NDArray[np.float32]

# bge-*-v1.5 retrieval convention: prefix short queries, not the documents.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Encoder:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        self._model: SentenceTransformer | None = None
        self._memo: dict[str, Matrix] = {}

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], show_progress: bool = False) -> Matrix:
        vectors = self.model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def documents(self, texts: list[str], name: str) -> Matrix:
        """Embed documents, reusing `<cache_dir>/<name>-<hash>.npy` when present."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        path = self._cache_path(texts, name)
        if path is not None and path.exists():
            cached: Matrix = np.load(path)
            if cached.shape[0] == len(texts):
                return cached
        matrix = self._encode(texts, show_progress=len(texts) > 5_000)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return matrix

    def queries(self, texts: list[str], instruction: bool = True) -> Matrix:
        prefix = self._query_instruction if instruction else ""
        missing = [t for t in dict.fromkeys(texts) if prefix + t not in self._memo]
        if missing:
            for text, vector in zip(missing, self._encode([prefix + t for t in missing])):
                self._memo[prefix + text] = vector
        return np.stack([self._memo[prefix + t] for t in texts])

    def _cache_path(self, texts: list[str], name: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps([self.model_name, texts]).encode()).hexdigest()[:16]
        return self._cache_dir / f"{name}-{digest}.npy"


class EncoderEmbeddingProvider(EmbeddingProvider):
    """Adapts an Encoder to the async EmbeddingProvider interface, so the
    system's own recall path can run inside the benchmark without loading
    the model twice. Queries are embedded without the retrieval instruction,
    matching how the API embeds them."""

    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder

    @property
    def dimension(self) -> int:
        dim = self._encoder.model.get_embedding_dimension()
        if dim is None:
            raise ValueError("encoder model does not report an embedding dimension")
        return int(dim)

    async def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._encoder.queries([text], instruction=False)[0]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._encoder.queries(texts, instruction=False)
        return [[float(x) for x in row] for row in matrix]


def top_k(scores: npt.NDArray[np.float32], k: int) -> list[int]:
    """Indices of the k highest scores, best first."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    candidates = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in candidates[np.argsort(-scores[candidates], kind="stable")]]
