"""Benchmark run configuration (loaded from YAML under configs/bench/)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

RETRIEVERS = (
    "dense_paragraph",
    "bm25_paragraph",
    "hybrid_paragraph",
    "dense_sentence",
    "dense_triple",
    "graph_bfs_d1",
    "graph_bfs_d2",
    "graph_ppr",
    "oracle_sentence",
    "oracle_triple",
)


class BenchConfig(BaseModel):
    name: str
    split: str = "validation"
    # Questions pooled into the shared corpus, and how many of them to evaluate.
    n_corpus: int = Field(default=3000, ge=1)
    n_eval: int = Field(default=1000, ge=1)
    seed: int = 0
    # Split providing background questions related to the sample (see
    # bench/data.py); None for a corpus of the sample alone.
    background_split: str | None = "train"
    # Entities in more evidence triples than this are hubs and never link
    # questions (e.g. "American" appears in ~35k).
    hub_max: int = 20
    budgets: list[int] = Field(default_factory=lambda: [64, 128, 256, 512, 1024])
    retrievers: list[str] = Field(default_factory=lambda: list(RETRIEVERS))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    n_seeds: int = 5
    bootstrap_resamples: int = 1000

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchConfig":
        config = cls.model_validate(yaml.safe_load(path.read_text()))
        unknown = set(config.retrievers) - set(RETRIEVERS)
        if unknown:
            raise ValueError(f"unknown retrievers: {sorted(unknown)}")
        return config
