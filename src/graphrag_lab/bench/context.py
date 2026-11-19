"""Retrieval units and token-budgeted context packing.

Every retriever returns a ranked list of `Unit`s: sentences for text
retrievers, triples for graph retrievers. A run then packs the ranked list
into a fixed token budget, so methods are compared on equal context size
rather than on an equal number of hits of very different lengths.
"""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

UnitKind = Literal["sentence", "triple"]


@dataclass(frozen=True)
class Unit:
    kind: UnitKind
    # Identity used for metrics: (title, sentence_index) or a Triple.
    key: Hashable
    text: str
    # Shared heading charged once per packed context (a sentence's paragraph
    # title); None for self-contained units such as triples.
    header: str | None = None


TokenCounter = Callable[[str], int]


class HFTokenCounter:
    """Counts tokens with a Hugging Face tokenizer, memoized per text."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._count = lru_cache(maxsize=1_000_000)(self._uncached)

    def _uncached(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def __call__(self, text: str) -> int:
        return int(self._count(text))


def whitespace_token_counter(text: str) -> int:
    """Dependency-free counter for tests."""
    return len(text.split())


def pack(units: Sequence[Unit], budget: int, count: TokenCounter) -> list[Unit]:
    """The longest prefix of `units` whose total cost fits in `budget` tokens.

    A unit costs its own tokens plus its header's tokens the first time that
    header appears. Packing stops at the first unit that does not fit, so the
    result is always a prefix of the retriever's ranking.
    """
    packed: list[Unit] = []
    used = 0
    seen_headers: set[str] = set()
    for unit in units:
        cost = count(unit.text)
        if unit.header is not None and unit.header not in seen_headers:
            cost += count(unit.header)
        if used + cost > budget:
            break
        packed.append(unit)
        used += cost
        if unit.header is not None:
            seen_headers.add(unit.header)
    return packed


def context_tokens(units: Sequence[Unit], count: TokenCounter) -> int:
    headers = {unit.header for unit in units if unit.header is not None}
    return sum(count(unit.text) for unit in units) + sum(count(h) for h in headers)


def render(units: Sequence[Unit]) -> str:
    """Context as an LLM would see it: sentences grouped under their title."""
    lines: list[str] = []
    current_header: str | None = None
    for unit in units:
        if unit.header is not None and unit.header != current_header:
            lines.append(f"## {unit.header}")
            current_header = unit.header
        lines.append(unit.text)
    return "\n".join(lines)
