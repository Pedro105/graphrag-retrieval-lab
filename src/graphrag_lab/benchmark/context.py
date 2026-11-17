"""The context a retriever hands to the answer model, and its token budget."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Words plus punctuation marks.

    Not any model's real tokenizer. It only has to treat every retriever the
    same way, since budgets are compared between retrievers, never against a
    model's context window.
    """
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    claim_id: UUID

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ')} {self.object}."


@dataclass(frozen=True)
class RetrievedContext:
    triples: tuple[Triple, ...]
    tokens: int

    @property
    def text(self) -> str:
        return "\n".join(triple.text for triple in self.triples)


def pack(ranked: Iterable[Triple], token_budget: int) -> RetrievedContext:
    """Take triples in rank order until the next one would exceed the budget.

    Stopping at the first triple that does not fit, rather than skipping it
    and trying shorter ones, keeps the context a prefix of the ranking. A
    retriever is then scored on its ranking alone.
    """
    kept: list[Triple] = []
    used = 0
    for triple in ranked:
        cost = count_tokens(triple.text)
        if used + cost > token_budget:
            break
        kept.append(triple)
        used += cost
    return RetrievedContext(tuple(kept), used)
