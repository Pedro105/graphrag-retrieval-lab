"""Per-question retrieval metrics and bootstrap confidence intervals.

- answer_recall: the normalized gold answer appears in the packed context.
  Only defined for bridge-type questions (compositional, inference), whose
  answer is reached through the chain; for comparison questions the answer
  is named in the question itself, and yes/no answers cannot be matched.
- support_recall: fraction of the question's gold units in the packed
  context -- evidence triples for graph retrievers, supporting sentences for
  text retrievers.
- full_support: every gold unit is in the context, i.e. the whole reasoning
  chain was retrieved.
"""

import re
import string
from collections.abc import Hashable, Sequence

import numpy as np

from graphrag_lab.bench.context import Unit, render
from graphrag_lab.bench.data import Question

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles,
    collapse whitespace."""
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_recall(question: Question, units: Sequence[Unit]) -> bool | None:
    if not question.is_bridge or normalize(question.answer) in ("yes", "no"):
        return None
    answer = normalize(question.answer)
    if not answer:
        return None
    # Pad with spaces so "1950" does not match inside "19500".
    return f" {answer} " in f" {normalize(render(units))} "


def gold_keys(question: Question, kind: str) -> set[Hashable]:
    if kind == "triple":
        return set(question.evidence)
    return set(question.supporting)


def support_recall(question: Question, units: Sequence[Unit], kind: str) -> float:
    gold = gold_keys(question, kind)
    if not gold:
        return 0.0
    retrieved = {unit.key for unit in units}
    return len(gold & retrieved) / len(gold)


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """(mean, low, high) with a percentile bootstrap over questions."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_resamples, data.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(data.mean()), float(low), float(high)
