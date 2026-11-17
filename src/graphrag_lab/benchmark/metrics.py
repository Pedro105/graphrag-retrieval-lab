"""Answer and retrieval metrics for multi-hop QA."""

import re
import string
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace. This is
    the SQuAD normalization, which 2WikiMultiHopQA and MuSiQue also use."""
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    predicted = normalize_answer(prediction)
    return float(any(predicted == normalize_answer(gold) for gold in gold_answers))


def _f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Token-level F1 against the best-matching gold answer."""
    return max((_f1(prediction, gold) for gold in gold_answers), default=0.0)


def answer_coverage(context: str, gold_answers: Sequence[str]) -> float:
    """1.0 if any gold answer appears in the retrieved context.

    Measures retrieval alone: if this is 0, no answer model could have
    answered from the context.
    """
    normalized = f" {normalize_answer(context)} "
    return float(
        any(f" {answer} " in normalized for answer in map(normalize_answer, gold_answers) if answer)
    )


def recall(retrieved: Iterable[Hashable], gold: Collection[Hashable]) -> float:
    """Fraction of the gold items (supporting facts, say) that were retrieved."""
    if not gold:
        raise ValueError("recall is undefined without gold items")
    return len(set(retrieved) & set(gold)) / len(set(gold))
