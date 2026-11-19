import pytest

from graphrag_lab.bench.context import Unit, context_tokens, pack, render
from graphrag_lab.bench.context import whitespace_token_counter as count
from graphrag_lab.bench.data import Triple, parse_row
from graphrag_lab.bench.metrics import answer_recall, bootstrap_ci, normalize, support_recall
from tests.unit.bench.helpers import FILM_ROW, make_row

QUESTION, _ = parse_row(FILM_ROW)


def sentence(title: str, text: str) -> Unit:
    return Unit("sentence", (title, text), text, header=title)


def test_pack_charges_each_header_once_and_stops_at_first_overflow() -> None:
    units = [
        sentence("Two Words", "one two three"),  # 3 + header 2 = 5
        sentence("Two Words", "four five"),  # 2 (header already paid) -> 7
        sentence("Other", "six seven eight nine"),  # 4 + 1 = 5 -> 12, over 10
        sentence("Two Words", "x"),  # would fit, but packing keeps a prefix
    ]
    packed = pack(units, budget=10, count=count)

    assert [u.text for u in packed] == ["one two three", "four five"]
    assert context_tokens(packed, count) == 7


def test_pack_empty_when_first_unit_exceeds_budget() -> None:
    assert pack([sentence("T", "a b c d e")], budget=3, count=count) == []


def test_render_groups_sentences_under_titles() -> None:
    text = render([sentence("A", "s1"), sentence("A", "s2"), Unit("triple", 1, "x | p | y")])
    assert text == "## A\ns1\ns2\nx | p | y"


def test_normalize_follows_squad_conventions() -> None:
    assert normalize("The  Małgorzata, Braunek!") == "małgorzata braunek"


def test_answer_recall_matches_whole_tokens_only() -> None:
    hit = [Unit("triple", t, t.text) for t in QUESTION.evidence]
    assert answer_recall(QUESTION, hit) is True
    assert answer_recall(QUESTION, hit[:1]) is False

    dated = make_row("d", "When?", "1950", [["a", "b", "1950"]], {}, [])
    question, _ = parse_row(dated)
    assert answer_recall(question, [Unit("triple", 1, "born 19500")]) is False
    assert answer_recall(question, [Unit("triple", 1, "born 1950.")]) is True


@pytest.mark.parametrize("qtype,answer", [("comparison", "Film A"), ("compositional", "yes")])
def test_answer_recall_undefined_for_comparison_and_yes_no(qtype: str, answer: str) -> None:
    question, _ = parse_row(make_row("c", "?", answer, [["a", "b", "c"]], {}, [], qtype=qtype))
    assert answer_recall(question, [Unit("triple", 1, answer)]) is None


def test_support_recall_uses_triples_or_sentences_by_unit_kind() -> None:
    first_triple = Unit("triple", QUESTION.evidence[0], QUESTION.evidence[0].text)
    assert support_recall(QUESTION, [first_triple], "triple") == 0.5
    title, text = QUESTION.supporting[0]
    assert support_recall(QUESTION, [sentence(title, text)], "sentence") == 0.5
    assert support_recall(QUESTION, [Unit("triple", Triple("x", "y", "z"), "")], "triple") == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded() -> None:
    values = [1.0] * 30 + [0.0] * 70
    mean, low, high = bootstrap_ci(values, n_resamples=500, seed=1)
    assert mean == pytest.approx(0.3)
    assert low < mean < high
    assert (mean, low, high) == bootstrap_ci(values, n_resamples=500, seed=1)
    assert bootstrap_ci([1.0, 1.0]) == (1.0, 1.0, 1.0)
