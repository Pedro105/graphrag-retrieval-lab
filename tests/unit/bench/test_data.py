from graphrag_lab.bench.data import (
    Triple,
    build_corpus,
    parse_row,
    related_ids,
    sample_rows,
)
from tests.unit.bench.helpers import FILM_ROW, make_row


def test_parse_row_resolves_supporting_facts_to_sentence_text() -> None:
    question, paragraphs = parse_row(FILM_ROW)

    assert question.evidence[0] == Triple("Polish-Russian War", "director", "Xawery Żuławski")
    assert question.topic_entity == "Polish-Russian War"
    assert question.supporting == (
        ("Polish-Russian War (film)", "It was directed by Xawery Żuławski."),
        ("Xawery Żuławski", "His mother is Małgorzata Braunek."),
    )
    assert [p.title for p in paragraphs][-1] == "Snow White"
    assert question.is_bridge


def test_parse_row_drops_out_of_range_supporting_facts() -> None:
    row = make_row(
        "q", "?", "a", [["s", "p", "o"]], {"T": ["only one"]}, [("T", 5), ("Missing", 0)]
    )
    question, _ = parse_row(row)
    assert question.supporting == ()


def test_sample_is_seeded_and_skips_unlabelled_questions() -> None:
    rows = [make_row(f"q{i}", "?", "a", [["s", "p", f"o{i}"]], {}, []) for i in range(20)]
    rows.append(make_row("unlabelled", "?", "a", [], {}, []))

    first = sample_rows(rows, 10, seed=7)
    assert [r["id"] for r in first] == [r["id"] for r in sample_rows(rows, 10, seed=7)]
    assert "unlabelled" not in {r["id"] for r in sample_rows(rows, 21, seed=7)}


def test_build_corpus_pools_paragraphs_and_triples_and_splits_eval() -> None:
    other = make_row(
        "q2",
        "?",
        "1950",
        [["Xawery Żuławski", "date of birth", "1950"]],
        {"Xawery Żuławski": ["A different split of the same title."]},
        [],
    )
    corpus = build_corpus([FILM_ROW, other], n_eval=1)

    assert [q.id for q in corpus.eval_questions] == ["q1"]
    assert len(corpus.triples) == 3
    # First occurrence of a title wins.
    assert corpus.paragraphs["Xawery Żuławski"].sentences[1] == "His mother is Małgorzata Braunek."
    assert "1950" in corpus.entities
    assert (
        corpus.fingerprint and corpus.fingerprint == build_corpus([FILM_ROW, other], 1).fingerprint
    )


def test_background_adds_triples_and_only_supporting_paragraphs() -> None:
    background = make_row(
        "b1",
        "?",
        "x",
        [["Xawery Żuławski", "spouse", "Maria Strzelecka"]],
        {"Xawery Żuławski (bio)": ["He married Maria Strzelecka."], "Noise": ["Unrelated."]},
        [("Xawery Żuławski (bio)", 0)],
    )
    corpus = build_corpus([FILM_ROW], n_eval=1, background=[background])

    assert Triple("Xawery Żuławski", "spouse", "Maria Strzelecka") in corpus.triples
    assert "Xawery Żuławski (bio)" in corpus.paragraphs
    assert "Noise" not in corpus.paragraphs
    assert corpus.background_questions == 1
    assert [q.id for q in corpus.questions] == ["q1"]


def test_related_ids_link_on_specific_entities_not_hubs_or_dates() -> None:
    sample = [
        {
            "id": "s",
            "evidences": [["Film A", "director", "Jane Doe"], ["Jane Doe", "country", "American"]],
        }
    ]
    background = [
        {"id": "shares_director", "evidences": [["Film B", "director", "Jane Doe"]]},
        {"id": "shares_hub", "evidences": [["Film C", "country", "American"]]},
        {"id": "shares_date", "evidences": [["Film A2", "publication date", "1999"]]},
        {"id": "unrelated", "evidences": [["Film D", "director", "John Roe"]]},
    ]
    # Make "American" a hub: more triples than hub_max.
    background += [
        {"id": f"h{i}", "evidences": [[f"P{i}", "citizenship", "American"]]} for i in range(5)
    ]
    sample[0]["evidences"].append(["Film A", "publication date", "1999"])

    assert related_ids(sample, background, hub_max=3) == {"shares_director"}
