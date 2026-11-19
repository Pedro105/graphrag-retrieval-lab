"""Tiny 2WikiMultiHopQA-shaped rows for benchmark tests."""

from typing import Any


def make_row(
    qid: str,
    question: str,
    answer: str,
    evidences: list[list[str]],
    paragraphs: dict[str, list[str]],
    supporting: list[tuple[str, int]],
    qtype: str = "compositional",
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "evidences": evidences,
        "supporting_facts": {
            "title": [title for title, _ in supporting],
            "sent_id": [index for _, index in supporting],
        },
        "context": {"title": list(paragraphs), "sentences": list(paragraphs.values())},
    }


FILM_ROW = make_row(
    "q1",
    "Who is the mother of the director of film Polish-Russian War?",
    "Małgorzata Braunek",
    [
        ["Polish-Russian War", "director", "Xawery Żuławski"],
        ["Xawery Żuławski", "mother", "Małgorzata Braunek"],
    ],
    {
        "Polish-Russian War (film)": [
            "Polish-Russian War is a 2009 Polish film.",
            "It was directed by Xawery Żuławski.",
        ],
        "Xawery Żuławski": [
            "Xawery Żuławski is a Polish film director.",
            "His mother is Małgorzata Braunek.",
        ],
        "Snow White": ["An unrelated distractor paragraph."],
    },
    [("Polish-Russian War (film)", 1), ("Xawery Żuławski", 1)],
)
