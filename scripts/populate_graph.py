"""Populate the `eval_test` graph with three unrelated clusters for the smoke eval.

    uv run python scripts/populate_graph.py [--base-url http://localhost:8000]

Each (content, source) pair is one POST /v1/memories, so a relation can only
form between entities mentioned in the same pair.

Each cluster hides a two-hop chain whose ends are never mentioned together,
so the far end is reachable only through the middle entity:

  A: Dr. Sarah Chen -> CRISPR-X stability study -> RG-88
  B: Dr. Amara Osei -> Cascade v3 architecture whitepaper -> AuroraML
  C: Korrigan Cells Ltd -> LFP-9 Cell -> Voluntary Safety Recall R-2025-014

The relation names on those edges are chosen by the extractor and vary.

Cluster A also has a property stated differently by two sources: the Series A
round is "$42 million" according to agent_finance and "$45 million" according
to agent_pr.
"""

import argparse
import sys
from typing import Any

import httpx

TENANT_ID = "eval_test"

# (content, source)
CLUSTER_A_FACTS: list[tuple[str, str]] = [
    (
        "Dr. Sarah Chen and Dr. Marcus Webb co-founded Helios Therapeutics, "
        "a biotech startup developing treatments for pancreatic cancer.",
        "agent_research",
    ),
    (
        "Dr. Sarah Chen earned her PhD at Stanford University before "
        "co-founding Helios Therapeutics.",
        "agent_research",
    ),
    (
        "Dr. Marcus Webb completed his doctoral studies at the "
        "Massachusetts Institute of Technology.",
        "agent_research",
    ),
    (
        # First half of chain A; RG-88 and RiboGen are not mentioned. The
        # title is unquoted: with quotes, the model returned malformed tool
        # calls every time once the graph held many relation types.
        "Dr. Sarah Chen authored the paper CRISPR-X stability study while at Stanford University.",
        "agent_research",
    ),
    (
        "Helios Therapeutics' lead pipeline compound, HLX-203, is being "
        "developed specifically to target pancreatic cancer.",
        "agent_research",
    ),
    (
        # One side of the funding conflict (agent_pr states the other below).
        # The extractor picks property names itself. This phrasing gave the
        # same `amount_raised` key in 7 of 8 runs; if a re-run lands on a
        # different key, update the provenance case in eval/dataset.yaml.
        "Sequoia Capital and Atlas Venture both invested in Helios "
        "Therapeutics' Series A funding round; the amount raised was "
        "$42 million.",
        "agent_finance",
    ),
    (
        "Helios Therapeutics' Series A funding round amount raised was actually $45 million.",
        "agent_pr",
    ),
    (
        "HLX-203 received an FDA Orphan Drug Designation, recognizing its "
        "potential to treat a rare form of pancreatic cancer.",
        "agent_regulatory",
    ),
    (
        "RiboGen Inc, a rival biotech lab, is also targeting pancreatic "
        "cancer with its own compound, RG-88, putting it in direct "
        "competition with Helios Therapeutics' HLX-203.",
        "agent_competitive_intel",
    ),
    (
        "Dr. Elena Petrov leads the research team at RiboGen Inc that is developing RG-88.",
        "agent_competitive_intel",
    ),
    (
        # Second half of chain A; Sarah Chen is not mentioned. RG-88 is the
        # subject on purpose: phrased around "RiboGen Inc's scientists", 3 of
        # 8 runs attached the paper to RiboGen Inc and made the chain 3 hops.
        "RG-88's development was directly informed by findings published "
        "in CRISPR-X stability study.",
        "agent_competitive_intel",
    ),
    (
        "The Pancreatic Cancer Action Network announced a patient advocacy "
        "partnership with Helios Therapeutics.",
        "agent_ops",
    ),
]

CLUSTER_B_FACTS: list[tuple[str, str]] = [
    (
        "Dr. Amara Osei created and maintains the open source machine learning framework Cascade.",
        "agent_oss",
    ),
    ("Liang Wu is a core contributor to Cascade.", "agent_oss"),
    ("Dr. Amara Osei works at Vertex AI Labs.", "agent_oss"),
    ("Liang Wu works at Nordic Compute.", "agent_oss"),
    (
        "Dr. Amara Osei studied at UC Berkeley AI Research Lab before joining Vertex AI Labs.",
        "agent_oss",
    ),
    ("Liang Wu studied at Helsinki Polytechnic.", "agent_oss"),
    (
        # First half of chain B; AuroraML is not mentioned. Unquoted title
        # and a single clause, for the reason given on the CRISPR-X fact.
        "Dr. Amara Osei authored the Cascade v3 architecture whitepaper.",
        "agent_oss",
    ),
    (
        # Second half of chain B; Osei is not mentioned. With the whitepaper
        # as the subject the same malformed tool calls showed up now and then.
        "Priya Nair maintains AuroraML, a downstream library influenced by "
        "the Cascade v3 architecture whitepaper.",
        "agent_downstream",
    ),
    (
        "AuroraML depends on Cascade as its core computational backend.",
        "agent_downstream",
    ),
    (
        "Cascade is regularly evaluated on the MLPerf Training Benchmark.",
        "agent_oss",
    ),
    ("NeoSilicon sponsors the Cascade project.", "agent_vendor"),
    (
        "NeoSilicon's Quantix GPU hardware is optimized specifically for "
        "running Cascade workloads.",
        "agent_vendor",
    ),
    ("The Cascade Foundation governs the Cascade open source project.", "agent_oss"),
    (
        "Cascade v4.0 was released to patch a critical vulnerability tracked as CVE-2025-41122.",
        "agent_security",
    ),
]

CLUSTER_C_FACTS: list[tuple[str, str]] = [
    (
        "Vanta Battery Co is a tier-1 supplier that manufactures the Aegis "
        "Battery Pack for Solstice Motors.",
        "agent_procurement",
    ),
    (
        # First half of chain C; the recall is not mentioned.
        "Korrigan Cells Ltd is a tier-2 supplier that produces the LFP-9 Cell.",
        "agent_procurement",
    ),
    (
        "The Aegis Battery Pack is built using LFP-9 Cells as its core component.",
        "agent_procurement",
    ),
    ("The Aegis Battery Pack holds ISO 26262 safety certification.", "agent_quality"),
    (
        "Solstice Motors assembles the Solstice EV-1 using the Aegis Battery Pack.",
        "agent_ops",
    ),
    (
        "Solstice Motors also produces the Solstice EV-2, a separate model "
        "that does not use the Aegis Battery Pack.",
        "agent_ops",
    ),
    (
        "Brightline Retail Group is the exclusive retail partner selling the Solstice EV-1.",
        "agent_retail",
    ),
    ("Meridian Auto Retailers sells the Solstice EV-2.", "agent_retail"),
    (
        "Harbor Logistics handles shipping for Solstice Motors' vehicles.",
        "agent_logistics",
    ),
    (
        "Port of Rotterdam congestion caused significant delays for Harbor Logistics shipments.",
        "agent_logistics",
    ),
    (
        "Vanta Battery Co experienced a Q3 2025 Cell Shortage that affected its production output.",
        "agent_quality",
    ),
    ("Raj Patel is the quality lead at Vanta Battery Co.", "agent_quality"),
    ("Tessa Lindqvist is the VP of Supply Chain at Solstice Motors.", "agent_ops"),
    (
        "Solstice Motors issued Voluntary Safety Recall R-2025-014 for the Solstice EV-1.",
        "agent_quality",
    ),
    (
        # Second half of chain C; Korrigan is not mentioned. A longer version
        # that also named a "battery defect" put the defect between the cell
        # and the recall, making the chain three hops.
        "The LFP-9 Cell was identified as the root cause of Solstice "
        "Motors' Voluntary Safety Recall R-2025-014.",
        "agent_quality",
    ),
]

CLUSTERS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Cluster A (Helios Therapeutics)", CLUSTER_A_FACTS),
    ("Cluster B (Cascade)", CLUSTER_B_FACTS),
    ("Cluster C (Solstice Motors)", CLUSTER_C_FACTS),
]


_MAX_ATTEMPTS = 5


def _post_memory(client: httpx.Client, content: str, source: str) -> dict[str, Any]:
    """POST one fact. Extraction occasionally fails validation when the model
    emits two tool calls instead of one; the same text succeeds on retry."""
    last_exc: httpx.HTTPStatusError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        response = client.post(
            "/v1/memories",
            json={"content": content, "source": source, "tenant_id": TENANT_ID},
        )
        try:
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            print(f"  attempt {attempt}/{_MAX_ATTEMPTS} failed ({exc}), retrying...")
    assert last_exc is not None
    raise last_exc


def ingest(client: httpx.Client) -> None:
    for cluster_name, facts in CLUSTERS:
        print(f"\n=== {cluster_name}: {len(facts)} facts ===")
        for content, source in facts:
            memory = _post_memory(client, content, source)
            entity_names = [e["name"] for e in memory["entities"]]
            relation_count = len(memory["relations"])
            print(
                f"  [{source}] {len(entity_names)} entities, {relation_count} relations: {entity_names}"
            )


def print_graph_summary(client: httpx.Client) -> None:
    response = client.get("/v1/graph", params={"tenant_id": TENANT_ID})
    response.raise_for_status()
    graph = response.json()
    entities = graph["entities"]
    relations = graph["relations"]

    connected_ids = {r["source_entity_id"] for r in relations} | {
        r["target_entity_id"] for r in relations
    }
    isolated = [e for e in entities if e["id"] not in connected_ids]

    print("\n=== Graph summary for tenant eval_test ===")
    print(f"total nodes: {len(entities)}")
    print(f"total edges: {len(relations)}")
    print(f"isolated nodes: {len(isolated)} ({len(isolated) / max(len(entities), 1):.1%})")
    if isolated:
        print("  isolated entity names:", [e["name"] for e in isolated])
    if entities and len(isolated) / len(entities) > 0.10:
        print(
            "  WARNING: isolated node rate exceeds 10% -- investigate "
            "extraction before writing eval cases.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate the eval_test tenant graph.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        ingest(client)
        print_graph_summary(client)

    return 0


if __name__ == "__main__":
    sys.exit(main())
