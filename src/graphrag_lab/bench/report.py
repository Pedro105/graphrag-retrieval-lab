"""Aggregate per-question records into summary metrics and markdown tables."""

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graphrag_lab.bench.metrics import bootstrap_ci

Record = dict[str, Any]

METRICS = {
    "answer_recall": "Answer recall (bridge questions)",
    "full_support": "Full reasoning chain retrieved (all questions)",
    "support_recall": "Gold units retrieved (all questions)",
}


def write_records(records: Iterable[Record], path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[Record]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def summarize(records: list[Record], resamples: int = 1000) -> list[Record]:
    """One row per (retriever, budget): each metric's mean and 95% CI, over
    the questions where the metric is defined."""
    grouped: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for record in records:
        grouped[(record["retriever"], record["budget"])].append(record)
    rows: list[Record] = []
    for (retriever, budget), group in sorted(grouped.items()):
        row: Record = {
            "retriever": retriever,
            "unit_kind": group[0]["unit_kind"],
            "budget": budget,
            "mean_tokens": sum(r["tokens"] for r in group) / len(group),
        }
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r[metric] is not None]
            mean, low, high = bootstrap_ci(values, n_resamples=resamples)
            row[metric] = {"mean": mean, "low": low, "high": high, "n": len(values)}
        seed_hits = [float(r["seed_hit"]) for r in group if r.get("seed_hit") is not None]
        if seed_hits:
            row["seed_hit"] = sum(seed_hits) / len(seed_hits)
        rows.append(row)
    return rows


def _cell(stat: Record) -> str:
    return f"{100 * stat['mean']:.1f} <sub>{100 * stat['low']:.1f}–{100 * stat['high']:.1f}</sub>"


def markdown_tables(summary: list[Record], retriever_order: list[str]) -> str:
    budgets = sorted({row["budget"] for row in summary})
    by_key = {(row["retriever"], row["budget"]): row for row in summary}
    retrievers = [r for r in retriever_order if any(row["retriever"] == r for row in summary)]
    lines: list[str] = []
    for metric, title in METRICS.items():
        n = next(row[metric]["n"] for row in summary)
        lines += [f"### {title} — n={n}, % with 95% CI", ""]
        lines.append("| Retriever | Units | " + " | ".join(f"{b} tok" for b in budgets) + " |")
        lines.append("|---|---|" + "---|" * len(budgets))
        for retriever in retrievers:
            kind = by_key[(retriever, budgets[0])]["unit_kind"]
            cells = [_cell(by_key[(retriever, b)][metric]) for b in budgets]
            lines.append(f"| `{retriever}` | {kind} | " + " | ".join(cells) + " |")
        lines.append("")
    seeded = [r for r in retrievers if "seed_hit" in by_key[(r, budgets[0])]]
    if seeded:
        lines += ["### Entity linking: topic entity among the seeds", ""]
        lines += ["| Retriever | Seed hit |", "|---|---|"]
        for retriever in seeded:
            lines.append(
                f"| `{retriever}` | {100 * by_key[(retriever, budgets[0])]['seed_hit']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def type_table(records: list[Record], budget: int, retriever_order: list[str]) -> str:
    """Full-chain retrieval by question type at one budget."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    types: set[str] = set()
    for record in records:
        if record["budget"] == budget:
            cells[(record["retriever"], record["type"])].append(float(record["full_support"]))
            types.add(record["type"])
    columns = sorted(types)
    retrievers = [r for r in retriever_order if any((r, t) in cells for t in columns)]
    counts = {t: len(cells[(retrievers[0], t)]) for t in columns}
    lines = [
        f"### Full reasoning chain retrieved by question type — {budget} tokens, %",
        "",
        "| Retriever | " + " | ".join(f"{t} (n={counts[t]})" for t in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for retriever in retrievers:
        values = [cells[(retriever, t)] for t in columns]
        lines.append(
            f"| `{retriever}` | "
            + " | ".join(f"{100 * sum(v) / len(v):.1f}" if v else "–" for v in values)
            + " |"
        )
    return "\n".join(lines) + "\n"


# Reference categorical palette (light mode), assigned in fixed order. Three
# slots are below 3:1 contrast on white, so every line also gets a direct
# end label and its own marker shape.
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARKERS = ["o", "s", "D", "^", "v", "P"]
_SURFACE, _TEXT, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

FIGURE_SERIES = {
    "graph_ppr": "Graph: personalized PageRank",
    "graph_bfs_d2": "Graph: BFS depth 2 (current system)",
    "dense_triple": "Facts: dense search",
    "dense_sentence": "Text: dense sentences",
    "hybrid_paragraph": "Text: hybrid paragraphs",
    "dense_paragraph": "Text: dense paragraphs",
}


def plot_budget_curves(
    summary: list[Record], path: Path, metric: str = "answer_recall", title: str = ""
) -> None:
    """Metric vs token budget, one line per retriever, oracle as a reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_retriever: dict[str, list[Record]] = defaultdict(list)
    for row in summary:
        by_retriever[row["retriever"]].append(row)
    budgets = sorted({row["budget"] for row in summary})

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    oracle = "oracle_triple" if metric == "answer_recall" else None
    if oracle and oracle in by_retriever:
        ys = [
            100 * r[metric]["mean"] for r in sorted(by_retriever[oracle], key=lambda r: r["budget"])
        ]
        ax.plot(budgets, ys, color=_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    ends: list[tuple[float, str, str]] = []
    for index, (name, label) in enumerate(FIGURE_SERIES.items()):
        if name not in by_retriever:
            continue
        rows = sorted(by_retriever[name], key=lambda r: r["budget"])
        color = _SERIES_COLORS[index]
        ys = [100 * r[metric]["mean"] for r in rows]
        lows = [100 * r[metric]["low"] for r in rows]
        highs = [100 * r[metric]["high"] for r in rows]
        ax.fill_between(budgets, lows, highs, color=color, alpha=0.08, linewidth=0, zorder=2)
        ax.plot(
            budgets,
            ys,
            color=color,
            linewidth=2,
            marker=_MARKERS[index],
            markersize=6,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.5,
            label=label,
            zorder=3,
        )
        ends.append((ys[-1], label, color))

    # Direct labels at the right edge, nudged apart so they never overlap.
    min_gap = 4.2
    placed: list[float] = []
    for y, label, color in sorted(ends, reverse=True):
        y_label = min(y, placed[-1] - min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            label,
            xy=(budgets[-1], y),
            xytext=(budgets[-1] * 1.12, y_label),
            color=_TEXT,
            fontsize=8.5,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 1,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )
    if oracle and oracle in by_retriever:
        ax.annotate("Gold facts (upper bound)", xy=(budgets[0], 101.5), color=_MUTED, fontsize=8.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlim(budgets[0] * 0.85, budgets[-1] * 1.08)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Context budget (tokens)", color=_MUTED, fontsize=9)
    ax.set_ylabel("Answer in retrieved context (%)", color=_MUTED, fontsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=_TEXT,
    )
    if title:
        ax.set_title(title, loc="left", color=_TEXT, fontsize=11, pad=14)
    fig.subplots_adjust(right=0.74, bottom=0.25)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
