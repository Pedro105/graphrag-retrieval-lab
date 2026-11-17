"""Personalized PageRank by power iteration, on an undirected graph."""

from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import TypeVar

Node = TypeVar("Node", bound=Hashable)


def personalized_pagerank(
    edges: Iterable[tuple[Node, Node]],
    seeds: Iterable[Node],
    damping: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> dict[Node, float]:
    """Stationary distribution of a random walk that restarts at the seeds.

    At each step the walker follows a random edge with probability `damping`
    and otherwise jumps back to a seed chosen uniformly. 0.5 is the value
    HippoRAG uses; it keeps the mass close to the seeds, which suits
    questions that are a few hops deep. Parallel edges count separately, so
    two entities linked by several facts are more strongly tied.

    Scores sum to 1. Nodes the seeds cannot reach score 0.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")

    neighbours: dict[Node, list[Node]] = defaultdict(list)
    for a, b in edges:
        if a == b:
            continue
        neighbours[a].append(b)
        neighbours[b].append(a)

    seed_list = list(dict.fromkeys(seeds))
    if not seed_list:
        return {}
    restart = {seed: 1.0 / len(seed_list) for seed in seed_list}

    scores = dict(restart)
    for _ in range(max_iterations):
        following = 0.0
        updated: dict[Node, float] = defaultdict(float)
        for node, score in scores.items():
            targets = neighbours.get(node)
            if not targets:
                continue  # an isolated seed: its mass goes back to the seeds
            following += score
            share = damping * score / len(targets)
            for target in targets:
                updated[target] += share
        # Whatever did not move along an edge restarts.
        restarting = 1.0 - damping * following
        for seed, weight in restart.items():
            updated[seed] += restarting * weight

        change = sum(abs(updated[node] - scores.get(node, 0.0)) for node in updated)
        scores = dict(updated)
        if change < tolerance:
            break
    return scores
