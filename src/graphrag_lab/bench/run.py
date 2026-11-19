"""Benchmark runner.

    uv run python -m graphrag_lab.bench run --config configs/bench/2wiki-dev.yaml
    uv run python -m graphrag_lab.bench report results/2wiki-dev

`run` downloads the pinned dataset, builds the pooled corpus, embeds it
(cached under .cache/bench), loads the gold graph into Neo4j, ranks every
evaluation question with every retriever, packs each ranking into every
token budget, and writes:

    results/<name>/per_question.jsonl.gz   one record per question x retriever x budget
    results/<name>/summary.json            metrics with bootstrap 95% CIs
    results/<name>/results.md              the tables
    results/<name>/manifest.json           config, corpus stats, versions
"""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import structlog

from graphrag_lab.bench.config import RETRIEVERS, BenchConfig
from graphrag_lab.bench.context import HFTokenCounter, TokenCounter, context_tokens, pack
from graphrag_lab.bench.data import (
    Corpus,
    build_corpus,
    download_two_wiki,
    load_rows,
    load_rows_by_id,
    related_ids,
    sample_rows,
)
from graphrag_lab.bench.encoding import Encoder
from graphrag_lab.bench.graph import GraphBFSRetriever, GraphPPRRetriever, load_gold_graph
from graphrag_lab.bench.metrics import answer_recall, support_recall
from graphrag_lab.bench.report import markdown_tables, read_records, summarize, write_records
from graphrag_lab.bench.retrievers import (
    BM25ParagraphRetriever,
    DenseParagraphRetriever,
    DenseSentenceRetriever,
    DenseTripleRetriever,
    HybridParagraphRetriever,
    OracleSentenceRetriever,
    OracleTripleRetriever,
    Retriever,
)
from graphrag_lab.core.config import get_settings
from graphrag_lab.graph.neo4j_store import Neo4jGraphStore
from graphrag_lab.graph.store import GraphStore

logger = structlog.get_logger()

CACHE_DIR = Path(".cache/bench")


async def build_retrievers(
    names: list[str],
    corpus: Corpus,
    encoder: Encoder,
    store: GraphStore | None,
    config: BenchConfig,
) -> list[Retriever]:
    built: dict[str, Retriever] = {}
    tenant_id = f"bench_{corpus.fingerprint}"
    if any(name.startswith("graph_") for name in names):
        if store is None:
            raise ValueError("graph retrievers need a graph store")
        await load_gold_graph(store, corpus, encoder, tenant_id)

    for name in names:
        if name == "dense_paragraph":
            built[name] = DenseParagraphRetriever(corpus, encoder)
        elif name == "bm25_paragraph":
            built[name] = BM25ParagraphRetriever(corpus)
        elif name == "hybrid_paragraph":
            dense = built.get("dense_paragraph") or DenseParagraphRetriever(corpus, encoder)
            bm25 = built.get("bm25_paragraph") or BM25ParagraphRetriever(corpus)
            assert isinstance(dense, DenseParagraphRetriever)
            assert isinstance(bm25, BM25ParagraphRetriever)
            built[name] = HybridParagraphRetriever(dense, bm25, corpus)
        elif name == "dense_sentence":
            built[name] = DenseSentenceRetriever(corpus, encoder)
        elif name == "dense_triple":
            built[name] = DenseTripleRetriever(corpus, encoder)
        elif name in ("graph_bfs_d1", "graph_bfs_d2"):
            assert store is not None
            built[name] = GraphBFSRetriever(
                store, tenant_id, encoder, depth=int(name[-1]), n_seeds=config.n_seeds
            )
        elif name == "graph_ppr":
            assert store is not None
            built[name] = await GraphPPRRetriever(
                store, tenant_id, encoder, n_seeds=config.n_seeds
            ).build()
        elif name == "oracle_sentence":
            built[name] = OracleSentenceRetriever()
        elif name == "oracle_triple":
            built[name] = OracleTripleRetriever()
        else:
            raise ValueError(f"unknown retriever {name!r}")
    return [built[name] for name in names]


async def evaluate(
    retrievers: list[Retriever], corpus: Corpus, budgets: list[int], count: TokenCounter
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for retriever in retrievers:
        started = time.perf_counter()
        for question in corpus.eval_questions:
            ranking = await retriever.rank(question)
            for budget in budgets:
                packed = pack(ranking.units, budget, count)
                hit = answer_recall(question, packed)
                recall = support_recall(question, packed, retriever.unit_kind)
                records.append(
                    {
                        "question_id": question.id,
                        "type": question.type,
                        "retriever": retriever.name,
                        "unit_kind": retriever.unit_kind,
                        "budget": budget,
                        "units": len(packed),
                        "tokens": context_tokens(packed, count),
                        "answer_recall": hit,
                        "support_recall": recall,
                        "full_support": recall == 1.0,
                        **ranking.diagnostics,
                    }
                )
        logger.info(
            "bench.retriever_done",
            retriever=retriever.name,
            questions=len(corpus.eval_questions),
            seconds=round(time.perf_counter() - started, 1),
        )
    return records


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_outputs(
    out_dir: Path, config: BenchConfig, corpus: Corpus, records: list[dict[str, Any]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(records, out_dir / "per_question.jsonl.gz")
    summary = summarize(records, resamples=config.bootstrap_resamples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    order = [name for name in RETRIEVERS if name in config.retrievers]
    (out_dir / "results.md").write_text(
        f"# {config.name}\n\n" + markdown_tables(summary, order) + "\n"
    )
    manifest = {
        "config": config.model_dump(),
        "corpus": {
            "fingerprint": corpus.fingerprint,
            "questions": len(corpus.questions),
            "background_questions": corpus.background_questions,
            "eval_questions": len(corpus.eval_questions),
            "bridge_eval_questions": sum(q.is_bridge for q in corpus.eval_questions),
            "paragraphs": len(corpus.paragraphs),
            "sentences": sum(len(p.sentences) for p in corpus.paragraphs.values()),
            "triples": len(corpus.triples),
            "entities": len(corpus.entities),
        },
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("sentence-transformers", "torch", "neo4j", "bm25s", "numpy")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_corpus(config: BenchConfig) -> Corpus:
    sample = sample_rows(load_rows(download_two_wiki(config.split)), config.n_corpus, config.seed)
    background = None
    if config.background_split is not None:
        paths = download_two_wiki(config.background_split)
        index = load_rows(paths, columns=["id", "evidences"])
        background = load_rows_by_id(paths, related_ids(sample, index, config.hub_max))
    return build_corpus(sample, config.n_eval, background)


async def run(config: BenchConfig, out_dir: Path) -> None:
    corpus = load_corpus(config)
    logger.info(
        "bench.corpus",
        fingerprint=corpus.fingerprint,
        paragraphs=len(corpus.paragraphs),
        triples=len(corpus.triples),
        background_questions=corpus.background_questions,
        eval_questions=len(corpus.eval_questions),
    )
    encoder = Encoder(config.embedding_model, CACHE_DIR / "embeddings" / corpus.fingerprint)
    count = HFTokenCounter(config.embedding_model)

    store: Neo4jGraphStore | None = None
    if any(name.startswith("graph_") for name in config.retrievers):
        settings = get_settings()
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password.get_secret_value(),
            database=settings.neo4j_database,
        )
    try:
        retrievers = await build_retrievers(config.retrievers, corpus, encoder, store, config)
        records = await evaluate(retrievers, corpus, config.budgets, count)
    finally:
        if store is not None:
            await store.aclose()
    write_outputs(out_dir, config, corpus, records)
    logger.info("bench.done", out_dir=str(out_dir), records=len(records))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m graphrag_lab.bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="run a benchmark config")
    run_cmd.add_argument("--config", type=Path, required=True)
    run_cmd.add_argument("--out", type=Path, help="defaults to results/<config name>")
    report_cmd = commands.add_parser("report", help="rebuild tables from per_question.jsonl.gz")
    report_cmd.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        config = BenchConfig.from_yaml(args.config)
        asyncio.run(run(config, args.out or Path("results") / config.name))
    else:
        manifest = json.loads((args.results_dir / "manifest.json").read_text())
        config = BenchConfig.model_validate(manifest["config"])
        records = read_records(args.results_dir / "per_question.jsonl.gz")
        summary = summarize(records, resamples=config.bootstrap_resamples)
        (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        order = [name for name in RETRIEVERS if name in config.retrievers]
        (args.results_dir / "results.md").write_text(
            f"# {config.name}\n\n" + markdown_tables(summary, order) + "\n"
        )
