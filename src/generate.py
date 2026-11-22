"""Builds the RAG pipeline from a config file.

    from src.pipeline import build_pipeline
    pipe = build_pipeline("configs/dense.yaml")
    result = pipe("What loss function does the paper use?")

Hand-test from the command line:
    python -m src.pipeline --config configs/dense.yaml "your question here"

Every query is appended to logs/<config name>.jsonl (question, chunks,
answer, timings, tokens) so you can do the failure analysis later.
"""
import argparse
import json
import pathlib
import time

import yaml

from src.generate import MODEL, generate
from src.retrieval.dense import DenseRetriever


def load_config(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


def build_retriever(cfg: dict):
    kind = cfg.get("retriever", "dense")
    if kind == "dense":
        return DenseRetriever(
            collection=cfg["collection"],
            url=cfg.get("qdrant_url", "http://localhost:6333"),
            embedding_model=cfg.get("embedding_model", "BAAI/bge-m3"),
            device=cfg.get("embedding_device", "cpu"),
        )
    # Day 2: write src/retrieval/bm25.py and hybrid.py with the same
    # .search(question, k) -> list[dict] method, then add them here.
    raise NotImplementedError(f"retriever '{kind}' is not written yet")


def build_pipeline(config_path: str):
    cfg = load_config(config_path)
    if cfg.get("rerank"):
        # Day 2: add src/retrieval/rerank.py and apply it between search and generate.
        raise NotImplementedError("reranker is not written yet")

    retriever = build_retriever(cfg)
    top_k = cfg.get("top_k", 5)
    generator_model = cfg.get("generator_model", MODEL)

    name = cfg.get("name", pathlib.Path(config_path).stem)
    log_path = pathlib.Path("logs") / f"{name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)

    def pipe(question: str) -> dict:
        t0 = time.perf_counter()
        chunks = retriever.search(question, top_k)
        t1 = time.perf_counter()
        gen = generate(question, chunks, model=generator_model)
        t2 = time.perf_counter()

        result = {
            "answer": gen["answer"],
            "contexts": [c["text"] for c in chunks],
            "retrieved_sources": [c.get("source", "unknown") for c in chunks],
            "latency": t2 - t0,  # end-to-end, this is what p95 is computed from
            "retrieval_latency": t1 - t0,
            "generation_latency": t2 - t1,
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"],
        }
        with log_path.open("a") as f:
            f.write(json.dumps({"question": question, "config": name, **result},
                               ensure_ascii=False) + "\n")
        return result

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("question")
    args = parser.parse_args()

    out = build_pipeline(args.config)(args.question)
    print(out["answer"])
    print("\n--- sources ---")
    for src in out["retrieved_sources"]:
        print(" ", src)
    print(f"\nlatency {out['latency']:.1f}s | prompt tokens {out['prompt_tokens']}")