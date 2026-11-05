"""Embed chunks and questions with a cached, L2-normalized encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BGE_QUERY_PREFIX,
    EMB_DIR,
    EMBED_MODEL_FALLBACK,
    EMBED_MODEL_PRIMARY,
    PROCESSED,
    SEED,
)


def _load_model() -> tuple[object, str, str | None]:
    from sentence_transformers import SentenceTransformer

    last_error = None
    for name in (EMBED_MODEL_PRIMARY, EMBED_MODEL_FALLBACK):
        try:
            model = SentenceTransformer(name)
            prefix = BGE_QUERY_PREFIX if name.startswith("BAAI/bge") else None
            print(f"Loaded embedding model: {name}")
            return model, name, prefix
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Failed to load {name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load any embedding model. Last error: {last_error}")


def _encode(model, texts: list[str], prefix: str | None, is_query: bool) -> np.ndarray:
    payload = [((prefix or "") + t) if is_query and prefix else t for t in texts]
    emb = model.encode(
        payload,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    emb = emb / norms
    return emb


def _cache_ok(path: Path, n: int, dim: int | None = None) -> bool:
    if not path.exists():
        return False
    arr = np.load(path)
    if arr.shape[0] != n:
        return False
    if dim is not None and arr.shape[1] != dim:
        return False
    return True


def embed_corpus(chunks: pd.DataFrame, queries: pd.DataFrame) -> dict:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = EMB_DIR / "meta.json"
    chunk_path = EMB_DIR / "chunk_embeddings.npy"
    question_path = EMB_DIR / "question_embeddings.npy"
    question_key_path = EMB_DIR / "question_texts.json"

    unique_questions = sorted(queries["question"].astype(str).unique())
    chunk_texts = chunks["text"].astype(str).tolist()

    reuse = False
    if meta_path.exists() and question_key_path.exists():
        meta = json.loads(meta_path.read_text())
        saved_qs = json.loads(question_key_path.read_text())
        if (
            saved_qs == unique_questions
            and _cache_ok(chunk_path, len(chunks))
            and _cache_ok(question_path, len(unique_questions))
        ):
            reuse = True
            model_name = meta.get("model_name")
            query_prefix = meta.get("query_prefix")
            print(f"Reusing cached embeddings from {EMB_DIR} (model={model_name})")
            chunk_emb = np.load(chunk_path)
            question_emb = np.load(question_path)
        else:
            print("Embedding cache does not match current tables; recomputing.")

    if not reuse:
        model, model_name, query_prefix = _load_model()
        print(f"Embedding {len(chunk_texts)} chunks (no query prefix)...")
        chunk_emb = _encode(model, chunk_texts, query_prefix, is_query=False)
        print(f"Embedding {len(unique_questions)} unique questions (query prefix={query_prefix!r})...")
        question_emb = _encode(model, unique_questions, query_prefix, is_query=True)
        np.save(chunk_path, chunk_emb)
        np.save(question_path, question_emb)
        question_key_path.write_text(json.dumps(unique_questions, indent=2))
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "query_prefix": query_prefix,
                    "seed": SEED,
                    "n_chunks": int(len(chunks)),
                    "n_questions": int(len(unique_questions)),
                    "dim": int(chunk_emb.shape[1]),
                    "normalized": True,
                },
                indent=2,
            )
        )

    qmap = {q: i for i, q in enumerate(unique_questions)}
    query_emb = question_emb[np.array([qmap[q] for q in queries["question"].astype(str)], dtype=int)]
    out = {
        "chunk_embeddings": chunk_emb,
        "question_embeddings": question_emb,
        "query_embeddings": query_emb,
        "question_texts": unique_questions,
        "question_index": qmap,
        "model_name": json.loads(meta_path.read_text())["model_name"],
        "query_prefix": json.loads(meta_path.read_text())["query_prefix"],
        "dim": int(chunk_emb.shape[1]),
    }
    print(
        f"Embeddings ready: chunks={chunk_emb.shape} questions={question_emb.shape} "
        f"model={out['model_name']}"
    )
    return out


def main() -> dict:
    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    return embed_corpus(chunks, queries)


if __name__ == "__main__":
    main()
