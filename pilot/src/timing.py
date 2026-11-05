"""Latency pilot: exact, FAISS HNSW, and filtered exact search at several N."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np

from config import (
    EMB_DIR,
    HNSW_EFSEARCH,
    HNSW_M,
    NOISE_STD,
    RESULTS,
    SEED,
    SELECTIVITIES,
    SYNTHETIC_NS,
    TIMING_N_QUERIES,
    TIMING_WARMUP,
)


def _set_threads(n: int) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    info: dict[str, Any] = {"requested_threads": n}
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
        info["threadpoolctl"] = True
    except Exception as exc:  # noqa: BLE001
        info["threadpoolctl"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import torch

        torch.set_num_threads(n)
        info["torch_num_threads"] = torch.get_num_threads()
    except Exception as exc:  # noqa: BLE001
        info["torch_num_threads"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import faiss

        faiss.omp_set_num_threads(n)
        info["faiss_threads"] = n
    except Exception as exc:  # noqa: BLE001
        info["faiss_threads"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


def _summarize(latencies_ms: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(latencies_ms, dtype=float)
    if arr.size == 0:
        return {"median": None, "p95": None, "n": 0, "note": "no timings"}
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }


def _time_loop(fn: Callable[[int], None], n_queries: int, warmup: int) -> np.ndarray:
    for i in range(warmup):
        fn(i)
    out = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        t0 = time.perf_counter()
        fn(i)
        out[i] = (time.perf_counter() - t0) * 1000.0
    return out


def make_scaled_corpus(real: np.ndarray, n: int, seed: int, synthetic: bool) -> np.ndarray:
    real = np.asarray(real, dtype=np.float32)
    if not synthetic and n == len(real):
        return real
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(real), size=n)
    x = real[idx] + rng.normal(0.0, NOISE_STD, size=(n, real.shape[1])).astype(np.float32)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x


def make_queries(real_q: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(real_q), size=n)
    return np.asarray(real_q[idx], dtype=np.float32)


def _try_faiss_hnsw(x: np.ndarray, queries: np.ndarray, threads: int) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    try:
        import faiss
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "method": "faiss_hnsw",
                "M": HNSW_M,
                "efSearch": ef,
                "status": "failed",
                "error": f"faiss import failed: {type(exc).__name__}: {exc}",
                "N": int(x.shape[0]),
            }
            for ef in HNSW_EFSEARCH
        ]
    try:
        faiss.omp_set_num_threads(threads)
        index = faiss.IndexHNSWFlat(int(x.shape[1]), HNSW_M)
        t0 = time.perf_counter()
        index.add(np.ascontiguousarray(x, dtype=np.float32))
        build_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "method": "faiss_hnsw",
                "M": HNSW_M,
                "efSearch": ef,
                "status": "failed",
                "error": f"index build failed: {type(exc).__name__}: {exc}",
                "N": int(x.shape[0]),
            }
            for ef in HNSW_EFSEARCH
        ]
    for ef in HNSW_EFSEARCH:
        rec: dict[str, Any] = {
            "method": "faiss_hnsw",
            "M": HNSW_M,
            "efSearch": ef,
            "N": int(x.shape[0]),
            "build_ms": float(build_ms),
        }
        try:
            index.hnsw.efSearch = int(ef)

            q_cont = np.ascontiguousarray(queries, dtype=np.float32)

            def _one(i: int, _index=index, _q=q_cont) -> None:
                _index.search(_q[i : i + 1], 10)

            lat = _time_loop(_one, TIMING_N_QUERIES, TIMING_WARMUP)
            rec.update({"status": "ok", "latency_ms": _summarize(lat)})
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        recs.append(rec)
    return recs


def run_timing_pilot(
    chunk_emb: np.ndarray,
    query_emb: np.ndarray,
    n_nodes: int,
    node_emb: np.ndarray,
    n_cores: int,
) -> dict[str, Any]:
    real_n = int(chunk_emb.shape[0])
    sizes = [{"N": real_n, "synthetic": False, "label": f"real_N={real_n}"}]
    for n in SYNTHETIC_NS:
        sizes.append({"N": int(n), "synthetic": True, "label": f"synthetic_N={n}"})

    thread_settings = [
        {"name": "single_thread", "threads": 1},
        {"name": "all_threads", "threads": int(n_cores or 1)},
    ]
    q_pool = make_queries(query_emb, TIMING_N_QUERIES + TIMING_WARMUP, SEED)

    results: list[dict[str, Any]] = []
    savings: list[dict[str, Any]] = []
    failures: list[str] = []

    for thr in thread_settings:
        thr_info = _set_threads(thr["threads"])
        print(f"\n=== Timing threads={thr['name']} ({thr['threads']}) ===")
        for size in sizes:
            print(f"  corpus {size['label']}")
            try:
                x = make_scaled_corpus(chunk_emb, size["N"], SEED + size["N"], size["synthetic"])
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to materialize corpus {size['label']}: {type(exc).__name__}: {exc}"
                print(msg)
                failures.append(msg)
                continue

            # Exact unfiltered
            def _exact_factory(mat: np.ndarray, qs: np.ndarray) -> Callable[[int], None]:
                acc = np.empty(1, dtype=np.float32)

                def _one(i: int) -> None:
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        acc[0] = float(np.max(mat @ qs[i]))

                return _one

            try:
                lat_exact = _time_loop(_exact_factory(x, q_pool), TIMING_N_QUERIES, TIMING_WARMUP)
                exact_rec = {
                    "status": "ok",
                    "method": "exact_numpy_matmul",
                    "filtered": False,
                    "selectivity": 1.0,
                    "N": size["N"],
                    "synthetic": size["synthetic"],
                    "label": size["label"],
                    "threads": thr["name"],
                    "n_threads": thr["threads"],
                    "thread_control": thr_info,
                    "latency_ms": _summarize(lat_exact),
                }
            except Exception as exc:  # noqa: BLE001
                exact_rec = {
                    "status": "failed",
                    "method": "exact_numpy_matmul",
                    "N": size["N"],
                    "synthetic": size["synthetic"],
                    "label": size["label"],
                    "threads": thr["name"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(f"exact {size['label']} {thr['name']}: {exact_rec['error']}")
                lat_exact = np.array([])
            results.append(exact_rec)

            # Filtered exact
            rng = np.random.default_rng(SEED)
            for s in SELECTIVITIES:
                k = max(1, int(round(s * size["N"])))
                subset = x[rng.choice(size["N"], size=k, replace=False)]
                try:
                    lat_f = _time_loop(_exact_factory(subset, q_pool), TIMING_N_QUERIES, TIMING_WARMUP)
                    rec = {
                        "status": "ok",
                        "method": "exact_numpy_matmul",
                        "filtered": True,
                        "selectivity": float(s),
                        "subset_size": k,
                        "N": size["N"],
                        "synthetic": size["synthetic"],
                        "label": size["label"],
                        "threads": thr["name"],
                        "n_threads": thr["threads"],
                        "latency_ms": _summarize(lat_f),
                    }
                    results.append(rec)
                    if exact_rec.get("status") == "ok":
                        save = float(exact_rec["latency_ms"]["median"] - rec["latency_ms"]["median"])
                        savings.append(
                            {
                                "N": size["N"],
                                "synthetic": size["synthetic"],
                                "threads": thr["name"],
                                "selectivity": float(s),
                                "unfiltered_median_ms": exact_rec["latency_ms"]["median"],
                                "filtered_median_ms": rec["latency_ms"]["median"],
                                "saving_ms": save,
                                "exceeds_1ms": bool(save > 1.0),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    msg = f"filtered s={s} {size['label']} {thr['name']}: {type(exc).__name__}: {exc}"
                    failures.append(msg)
                    results.append(
                        {
                            "status": "failed",
                            "method": "exact_numpy_matmul",
                            "filtered": True,
                            "selectivity": float(s),
                            "N": size["N"],
                            "synthetic": size["synthetic"],
                            "threads": thr["name"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            _write_partial(results, savings, failures, real_n)

            # FAISS HNSW: all-threads mode segfaulted on this Mac (exit 139) in the first
            # run, so multi-thread HNSW is skipped and recorded as a failure.
            if thr["threads"] > 1:
                msg = (
                    f"Skipped FAISS HNSW for threads={thr['name']} N={size['N']}: "
                    "faiss-cpu HNSW aborted with a native segfault (exit 139) when "
                    "omp threads > 1 on this machine."
                )
                print(msg)
                failures.append(msg)
                for ef in HNSW_EFSEARCH:
                    results.append(
                        {
                            "status": "failed",
                            "method": "faiss_hnsw",
                            "M": HNSW_M,
                            "efSearch": ef,
                            "filtered": False,
                            "selectivity": 1.0,
                            "N": size["N"],
                            "synthetic": size["synthetic"],
                            "label": size["label"],
                            "threads": thr["name"],
                            "n_threads": thr["threads"],
                            "error": msg,
                        }
                    )
                del x
                _write_partial(results, savings, failures, real_n)
                continue

            # FAISS HNSW unfiltered; one index build per (N, threads), both efSearch values
            for rec in _try_faiss_hnsw(x, q_pool, thr["threads"]):
                rec.update(
                    {
                        "filtered": False,
                        "selectivity": 1.0,
                        "synthetic": size["synthetic"],
                        "label": size["label"],
                        "threads": thr["name"],
                        "n_threads": thr["threads"],
                    }
                )
                results.append(rec)
                if rec.get("status") != "ok":
                    failures.append(
                        f"faiss ef={rec.get('efSearch')} {size['label']} {thr['name']}: {rec.get('error')}"
                    )

            del x
            _write_partial(results, savings, failures, real_n)

        # Topic-scoring overhead: query vs node centroids (independent of N)
        try:
            nodes = np.asarray(node_emb, dtype=np.float32)

            def _topic(i: int) -> None:
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    _ = nodes @ q_pool[i]

            lat_t = _time_loop(_topic, TIMING_N_QUERIES, TIMING_WARMUP)
            results.append(
                {
                    "status": "ok",
                    "method": "topic_scoring_overhead",
                    "n_nodes": int(n_nodes),
                    "N": None,
                    "synthetic": False,
                    "threads": thr["name"],
                    "n_threads": thr["threads"],
                    "latency_ms": _summarize(lat_t),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"topic scoring {thr['name']}: {type(exc).__name__}: {exc}")

    first_n_over_1ms = None
    over = [s for s in savings if s["exceeds_1ms"]]
    if over:
        first_n_over_1ms = {
            "min_N": min(s["N"] for s in over),
            "examples": over,
            "note": "N values (and settings) where median filtered exact search was >1 ms faster than unfiltered exact.",
        }
    else:
        first_n_over_1ms = {
            "min_N": None,
            "examples": [],
            "note": "In this run, filtered exact search was never more than 1 ms faster than unfiltered exact at any measured N.",
        }

    return {
        "n_queries_timed": TIMING_N_QUERIES,
        "warmup": TIMING_WARMUP,
        "real_corpus_size": real_n,
        "settings": results,
        "savings_filtered_vs_unfiltered_exact": savings,
        "where_saving_exceeds_1ms": first_n_over_1ms,
        "failures": failures,
    }


def _write_partial(results, savings, failures, real_n) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries_timed": TIMING_N_QUERIES,
        "warmup": TIMING_WARMUP,
        "real_corpus_size": real_n,
        "settings": results,
        "savings_filtered_vs_unfiltered_exact": savings,
        "failures": failures,
        "partial": True,
    }
    (RESULTS / "timing_partial.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    chunk_emb = np.load(EMB_DIR / "chunk_embeddings.npy")
    query_emb = np.load(EMB_DIR / "question_embeddings.npy")
    node_emb = np.load(EMB_DIR / "node_embeddings.npy")
    env_cores = os.cpu_count() or 1
    out = run_timing_pilot(chunk_emb, query_emb, int(node_emb.shape[0]), node_emb, env_cores)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "timing_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RESULTS / 'timing_results.json'}")


if __name__ == "__main__":
    main()
