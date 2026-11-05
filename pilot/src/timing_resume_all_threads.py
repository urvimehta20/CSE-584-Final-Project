"""Resume all-threads timing from timing_partial.json after a native crash."""

from __future__ import annotations

import json
import os

import numpy as np

from config import EMB_DIR, RESULTS, SEED, SELECTIVITIES, SYNTHETIC_NS
from timing import (
    _set_threads,
    _summarize,
    _time_loop,
    _try_faiss_hnsw,
    _write_partial,
    make_queries,
    make_scaled_corpus,
    TIMING_N_QUERIES,
    TIMING_WARMUP,
)


def main() -> None:
    partial_path = RESULTS / "timing_partial.json"
    data = json.loads(partial_path.read_text())
    results = data["settings"]
    savings = data["savings_filtered_vs_unfiltered_exact"]
    failures = data.get("failures") or []
    chunk_emb = np.load(EMB_DIR / "chunk_embeddings.npy")
    query_emb = np.load(EMB_DIR / "question_embeddings.npy")
    node_emb = np.load(EMB_DIR / "node_embeddings.npy")
    real_n = int(chunk_emb.shape[0])
    n_cores = os.cpu_count() or 1
    q_pool = make_queries(query_emb, TIMING_N_QUERIES + TIMING_WARMUP, SEED)
    thr_info = _set_threads(n_cores)

    done = {
        (s.get("threads"), s.get("N"), s.get("method"), s.get("filtered"), s.get("selectivity"), s.get("efSearch"))
        for s in results
    }
    sizes = [{"N": real_n, "synthetic": False, "label": f"real_N={real_n}"}]
    for n in SYNTHETIC_NS:
        sizes.append({"N": int(n), "synthetic": True, "label": f"synthetic_N={n}"})

    def missing(n, method, filtered, sel=None, ef=None) -> bool:
        return ("all_threads", n, method, filtered, sel, ef) not in done

    for size in sizes:
        n = size["N"]
        print(f"resume all_threads {size['label']}")
        x = make_scaled_corpus(chunk_emb, n, SEED + n, size["synthetic"])

        def _exact_factory(mat: np.ndarray, qs: np.ndarray):
            acc = np.empty(1, dtype=np.float32)

            def _one(i: int) -> None:
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    acc[0] = float(np.max(mat @ qs[i]))

            return _one

        if missing(n, "exact_numpy_matmul", False, 1.0):
            lat = _time_loop(_exact_factory(x, q_pool), TIMING_N_QUERIES, TIMING_WARMUP)
            exact_rec = {
                "status": "ok",
                "method": "exact_numpy_matmul",
                "filtered": False,
                "selectivity": 1.0,
                "N": n,
                "synthetic": size["synthetic"],
                "label": size["label"],
                "threads": "all_threads",
                "n_threads": n_cores,
                "thread_control": thr_info,
                "latency_ms": _summarize(lat),
            }
            results.append(exact_rec)
        else:
            exact_rec = next(
                s
                for s in results
                if s.get("threads") == "all_threads"
                and s.get("N") == n
                and s.get("method") == "exact_numpy_matmul"
                and s.get("filtered") is False
            )

        rng = np.random.default_rng(SEED)
        for s in SELECTIVITIES:
            if not missing(n, "exact_numpy_matmul", True, float(s)):
                continue
            k = max(1, int(round(s * n)))
            subset = x[rng.choice(n, size=k, replace=False)]
            lat_f = _time_loop(_exact_factory(subset, q_pool), TIMING_N_QUERIES, TIMING_WARMUP)
            rec = {
                "status": "ok",
                "method": "exact_numpy_matmul",
                "filtered": True,
                "selectivity": float(s),
                "subset_size": k,
                "N": n,
                "synthetic": size["synthetic"],
                "label": size["label"],
                "threads": "all_threads",
                "n_threads": n_cores,
                "latency_ms": _summarize(lat_f),
            }
            results.append(rec)
            if exact_rec.get("status") == "ok":
                save = float(exact_rec["latency_ms"]["median"] - rec["latency_ms"]["median"])
                savings.append(
                    {
                        "N": n,
                        "synthetic": size["synthetic"],
                        "threads": "all_threads",
                        "selectivity": float(s),
                        "unfiltered_median_ms": exact_rec["latency_ms"]["median"],
                        "filtered_median_ms": rec["latency_ms"]["median"],
                        "saving_ms": save,
                        "exceeds_1ms": bool(save > 1.0),
                    }
                )
        _write_partial(results, savings, failures, real_n)

        if missing(n, "faiss_hnsw", False, 1.0, 32) or missing(n, "faiss_hnsw", False, 1.0, 128):
            msg = (
                f"Skipped FAISS HNSW for all_threads N={n}: a previous all-threads "
                "HNSW run aborted this process with a native segfault (exit 139)."
            )
            print(msg)
            failures.append(msg)
            for ef in (32, 128):
                results.append(
                    {
                        "status": "failed",
                        "method": "faiss_hnsw",
                        "M": 32,
                        "efSearch": ef,
                        "filtered": False,
                        "selectivity": 1.0,
                        "N": n,
                        "synthetic": size["synthetic"],
                        "label": size["label"],
                        "threads": "all_threads",
                        "n_threads": n_cores,
                        "error": msg,
                    }
                )
        del x
        _write_partial(results, savings, failures, real_n)

    if missing(None, "topic_scoring_overhead", None, None):
        nodes = np.asarray(node_emb, dtype=np.float32)

        def _topic(i: int) -> None:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                _ = nodes @ q_pool[i]

        lat_t = _time_loop(_topic, TIMING_N_QUERIES, TIMING_WARMUP)
        results.append(
            {
                "status": "ok",
                "method": "topic_scoring_overhead",
                "n_nodes": int(nodes.shape[0]),
                "N": None,
                "synthetic": False,
                "threads": "all_threads",
                "n_threads": n_cores,
                "latency_ms": _summarize(lat_t),
            }
        )
    _write_partial(results, savings, failures, real_n)

    over = [s for s in savings if s.get("exceeds_1ms")]
    out = {
        "n_queries_timed": TIMING_N_QUERIES,
        "warmup": TIMING_WARMUP,
        "real_corpus_size": real_n,
        "settings": results,
        "savings_filtered_vs_unfiltered_exact": savings,
        "where_saving_exceeds_1ms": {
            "min_N": min((s["N"] for s in over), default=None),
            "examples": over,
            "note": (
                "N values where median filtered exact search was >1 ms faster than unfiltered exact."
                if over
                else "No completed setting had a filtered-vs-unfiltered exact saving above 1 ms."
            ),
        },
        "failures": failures + [
            "Original all-threads pass crashed after real_N exact/filtered; remaining all-threads settings were resumed in a new process."
        ],
        "resumed_all_threads": True,
    }
    (RESULTS / "timing_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RESULTS / 'timing_results.json'} settings={len(results)}")


if __name__ == "__main__":
    main()
