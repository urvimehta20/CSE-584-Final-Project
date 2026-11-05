"""Shared folds, metrics, cluster bootstrap, and tokenizer for the follow-up."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import GOLD_THRESHOLD, PROCESSED, SEED
from evaluate import assign_question_split

N_BOOT_FOLLOWUP = 2000
KS_FOLLOWUP = (20, 50, 100)
TOKEN_OVERHEAD = 8
TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000)
RECALL_TARGETS = (0.5, 0.8, 1.0)
PRIMARY_K = 50
PRIMARY_R = 0.8
PILLAR_LAMBDAS = (1, 2)
LEAF_LAMBDAS = (1, 2, 3)

SMALL_GOLD_QUESTIONS = (
    "Does the company provide definitions for climate change adaptation?",
    "Does the company report a plan to engage with downstream partners on water consumption or water pollution?",
    "Does the company refer to any third party scenarios when identifying climate-related risks or opportunities (e.g. IPCC trajectories, NGFS scenarios, etc.)?",
)


def core_folds(queries: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    _, info = assign_question_split(queries, seed)
    a = list(info["core_dev"])
    b = list(info["core_test"])
    return {
        "seed": seed,
        "original_core_dev": a,
        "original_core_test": b,
        "folds": [
            {"fold_id": 0, "tune": a, "test": b, "label": "tune=original_core_dev, test=original_core_test"},
            {"fold_id": 1, "tune": b, "test": a, "label": "tune=original_core_test, test=original_core_dev"},
        ],
        "rule": "2-fold swap of the seed-42 8/8 question split from the pilot. Every core question is held out once.",
    }


def gold_chunk_indices(
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    chunk_pos: dict[str, int],
    threshold: float = GOLD_THRESHOLD,
) -> dict[str, dict[str, np.ndarray]]:
    """Return gold idx lists keyed by query_id and by question (union)."""
    pos = qrels.loc[qrels["relevance"] >= threshold]
    by_query: dict[str, list[int]] = {qid: [] for qid in queries["query_id"]}
    by_question: dict[str, set[int]] = {}
    qmap = queries.set_index("query_id")["question"].to_dict()
    rel_by_qc: dict[tuple[str, int], float] = {}
    for qid, cid, rel in zip(pos["query_id"], pos["chunk_id"], pos["relevance"]):
        if cid not in chunk_pos:
            continue
        idx = chunk_pos[cid]
        by_query[qid].append(idx)
        question = qmap[qid]
        by_question.setdefault(question, set()).add(idx)
        rel_by_qc[(qid, idx)] = float(rel)
        # question-level: keep max relevance if the same chunk is gold for multiple reports
        key_q = ("__question__", question, idx)
        rel_by_qc[key_q] = max(float(rel), rel_by_qc.get(key_q, 0.0))
    return {
        "by_query": {k: np.array(v, dtype=int) for k, v in by_query.items()},
        "by_question": {k: np.array(sorted(v), dtype=int) for k, v in by_question.items()},
        "relevance": rel_by_qc,
    }


def capped_recall(gold: np.ndarray, ranked: np.ndarray, k: int) -> float | None:
    if gold.size == 0:
        return None
    denom = min(int(k), int(gold.size))
    hit = np.intersect1d(gold, ranked[:k], assume_unique=False).size
    return float(hit / denom)


def raw_recall(gold: np.ndarray, ranked: np.ndarray, k: int) -> float | None:
    if gold.size == 0:
        return None
    hit = np.intersect1d(gold, ranked[:k], assume_unique=False).size
    return float(hit / gold.size)


def miss_rate(gold: np.ndarray, candidates: np.ndarray) -> float | None:
    if gold.size == 0:
        return None
    in_cand = np.intersect1d(gold, candidates, assume_unique=False).size
    return float((gold.size - in_cand) / gold.size)


def cluster_bootstrap_mean(
    unit_question: list[str],
    values: np.ndarray,
    n_boot: int = N_BOOT_FOLLOWUP,
    seed: int = SEED,
    average: str = "unit",
) -> dict[str, Any]:
    """Cluster bootstrap over questions.

    average='unit': point estimate is mean(values) (setting R: queries).
    average='question': point estimate is mean of per-question means (setting G).
    """
    values = np.asarray(values, dtype=float)
    questions = np.asarray(unit_question)
    finite = np.isfinite(values)
    values, questions = values[finite], questions[finite]
    if values.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_units": 0, "n_questions": 0, "note": "no finite values"}

    uniq, inv = np.unique(questions, return_inverse=True)
    groups = [values[inv == i] for i in range(len(uniq))]

    if average == "question":
        qmeans = np.array([g.mean() for g in groups], dtype=float)
        point = float(qmeans.mean())
    else:
        point = float(values.mean())

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    nq = len(groups)
    for b in range(n_boot):
        idx = rng.integers(0, nq, nq)
        if average == "question":
            boots[b] = np.mean([groups[i].mean() for i in idx])
        else:
            stacked = np.concatenate([groups[i] for i in idx])
            boots[b] = stacked.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_units": int(values.size),
        "n_questions": int(nq),
        "n_boot": n_boot,
        "average": average,
        "seed": seed,
    }


def paired_cluster_diff(
    unit_question: list[str],
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = N_BOOT_FOLLOWUP,
    seed: int = SEED,
    average: str = "unit",
) -> dict[str, Any]:
    """Paired (B − A) cluster bootstrap. Same questions in both arms."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    qs = np.asarray(unit_question)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, qs = a[ok], b[ok], qs[ok]
    if a.size == 0:
        return {
            "mean_diff": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_units": 0,
            "n_questions": 0,
            "ci_entirely_above_0": False,
            "note": "no paired finite values",
        }
    diff = b - a
    summary = cluster_bootstrap_mean(qs.tolist(), diff, n_boot, seed, average)
    lo, hi = summary["ci95_low"], summary["ci95_high"]
    return {
        "mean_diff": summary["mean"],
        "ci95_low": lo,
        "ci95_high": hi,
        "n_units": summary["n_units"],
        "n_questions": summary["n_questions"],
        "n_boot": n_boot,
        "average": average,
        "ci_entirely_above_0": bool(lo is not None and lo > 0),
        "ci_excludes_0": bool(lo is not None and hi is not None and (hi < 0 or lo > 0)),
    }


def load_tokenizer() -> tuple[Any, str]:
    try:
        import tiktoken
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"tiktoken is required: {type(exc).__name__}: {exc}") from exc
    for name in ("o200k_base", "cl100k_base"):
        try:
            enc = tiktoken.get_encoding(name)
            print(f"Using tiktoken encoding {name}")
            return enc, name
        except Exception as exc:  # noqa: BLE001
            print(f"tiktoken encoding {name} unavailable: {type(exc).__name__}: {exc}")
    raise RuntimeError("Neither o200k_base nor cl100k_base could be loaded.")


def chunk_token_costs(texts: list[str], enc, overhead: int = TOKEN_OVERHEAD) -> np.ndarray:
    return np.array([len(enc.encode(t)) + overhead for t in texts], dtype=np.int32)


def tokens_until_recall(
    ranked: np.ndarray, gold: np.ndarray, costs: np.ndarray, r: float
) -> dict[str, Any]:
    if gold.size == 0:
        return {"tokens": None, "reachable": False, "note": "no gold"}
    gold_set = set(gold.tolist())
    need = r * gold.size
    got = 0
    toks = 0
    for idx in ranked.tolist():
        toks += int(costs[idx])
        if idx in gold_set:
            got += 1
            if got + 1e-12 >= need:
                return {"tokens": int(toks), "reachable": True, "recall_reached": float(got / gold.size)}
    return {
        "tokens": int(toks),
        "reachable": False,
        "recall_reached": float(got / gold.size) if gold.size else None,
    }


def recall_at_token_budget(
    ranked: np.ndarray, gold: np.ndarray, costs: np.ndarray, budget: int
) -> float | None:
    if gold.size == 0:
        return None
    gold_set = set(gold.tolist())
    got = 0
    toks = 0
    for idx in ranked.tolist():
        nxt = toks + int(costs[idx])
        if nxt > budget:
            break
        toks = nxt
        if idx in gold_set:
            got += 1
    return float(got / gold.size)
