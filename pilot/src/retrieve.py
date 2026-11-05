"""Report-scoped retrieval for configs A, B, D, and E."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import GOLD_THRESHOLD, K_LIST, LAMBDAS, LEVELS
from taxonomy import (
    Taxonomy,
    assign_leaves,
    membership_for_level,
    route_top_lambda,
)


@dataclass
class QueryPack:
    query_ids: np.ndarray
    questions: np.ndarray
    reports: np.ndarray
    set_type: np.ndarray
    question_split: np.ndarray
    eval_eligible: np.ndarray
    query_emb: np.ndarray
    gold_idx: list[np.ndarray]
    report_idx: list[np.ndarray]


def build_index_maps(
    chunks: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    query_emb: np.ndarray,
    chunk_ids_order: np.ndarray | None = None,
) -> QueryPack:
    chunk_pos = {cid: i for i, cid in enumerate(chunks["chunk_id"].tolist())}
    report_to_idx: dict[str, np.ndarray] = {}
    for report, grp in chunks.groupby("report", sort=False):
        report_to_idx[report] = np.array([chunk_pos[c] for c in grp["chunk_id"]], dtype=int)

    gold_by_q: dict[str, list[int]] = {qid: [] for qid in queries["query_id"]}
    pos = qrels.loc[qrels["relevance"] >= GOLD_THRESHOLD]
    for qid, cid in zip(pos["query_id"], pos["chunk_id"]):
        if cid in chunk_pos:
            gold_by_q[qid].append(chunk_pos[cid])

    gold_idx = [np.array(gold_by_q[qid], dtype=int) for qid in queries["query_id"]]
    report_idx = [report_to_idx[r] for r in queries["report"]]
    return QueryPack(
        query_ids=queries["query_id"].to_numpy(),
        questions=queries["question"].to_numpy(),
        reports=queries["report"].to_numpy(),
        set_type=queries["set_type"].to_numpy(),
        question_split=queries["question_split"].to_numpy(),
        eval_eligible=queries["eval_eligible"].to_numpy(dtype=bool),
        query_emb=np.asarray(query_emb, dtype=np.float32),
        gold_idx=gold_idx,
        report_idx=report_idx,
    )


def _ids_for_level(tax: Taxonomy, level: str) -> list[str]:
    return tax.leaf_ids if level == "leaf" else tax.pillar_ids


def _route_nodes(q_emb: np.ndarray, level_emb: np.ndarray, lam: int) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = level_emb @ q_emb
    return route_top_lambda(sims, lam)


def _oracle_nodes(gold_idx: np.ndarray, membership: np.ndarray) -> np.ndarray:
    if gold_idx.size == 0:
        return np.array([], dtype=int)
    hits = membership[gold_idx].any(axis=0)
    return np.flatnonzero(hits)


def candidate_indices(
    report_idx: np.ndarray,
    membership: np.ndarray,
    selected_nodes: np.ndarray,
) -> np.ndarray:
    if selected_nodes.size == 0:
        return np.array([], dtype=int)
    in_nodes = membership[report_idx][:, selected_nodes].any(axis=1)
    return report_idx[in_nodes]


def rank_candidates(q_emb: np.ndarray, chunk_emb: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
    if cand_idx.size == 0:
        return cand_idx
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = chunk_emb[cand_idx] @ q_emb
    order = np.argsort(-scores, kind="stable")
    return cand_idx[order]


def run_query(
    q_emb: np.ndarray,
    report_idx: np.ndarray,
    gold_idx: np.ndarray,
    chunk_emb: np.ndarray,
    config: str,
    membership: np.ndarray | None,
    level_emb: np.ndarray | None,
    lam: int,
    include_topic_scoring: bool,
) -> dict:
    t0 = time.perf_counter()
    topic_ms = 0.0
    if config == "A":
        cand = report_idx
    elif config == "E":
        t1 = time.perf_counter()
        selected = _oracle_nodes(gold_idx, membership)
        topic_ms = (time.perf_counter() - t1) * 1000.0
        cand = candidate_indices(report_idx, membership, selected)
    else:
        # B (hierarchy membership at chosen level) or D (leaf membership)
        t1 = time.perf_counter()
        selected = _route_nodes(q_emb, level_emb, lam)
        topic_ms = (time.perf_counter() - t1) * 1000.0
        cand = candidate_indices(report_idx, membership, selected)
    ranked = rank_candidates(q_emb, chunk_emb, cand)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not include_topic_scoring:
        # keep measured total; topic scoring is already inside latency_ms for B/D/E
        pass
    n_report = int(report_idx.size)
    n_cand = int(ranked.size)
    gold_set = set(gold_idx.tolist())
    cand_set = set(ranked.tolist())
    n_gold = len(gold_set)
    n_gold_in_cand = len(gold_set & cand_set)
    miss_rate = None if n_gold == 0 else (n_gold - n_gold_in_cand) / n_gold
    selectivity = None if n_report == 0 else n_cand / n_report
    rec = {}
    for k in K_LIST:
        top = set(ranked[:k].tolist())
        rec[f"recall@{k}"] = None if n_gold == 0 else len(gold_set & top) / n_gold
    missed_gold = np.array([g for g in gold_idx if g not in cand_set], dtype=int)
    return {
        "latency_ms": float(latency_ms),
        "topic_scoring_ms": float(topic_ms),
        "n_candidates": n_cand,
        "n_report_chunks": n_report,
        "n_gold": n_gold,
        "n_gold_in_candidates": n_gold_in_cand,
        "miss_rate": miss_rate,
        "selectivity": selectivity,
        "missed_gold_idx": missed_gold,
        "candidate_idx": ranked,
        **rec,
    }


def evaluate_config_on_queries(
    pack: QueryPack,
    mask: np.ndarray,
    chunk_emb: np.ndarray,
    config: str,
    membership: np.ndarray | None,
    level_emb: np.ndarray | None,
    lam: int,
) -> list[dict]:
    rows = []
    idxs = np.flatnonzero(mask)
    for i in idxs:
        out = run_query(
            q_emb=pack.query_emb[i],
            report_idx=pack.report_idx[i],
            gold_idx=pack.gold_idx[i],
            chunk_emb=chunk_emb,
            config=config,
            membership=membership,
            level_emb=level_emb,
            lam=lam,
            include_topic_scoring=True,
        )
        rows.append(
            {
                "query_id": str(pack.query_ids[i]),
                "question": str(pack.questions[i]),
                "report": str(pack.reports[i]),
                "set_type": str(pack.set_type[i]),
                "question_split": str(pack.question_split[i]),
                "config": config,
                **{k: v for k, v in out.items() if k not in {"missed_gold_idx", "candidate_idx"}},
                "missed_gold_idx": out["missed_gold_idx"],
                "candidate_idx": out["candidate_idx"],
            }
        )
    return rows


def precompute_memberships(chunk_emb: np.ndarray, leaf_emb: np.ndarray, taus: list[float]) -> dict:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        leaf_sim = chunk_emb @ leaf_emb.T
    mem = {}
    for tau in taus:
        mem[float(tau)] = assign_leaves(leaf_sim, float(tau))
    return {"leaf_sim": leaf_sim, "by_tau": mem}


def sweep_settings() -> list[dict]:
    settings = []
    for level in LEVELS:
        for lam in LAMBDAS:
            settings.append({"level": level, "lambda": lam})
    return settings
