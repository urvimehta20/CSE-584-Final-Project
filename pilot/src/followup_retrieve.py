"""Retrieval configs for follow-up settings R (per-report) and G (corpus-wide)."""

from __future__ import annotations

import time

import numpy as np

from followup_common import KS_FOLLOWUP, capped_recall, miss_rate, raw_recall
from taxonomy import route_top_lambda


def rank_by_cosine(q_emb: np.ndarray, chunk_emb: np.ndarray, cand: np.ndarray) -> np.ndarray:
    if cand.size == 0:
        return cand
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = chunk_emb[cand] @ q_emb
    return cand[np.argsort(-scores, kind="stable")]


def route_indices(q_emb: np.ndarray, node_emb: np.ndarray, lam: int) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = node_emb @ q_emb
    return route_top_lambda(sims, lam)


def candidates_from_nodes(universe: np.ndarray, membership: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    if nodes.size == 0 or universe.size == 0:
        return np.array([], dtype=int)
    hit = membership[universe][:, nodes].any(axis=1)
    return universe[hit]


def d_matched_candidates(
    universe: np.ndarray,
    leaf_mem: np.ndarray,
    q_leaf_sim: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add leaves by query score until |candidates| first reaches target_count."""
    if target_count <= 0 or universe.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-q_leaf_sim, kind="stable")
    selected: list[int] = []
    cand = np.array([], dtype=int)
    for li in order.tolist():
        selected.append(int(li))
        cand = candidates_from_nodes(universe, leaf_mem, np.array(selected, dtype=int))
        if cand.size >= target_count:
            break
    return cand, np.array(selected, dtype=int)


def run_config(
    q_emb: np.ndarray,
    chunk_emb: np.ndarray,
    universe: np.ndarray,
    gold: np.ndarray,
    config: str,
    leaf_mem: np.ndarray,
    pillar_mem: np.ndarray,
    leaf_emb: np.ndarray,
    pillar_emb: np.ndarray,
    lam: int,
    b_pillar_cand: np.ndarray | None = None,
) -> dict:
    t0 = time.perf_counter()
    routed = np.array([], dtype=int)
    if config == "A":
        cand = universe
    elif config == "B_leaf":
        routed = route_indices(q_emb, leaf_emb, lam)
        cand = candidates_from_nodes(universe, leaf_mem, routed)
    elif config == "B_pillar":
        routed = route_indices(q_emb, pillar_emb, lam)
        cand = candidates_from_nodes(universe, pillar_mem, routed)
    elif config == "D_matched":
        if b_pillar_cand is None:
            raise ValueError("D_matched needs b_pillar_cand")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            q_leaf_sim = leaf_emb @ q_emb
        cand, routed = d_matched_candidates(universe, leaf_mem, q_leaf_sim, int(b_pillar_cand.size))
    else:
        raise ValueError(config)
    ranked = rank_by_cosine(q_emb, chunk_emb, cand)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    n_univ = int(universe.size)
    out = {
        "config": config,
        "lambda": lam,
        "n_candidates": int(ranked.size),
        "n_universe": n_univ,
        "selectivity": None if n_univ == 0 else float(ranked.size / n_univ),
        "miss_rate": miss_rate(gold, ranked),
        "latency_ms": float(latency_ms),
        "routed_node_indices": routed,
        "ranked": ranked,
        "n_gold": int(gold.size),
        "n_gold_in_candidates": int(np.intersect1d(gold, ranked).size) if gold.size else 0,
    }
    for k in KS_FOLLOWUP:
        out[f"recall@{k}"] = raw_recall(gold, ranked, k)
        out[f"capped_recall@{k}"] = capped_recall(gold, ranked, k)
    return out
