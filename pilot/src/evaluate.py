"""Sweep, select settings on core-dev, and score frozen configs on each split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import K_LIST, N_BOOTSTRAP, SEED
from retrieve import (
    QueryPack,
    evaluate_config_on_queries,
    precompute_memberships,
    sweep_settings,
)
from taxonomy import Taxonomy, membership_for_level
from util import bootstrap_mean_ci, percentile


def assign_question_split(queries: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    out = queries.copy()
    core_qs = sorted(out.loc[out["set_type"] == "core", "question"].unique())
    rng = np.random.default_rng(seed)
    core_qs = list(core_qs)
    rng.shuffle(core_qs)
    if len(core_qs) != 16:
        print(
            f"WARNING: expected 16 core questions for the 8/8 split; found {len(core_qs)}. "
            "Splitting the first half as dev and the rest as test."
        )
    n_dev = len(core_qs) // 2
    dev = set(core_qs[:n_dev])
    test = set(core_qs[n_dev:])

    def _split(row: pd.Series) -> str:
        if row["set_type"] == "extra":
            return "extra"
        if row["question"] in dev:
            return "core_dev"
        if row["question"] in test:
            return "core_test"
        return "unassigned"

    out["question_split"] = out.apply(_split, axis=1)
    return out, {"core_dev": sorted(dev), "core_test": sorted(test), "seed": seed}


def _mask(pack: QueryPack, split: str, eligible_only: bool = True) -> np.ndarray:
    m = pack.question_split == split
    if eligible_only:
        m = m & pack.eval_eligible
    if split == "extra":
        # extra questions dropped for zero corpus-level gold are already eval_eligible=False
        # if every query had zero gold; remaining zero-gold queries are also excluded.
        pass
    return m


def _aggregate_rows(rows: list[dict], seed: int) -> dict[str, Any]:
    if not rows:
        return {
            "n_queries": 0,
            "note": "no eligible queries in this split/config",
        }
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n_queries": int(len(df))}
    for k in K_LIST:
        col = f"recall@{k}"
        out[col] = bootstrap_mean_ci(df[col].to_numpy(dtype=float), N_BOOTSTRAP, seed)
    out["miss_rate"] = bootstrap_mean_ci(df["miss_rate"].to_numpy(dtype=float), N_BOOTSTRAP, seed)
    out["selectivity"] = bootstrap_mean_ci(df["selectivity"].to_numpy(dtype=float), N_BOOTSTRAP, seed)
    lat = df["latency_ms"].to_numpy(dtype=float)
    out["latency_ms"] = {
        "median": percentile(lat, 50),
        "p95": percentile(lat, 95),
        "mean": float(lat.mean()) if lat.size else None,
        "n": int(lat.size),
    }
    topic = df["topic_scoring_ms"].to_numpy(dtype=float)
    out["topic_scoring_ms"] = {
        "median": percentile(topic, 50),
        "p95": percentile(topic, 95),
        "n": int(topic.size),
    }
    return out


def miss_attribution(b_rows: list[dict], e_rows: list[dict]) -> dict[str, Any]:
    e_by_q = {r["query_id"]: set(r["candidate_idx"].tolist()) for r in e_rows}
    routing = 0
    labeling = 0
    n_missed = 0
    n_queries_with_miss = 0
    for r in b_rows:
        missed = r["missed_gold_idx"]
        if missed.size == 0:
            continue
        n_queries_with_miss += 1
        e_cand = e_by_q.get(r["query_id"], set())
        for g in missed.tolist():
            n_missed += 1
            if g in e_cand:
                routing += 1
            else:
                labeling += 1
    return {
        "n_gold_missed_by_B": n_missed,
        "n_queries_where_B_missed_any_gold": n_queries_with_miss,
        "routing_error_gold_in_E": routing,
        "chunk_labeling_error_gold_missing_from_E": labeling,
        "definition": (
            "For gold chunks absent from B's candidate set: in E = routing error "
            "(query did not select the gold chunk's topic node); not in E = "
            "chunk-labeling error (gold chunk was not in the oracle topic nodes)."
        ),
    }


def _config_membership(
    config: str,
    level: str,
    leaf_mem: np.ndarray,
    tax: Taxonomy,
) -> np.ndarray | None:
    if config == "A":
        return None
    if config == "D":
        return membership_for_level(leaf_mem, tax, "leaf")
    return membership_for_level(leaf_mem, tax, level)


def _config_level_emb(
    config: str,
    level: str,
    mats: dict,
) -> np.ndarray | None:
    if config == "A" or config == "E":
        return None
    if config == "D":
        return mats["leaf"]
    return mats[level]


def run_one_setting(
    pack: QueryPack,
    mask: np.ndarray,
    chunk_emb: np.ndarray,
    tax: Taxonomy,
    mats: dict,
    leaf_mem: np.ndarray,
    level: str,
    lam: int,
    configs: tuple[str, ...] = ("A", "B", "D", "E"),
) -> dict[str, list[dict]]:
    out = {}
    for config in configs:
        mem = _config_membership(config, level, leaf_mem, tax)
        lemb = _config_level_emb(config, level, mats)
        out[config] = evaluate_config_on_queries(
            pack, mask, chunk_emb, config, mem, lemb, lam
        )
    return out


def select_best_on_dev(sweep_points: list[dict]) -> dict:
    """Maximize mean Recall@10; ties: lower miss_rate, then lower selectivity."""
    def key(p: dict) -> tuple:
        rec = p["metrics"]["B"]["recall@10"]["mean"]
        miss = p["metrics"]["B"]["miss_rate"]["mean"]
        sel = p["metrics"]["B"]["selectivity"]["mean"]
        rec = -1.0 if rec is None else rec
        miss = 2.0 if miss is None else miss
        sel = 2.0 if sel is None else sel
        return (-rec, miss, sel)

    ranked = sorted(sweep_points, key=key)
    best = ranked[0]
    return {
        "level": best["level"],
        "lambda": best["lambda"],
        "tau": best["tau"],
        "selection_rule": (
            "On core-dev only: maximize B mean Recall@10; ties broken by lower "
            "mean miss_rate, then lower mean selectivity."
        ),
        "core_dev_recall@10": best["metrics"]["B"]["recall@10"],
    }


def run_evaluation(
    pack: QueryPack,
    chunks: pd.DataFrame,
    chunk_emb: np.ndarray,
    tax: Taxonomy,
    node_mats: dict,
    taus: list[float],
    extra_dropped: list[str],
) -> dict[str, Any]:
    pre = precompute_memberships(chunk_emb, node_mats["leaf"], taus)
    splits = {
        "core_dev": _mask(pack, "core_dev", True),
        "core_test": _mask(pack, "core_test", True),
        "extra": _mask(pack, "extra", True),
    }
    extra_questions = sorted(
        {
            str(q)
            for q, st in zip(pack.questions, pack.set_type)
            if st == "extra" and q not in set(extra_dropped)
        }
    )
    print("\n========== EXTRA COVERAGE (evaluation gate) ==========")
    print(f"Extra questions retained: {extra_questions}")
    print(f"Extra questions dropped (zero gold across reports): {extra_dropped}")
    print(f"Eligible extra queries: {int(splits['extra'].sum())}")

    sweep_points = []
    print("\n========== SWEEP ON CORE-DEV ==========")
    for tau in taus:
        leaf_mem = pre["by_tau"][float(tau)]
        for setting in sweep_settings():
            raw = run_one_setting(
                pack,
                splits["core_dev"],
                chunk_emb,
                tax,
                node_mats,
                leaf_mem,
                setting["level"],
                setting["lambda"],
            )
            metrics = {cfg: _aggregate_rows(raw[cfg], SEED) for cfg in raw}
            point = {
                "level": setting["level"],
                "lambda": setting["lambda"],
                "tau": float(tau),
                "metrics": metrics,
                "n_core_dev_queries": int(splits["core_dev"].sum()),
            }
            sweep_points.append(point)
            b = metrics["B"]["recall@10"]
            print(
                f"  level={setting['level']} λ={setting['lambda']} τ={tau:.4f} "
                f"B Recall@10={b['mean']} sel={metrics['B']['selectivity']['mean']}"
            )

    chosen = select_best_on_dev(sweep_points)
    print(
        f"\nFrozen setting from core-dev: level={chosen['level']} "
        f"λ={chosen['lambda']} τ={chosen['tau']}"
    )

    leaf_mem = pre["by_tau"][float(chosen["tau"])]
    frozen_raw: dict[str, dict[str, list[dict]]] = {}
    frozen_metrics: dict[str, dict[str, Any]] = {}
    attributions: dict[str, Any] = {}
    for split_name, mask in splits.items():
        raw = run_one_setting(
            pack,
            mask,
            chunk_emb,
            tax,
            node_mats,
            leaf_mem,
            chosen["level"],
            chosen["lambda"],
        )
        frozen_raw[split_name] = raw
        frozen_metrics[split_name] = {cfg: _aggregate_rows(raw[cfg], SEED) for cfg in raw}
        attributions[split_name] = miss_attribution(raw["B"], raw["E"])
        print(
            f"Frozen eval {split_name}: n={int(mask.sum())} "
            f"B Recall@10={frozen_metrics[split_name]['B'].get('recall@10', {}).get('mean')}"
        )

    per_query = []
    for split_name, raw in frozen_raw.items():
        for cfg, rows in raw.items():
            for r in rows:
                per_query.append(
                    {
                        "split": split_name,
                        "config": cfg,
                        "query_id": r["query_id"],
                        "question": r["question"],
                        "report": r["report"],
                        "set_type": r["set_type"],
                        "recall@5": r["recall@5"],
                        "recall@10": r["recall@10"],
                        "recall@20": r["recall@20"],
                        "miss_rate": r["miss_rate"],
                        "selectivity": r["selectivity"],
                        "latency_ms": r["latency_ms"],
                        "n_candidates": r["n_candidates"],
                        "n_gold": r["n_gold"],
                    }
                )

    return {
        "chosen_setting": chosen,
        "sweep_points": sweep_points,
        "frozen_metrics": frozen_metrics,
        "miss_attribution": attributions,
        "per_query": per_query,
        "split_counts": {k: int(v.sum()) for k, v in splits.items()},
        "leaf_membership_chosen": leaf_mem,
        "leaf_sim": pre["leaf_sim"],
    }
