"""Run follow-up experiments 1–3 and write followup_results.json, figures, REPORT.md."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import EMB_DIR, FIGURES, GOLD_THRESHOLD, PROCESSED, RESULTS, SEED
from evaluate import assign_question_split
from followup_common import (
    KS_FOLLOWUP,
    LEAF_LAMBDAS,
    N_BOOT_FOLLOWUP,
    PILLAR_LAMBDAS,
    PRIMARY_K,
    PRIMARY_R,
    RECALL_TARGETS,
    SMALL_GOLD_QUESTIONS,
    TOKEN_BUDGETS,
    TOKEN_OVERHEAD,
    chunk_token_costs,
    cluster_bootstrap_mean,
    core_folds,
    gold_chunk_indices,
    load_tokenizer,
    paired_cluster_diff,
    recall_at_token_budget,
    tokens_until_recall,
)
from followup_retrieve import run_config
from setting_g_stats import compute_setting_g_question_stats
from taxonomy import (
    assign_leaves,
    build_taxonomy,
    choose_taus,
    chunk_assignment_stats,
    leaf_to_pillar_membership,
    node_matrices,
)
from util import collect_environment, json_safe, percentile
from write_followup_report import write_figures, write_report


PRIMARY_METRIC = f"capped_recall@{PRIMARY_K}"


def _load_data() -> dict:
    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    qrels = pd.read_parquet(PROCESSED / "qrels.parquet")
    queries, split_info = assign_question_split(queries, SEED)
    chunk_emb = np.load(EMB_DIR / "chunk_embeddings.npy")
    question_emb = np.load(EMB_DIR / "question_embeddings.npy")
    question_texts = json.loads((EMB_DIR / "question_texts.json").read_text())
    qmap = {q: i for i, q in enumerate(question_texts)}
    query_emb = question_emb[np.array([qmap[q] for q in queries["question"]], dtype=int)]
    node_emb = np.load(EMB_DIR / "node_embeddings.npy")
    tax = build_taxonomy()
    mats = node_matrices(tax, node_emb)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        leaf_sim = chunk_emb @ mats["leaf"].T
    taus, tau_dist = choose_taus(leaf_sim)
    chunk_pos = {cid: i for i, cid in enumerate(chunks["chunk_id"].tolist())}
    gold = gold_chunk_indices(queries, qrels, chunk_pos)
    report_idx = {
        report: np.array([chunk_pos[c] for c in grp["chunk_id"]], dtype=int)
        for report, grp in chunks.groupby("report", sort=False)
    }
    universe_all = np.arange(len(chunks), dtype=int)
    return {
        "chunks": chunks,
        "queries": queries,
        "qrels": qrels,
        "split_info": split_info,
        "chunk_emb": chunk_emb,
        "question_emb": question_emb,
        "question_texts": question_texts,
        "qmap": qmap,
        "query_emb": query_emb,
        "tax": tax,
        "mats": mats,
        "leaf_sim": leaf_sim,
        "taus": taus,
        "tau_dist": tau_dist,
        "gold": gold,
        "report_idx": report_idx,
        "universe_all": universe_all,
        "chunk_pos": chunk_pos,
    }


def _mem_for_tau(leaf_sim: np.ndarray, tax, tau: float) -> tuple[np.ndarray, np.ndarray]:
    leaf_mem = assign_leaves(leaf_sim, float(tau))
    return leaf_mem, leaf_to_pillar_membership(leaf_mem, tax)


def _score_one(
    q_emb: np.ndarray,
    gold: np.ndarray,
    universe: np.ndarray,
    data: dict,
    tau: float,
    config: str,
    lam: int,
    b_pillar_cand: np.ndarray | None = None,
    mem_cache: dict | None = None,
) -> dict:
    if mem_cache is None or tau not in mem_cache:
        mem_cache = mem_cache or {}
        mem_cache[tau] = _mem_for_tau(data["leaf_sim"], data["tax"], tau)
    leaf_mem, pillar_mem = mem_cache[tau]
    return run_config(
        q_emb=q_emb,
        chunk_emb=data["chunk_emb"],
        universe=universe,
        gold=gold,
        config=config,
        leaf_mem=leaf_mem,
        pillar_mem=pillar_mem,
        leaf_emb=data["mats"]["leaf"],
        pillar_emb=data["mats"]["pillar"],
        lam=lam,
        b_pillar_cand=b_pillar_cand,
    )


def _units_g(data: dict, questions: list[str]) -> list[dict]:
    out = []
    qset = data["queries"][["question", "set_type"]].drop_duplicates()
    st = dict(zip(qset["question"], qset["set_type"]))
    for q in questions:
        gold = data["gold"]["by_question"].get(q, np.array([], dtype=int))
        if gold.size == 0:
            continue
        out.append(
            {
                "unit_id": q,
                "question": q,
                "set_type": st[q],
                "q_emb": data["question_emb"][data["qmap"][q]],
                "gold": gold,
                "universe": data["universe_all"],
            }
        )
    return out


def _units_r(data: dict, questions: list[str], eligible_only: bool = True) -> list[dict]:
    out = []
    qdf = data["queries"]
    for i, row in qdf.iterrows():
        if row["question"] not in questions:
            continue
        if row["set_type"] != "core" and "extra" not in questions:
            pass
        gold = data["gold"]["by_query"][row["query_id"]]
        if eligible_only and gold.size == 0:
            continue
        out.append(
            {
                "unit_id": row["query_id"],
                "question": row["question"],
                "report": row["report"],
                "set_type": row["set_type"],
                "q_emb": data["query_emb"][i],
                "gold": gold,
                "universe": data["report_idx"][row["report"]],
            }
        )
    return out


def _eval_units(units: list[dict], data: dict, tau: float, setting: str, frozen: dict) -> list[dict]:
    mem_cache: dict = {}
    rows = []
    for u in units:
        recs = {}
        recs["A"] = _score_one(u["q_emb"], u["gold"], u["universe"], data, tau, "A", 0, mem_cache=mem_cache)
        recs["B_leaf"] = _score_one(
            u["q_emb"], u["gold"], u["universe"], data, tau, "B_leaf", frozen["lambda_leaf"], mem_cache=mem_cache
        )
        recs["B_pillar"] = _score_one(
            u["q_emb"], u["gold"], u["universe"], data, tau, "B_pillar", frozen["lambda_pillar"], mem_cache=mem_cache
        )
        recs["D_matched"] = _score_one(
            u["q_emb"],
            u["gold"],
            u["universe"],
            data,
            tau,
            "D_matched",
            frozen["lambda_pillar"],
            b_pillar_cand=recs["B_pillar"]["ranked"],
            mem_cache=mem_cache,
        )
        for cfg, rec in recs.items():
            rows.append(
                {
                    "setting": setting,
                    "unit_id": u["unit_id"],
                    "question": u["question"],
                    "set_type": u["set_type"],
                    "report": u.get("report"),
                    "config": cfg,
                    "tau": tau,
                    "lambda": rec["lambda"],
                    "n_gold": rec["n_gold"],
                    "n_candidates": rec["n_candidates"],
                    "n_universe": rec["n_universe"],
                    "selectivity": rec["selectivity"],
                    "miss_rate": rec["miss_rate"],
                    "latency_ms": rec["latency_ms"],
                    "recall@20": rec["recall@20"],
                    "recall@50": rec["recall@50"],
                    "recall@100": rec["recall@100"],
                    "capped_recall@20": rec["capped_recall@20"],
                    "capped_recall@50": rec["capped_recall@50"],
                    "capped_recall@100": rec["capped_recall@100"],
                    "ranked": rec["ranked"],
                    "routed_node_indices": rec["routed_node_indices"],
                    "gold": u["gold"],
                    "universe": u["universe"],
                    "q_emb": u["q_emb"],
                }
            )
    return rows


def _tune(units: list[dict], data: dict, setting: str) -> dict:
    mem_cache: dict = {}
    best_leaf = None
    best_pillar = None
    sweep = []
    for tau in data["taus"]:
        for lam in LEAF_LAMBDAS:
            vals = []
            for u in units:
                rec = _score_one(u["q_emb"], u["gold"], u["universe"], data, tau, "B_leaf", lam, mem_cache=mem_cache)
                if rec[PRIMARY_METRIC] is not None:
                    vals.append(rec[PRIMARY_METRIC])
            mean = float(np.mean(vals)) if vals else -1.0
            sweep.append({"config": "B_leaf", "tau": tau, "lambda": lam, "tune_capped_recall@50": mean, "n": len(vals)})
            cand = {"tau": tau, "lambda": lam, "score": mean}
            if best_leaf is None or (mean, -lam, -tau) > (best_leaf["score"], -best_leaf["lambda"], -best_leaf["tau"]):
                best_leaf = cand
        for lam in PILLAR_LAMBDAS:
            vals = []
            for u in units:
                rec = _score_one(u["q_emb"], u["gold"], u["universe"], data, tau, "B_pillar", lam, mem_cache=mem_cache)
                if rec[PRIMARY_METRIC] is not None:
                    vals.append(rec[PRIMARY_METRIC])
            mean = float(np.mean(vals)) if vals else -1.0
            sweep.append({"config": "B_pillar", "tau": tau, "lambda": lam, "tune_capped_recall@50": mean, "n": len(vals)})
            cand = {"tau": tau, "lambda": lam, "score": mean}
            if best_pillar is None or (mean, -lam, -tau) > (best_pillar["score"], -best_pillar["lambda"], -best_pillar["tau"]):
                best_pillar = cand
    # Shared τ: the τ of the better primary B on the tune fold; if split, keep each B's own τ
    # and use B-pillar's τ for D_matched and for a single "frozen tau" used when evaluating all
    # configs together. A does not depend on τ. B_leaf uses its own τ.
    frozen = {
        "tau_leaf": best_leaf["tau"],
        "lambda_leaf": best_leaf["lambda"],
        "tune_score_leaf": best_leaf["score"],
        "tau_pillar": best_pillar["tau"],
        "lambda_pillar": best_pillar["lambda"],
        "tune_score_pillar": best_pillar["score"],
        "selection_rule": (
            f"On the tuning fold, maximize mean {PRIMARY_METRIC}. "
            "B_leaf and B_pillar are tuned separately. D_matched is matched to B_pillar's "
            "candidate count at B_pillar's (τ, λ_p)."
        ),
        "sweep": sweep,
    }
    print(
        f"  {setting} tune: B_leaf τ={frozen['tau_leaf']:.4f} λ={frozen['lambda_leaf']} "
        f"{PRIMARY_METRIC}={frozen['tune_score_leaf']:.4f}; "
        f"B_pillar τ={frozen['tau_pillar']:.4f} λ={frozen['lambda_pillar']} "
        f"{PRIMARY_METRIC}={frozen['tune_score_pillar']:.4f}"
    )
    return frozen


def _eval_with_own_taus(units: list[dict], data: dict, setting: str, frozen: dict) -> list[dict]:
    """Evaluate A (any τ), B_leaf at its τ, B_pillar/D_matched at pillar τ."""
    mem_cache: dict = {}
    rows = []
    for u in units:
        rec_a = _score_one(u["q_emb"], u["gold"], u["universe"], data, frozen["tau_pillar"], "A", 0, mem_cache=mem_cache)
        rec_bl = _score_one(
            u["q_emb"], u["gold"], u["universe"], data, frozen["tau_leaf"], "B_leaf", frozen["lambda_leaf"], mem_cache=mem_cache
        )
        rec_bp = _score_one(
            u["q_emb"], u["gold"], u["universe"], data, frozen["tau_pillar"], "B_pillar", frozen["lambda_pillar"], mem_cache=mem_cache
        )
        rec_d = _score_one(
            u["q_emb"],
            u["gold"],
            u["universe"],
            data,
            frozen["tau_pillar"],
            "D_matched",
            frozen["lambda_pillar"],
            b_pillar_cand=rec_bp["ranked"],
            mem_cache=mem_cache,
        )
        for cfg, rec, tau in (
            ("A", rec_a, frozen["tau_pillar"]),
            ("B_leaf", rec_bl, frozen["tau_leaf"]),
            ("B_pillar", rec_bp, frozen["tau_pillar"]),
            ("D_matched", rec_d, frozen["tau_pillar"]),
        ):
            rows.append(
                {
                    "setting": setting,
                    "unit_id": u["unit_id"],
                    "question": u["question"],
                    "set_type": u["set_type"],
                    "report": u.get("report"),
                    "config": cfg,
                    "tau": tau,
                    "lambda": rec["lambda"],
                    "n_gold": rec["n_gold"],
                    "n_candidates": rec["n_candidates"],
                    "n_universe": rec["n_universe"],
                    "selectivity": rec["selectivity"],
                    "miss_rate": rec["miss_rate"],
                    "latency_ms": rec["latency_ms"],
                    "recall@20": rec["recall@20"],
                    "recall@50": rec["recall@50"],
                    "recall@100": rec["recall@100"],
                    "capped_recall@20": rec["capped_recall@20"],
                    "capped_recall@50": rec["capped_recall@50"],
                    "capped_recall@100": rec["capped_recall@100"],
                    "ranked": rec["ranked"],
                    "routed_node_indices": rec["routed_node_indices"],
                    "gold": u["gold"],
                    "universe": u["universe"],
                    "q_emb": u["q_emb"],
                }
            )
    return rows


def _aggregate(rows: list[dict], configs: tuple[str, ...], average: str) -> dict[str, Any]:
    by_cfg: dict[str, Any] = {}
    metric_keys = (
        [f"recall@{k}" for k in KS_FOLLOWUP]
        + [f"capped_recall@{k}" for k in KS_FOLLOWUP]
        + ["miss_rate", "selectivity"]
    )
    for cfg in configs:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            by_cfg[cfg] = {"n_units": 0, "note": "no rows"}
            continue
        qs = [r["question"] for r in sub]
        block: dict[str, Any] = {
            "n_units": len(sub),
            "n_questions": len({r["question"] for r in sub}),
        }
        for key in metric_keys:
            vals = np.array([r[key] if r[key] is not None else np.nan for r in sub], dtype=float)
            block[key] = cluster_bootstrap_mean(qs, vals, N_BOOT_FOLLOWUP, SEED, average)
        lat = np.array([r["latency_ms"] for r in sub], dtype=float)
        block["latency_ms"] = {"median": percentile(lat, 50), "p95": percentile(lat, 95), "n": int(lat.size)}
        by_cfg[cfg] = block
    diffs = {}
    for cfg in ("B_leaf", "B_pillar", "D_matched"):
        a = [r for r in rows if r["config"] == "A"]
        b = [r for r in rows if r["config"] == cfg]
        if not a or not b:
            continue
        a = sorted(a, key=lambda r: r["unit_id"])
        b = sorted(b, key=lambda r: r["unit_id"])
        assert [r["unit_id"] for r in a] == [r["unit_id"] for r in b]
        diffs[f"{cfg}_minus_A"] = {
            PRIMARY_METRIC: paired_cluster_diff(
                [r["question"] for r in a],
                np.array([r[PRIMARY_METRIC] for r in a], dtype=float),
                np.array([r[PRIMARY_METRIC] for r in b], dtype=float),
                N_BOOT_FOLLOWUP,
                SEED,
                average,
            )
        }
    return {"configs": by_cfg, "paired_diffs": diffs}


def _best_b(agg: dict) -> str:
    leaf = agg["configs"].get("B_leaf", {}).get(PRIMARY_METRIC, {}).get("mean")
    pillar = agg["configs"].get("B_pillar", {}).get(PRIMARY_METRIC, {}).get("mean")
    if leaf is None and pillar is None:
        return "B_pillar"
    if leaf is None:
        return "B_pillar"
    if pillar is None:
        return "B_leaf"
    if pillar >= leaf:
        return "B_pillar"
    return "B_leaf"


def experiment_1(data: dict, folds: dict) -> dict:
    print("\n========== EXPERIMENT 1: retrieval R and G ==========")
    results: dict[str, Any] = {"folds": {}, "pooled_heldout": {}, "original_core_test": {}, "extra": {}}
    all_heldout = {"R": [], "G": []}
    fold1_frozen = {}
    for fold in folds["folds"]:
        print(f"Fold {fold['fold_id']}: {fold['label']}")
        fold_out = {"tune_questions": fold["tune"], "test_questions": fold["test"]}
        for setting, unit_fn in (("G", _units_g), ("R", _units_r)):
            tune_units = unit_fn(data, fold["tune"])
            test_units = unit_fn(data, fold["test"])
            frozen = _tune(tune_units, data, setting)
            test_rows = _eval_with_own_taus(test_units, data, setting, frozen)
            average = "question" if setting == "G" else "unit"
            agg = _aggregate(test_rows, ("A", "B_leaf", "B_pillar", "D_matched"), average)
            fold_out[setting] = {"frozen": frozen, "heldout": agg, "n_tune_units": len(tune_units), "n_test_units": len(test_units)}
            for r in test_rows:
                r["fold_id"] = fold["fold_id"]
            all_heldout[setting].extend(test_rows)
            if fold["fold_id"] == 0:
                fold1_frozen[setting] = frozen
                # original core-test = this fold's test
                results["original_core_test"][setting] = agg
        results["folds"][fold["fold_id"]] = fold_out

    for setting in ("R", "G"):
        average = "question" if setting == "G" else "unit"
        results["pooled_heldout"][setting] = _aggregate(
            all_heldout[setting], ("A", "B_leaf", "B_pillar", "D_matched"), average
        )
        results["pooled_heldout"][setting]["frozen_best_B"] = _best_b(results["pooled_heldout"][setting])
        small = [r for r in all_heldout[setting] if r["question"] in SMALL_GOLD_QUESTIONS]
        results["pooled_heldout"][setting]["small_gold_questions"] = _aggregate(
            small, ("A", "B_leaf", "B_pillar", "D_matched"), average
        )
        results["pooled_heldout"][setting]["small_gold_question_list"] = list(SMALL_GOLD_QUESTIONS)

    extra_qs = sorted(data["queries"].loc[data["queries"]["set_type"] == "extra", "question"].unique())
    for setting, unit_fn in (("G", _units_g), ("R", _units_r)):
        units = unit_fn(data, extra_qs)
        rows = _eval_with_own_taus(units, data, setting, fold1_frozen[setting])
        average = "question" if setting == "G" else "unit"
        results["extra"][setting] = _aggregate(rows, ("A", "B_leaf", "B_pillar", "D_matched"), average)
        results["extra"][setting]["n_units"] = len(units)
        results["extra"][setting]["note"] = (
            "Secondary. Frozen settings from fold 0 (original core-dev tune). Never pooled with core."
        )
    results["heldout_rows"] = all_heldout
    results["fold1_frozen"] = fold1_frozen
    return results


def experiment_2(data: dict, exp1: dict, costs: np.ndarray, tokenizer_name: str) -> dict:
    print("\n========== EXPERIMENT 2: prompt tokens ==========")
    out: dict[str, Any] = {"tokenizer": tokenizer_name, "per_chunk_overhead": TOKEN_OVERHEAD, "settings": {}}
    for setting in ("R", "G"):
        rows = exp1["heldout_rows"][setting]
        average = "question" if setting == "G" else "unit"
        by_unit: dict[tuple[str, str], dict] = {}
        for r in rows:
            by_unit[(r["unit_id"], r["config"])] = r
        unit_ids = sorted({r["unit_id"] for r in rows})
        configs = ("A", "B_leaf", "B_pillar", "D_matched")
        token_rows = []
        for uid in unit_ids:
            base = by_unit[(uid, "A")]
            gold = base["gold"]
            for cfg in configs:
                rec = by_unit[(uid, cfg)]
                item = {
                    "unit_id": uid,
                    "question": rec["question"],
                    "config": cfg,
                    "n_gold": rec["n_gold"],
                }
                for r in RECALL_TARGETS:
                    tr = tokens_until_recall(rec["ranked"], gold, costs, r)
                    item[f"tokens_r{r}"] = tr["tokens"]
                    item[f"reachable_r{r}"] = tr["reachable"]
                for b in TOKEN_BUDGETS:
                    item[f"recall_budget_{b}"] = recall_at_token_budget(rec["ranked"], gold, costs, b)
                token_rows.append(item)

        setting_block: dict[str, Any] = {"configs": {}, "paired_vs_A": {}, "reachable": {}, "budget_recall": {}}
        for cfg in configs:
            sub = [t for t in token_rows if t["config"] == cfg]
            qs = [t["question"] for t in sub]
            setting_block["reachable"][cfg] = {}
            setting_block["configs"][cfg] = {}
            for r in RECALL_TARGETS:
                reach = np.array([1.0 if t[f"reachable_r{r}"] else 0.0 for t in sub])
                setting_block["reachable"][cfg][str(r)] = cluster_bootstrap_mean(qs, reach, N_BOOT_FOLLOWUP, SEED, average)
                toks = np.array(
                    [t[f"tokens_r{r}"] if t[f"reachable_r{r}"] else np.nan for t in sub], dtype=float
                )
                setting_block["configs"][cfg][f"tokens_r{r}_reachable_only"] = cluster_bootstrap_mean(
                    qs, toks, N_BOOT_FOLLOWUP, SEED, average
                )
            setting_block["budget_recall"][cfg] = {}
            for b in TOKEN_BUDGETS:
                vals = np.array([t[f"recall_budget_{b}"] for t in sub], dtype=float)
                setting_block["budget_recall"][cfg][str(b)] = cluster_bootstrap_mean(
                    qs, vals, N_BOOT_FOLLOWUP, SEED, average
                )

        for cfg in ("B_leaf", "B_pillar", "D_matched"):
            setting_block["paired_vs_A"][cfg] = {}
            for r in RECALL_TARGETS:
                pairs = []
                for uid in unit_ids:
                    a = by_unit[(uid, "A")]
                    b = by_unit[(uid, cfg)]
                    ta = tokens_until_recall(a["ranked"], a["gold"], costs, r)
                    tb = tokens_until_recall(b["ranked"], b["gold"], costs, r)
                    if ta["reachable"] and tb["reachable"] and ta["tokens"] and ta["tokens"] > 0:
                        pairs.append(
                            {
                                "question": a["question"],
                                "tokens_A": ta["tokens"],
                                "tokens_B": tb["tokens"],
                                "rel_saving": (ta["tokens"] - tb["tokens"]) / ta["tokens"],
                            }
                        )
                if not pairs:
                    setting_block["paired_vs_A"][cfg][str(r)] = {
                        "n_paired": 0,
                        "note": "no units where both configs reach r",
                        "relative_saving": {"mean": None, "ci95_low": None, "ci95_high": None},
                        "rule_ge_15pct": False,
                        "ci_excludes_0": False,
                    }
                    continue
                qs = [p["question"] for p in pairs]
                rel = np.array([p["rel_saving"] for p in pairs], dtype=float)
                rel_sum = cluster_bootstrap_mean(qs, rel, N_BOOT_FOLLOWUP, SEED, average)
                tok_diff = paired_cluster_diff(
                    qs,
                    np.array([p["tokens_A"] for p in pairs], dtype=float),
                    np.array([p["tokens_B"] for p in pairs], dtype=float),
                    N_BOOT_FOLLOWUP,
                    SEED,
                    average,
                )
                mean_rel = rel_sum["mean"]
                setting_block["paired_vs_A"][cfg][str(r)] = {
                    "n_paired": len(pairs),
                    "n_questions": len(set(qs)),
                    "tokens_A": cluster_bootstrap_mean(qs, np.array([p["tokens_A"] for p in pairs], dtype=float), N_BOOT_FOLLOWUP, SEED, average),
                    "tokens_B": cluster_bootstrap_mean(qs, np.array([p["tokens_B"] for p in pairs], dtype=float), N_BOOT_FOLLOWUP, SEED, average),
                    "relative_saving": rel_sum,
                    "token_diff_B_minus_A": tok_diff,
                    "rule_ge_15pct": bool(mean_rel is not None and mean_rel >= 0.15),
                    "ci_excludes_0": bool(
                        rel_sum["ci95_low"] is not None
                        and rel_sum["ci95_high"] is not None
                        and (rel_sum["ci95_high"] < 0 or rel_sum["ci95_low"] > 0)
                    ),
                }
        setting_block["note_unjudged"] = (
            "In setting G, tokens on unjudged chunks count as cost and may include unlabeled relevant text, "
            "so every config's token cost is an upper bound."
            if setting == "G"
            else "Setting R only retrieves within the annotated report."
        )
        out["settings"][setting] = setting_block
        out.setdefault("token_rows", {})[setting] = token_rows
    out["primary_r"] = PRIMARY_R
    return out


def experiment_3(data: dict, exp1: dict) -> dict:
    print("\n========== EXPERIMENT 3: pruned gold under A ==========")
    tax = data["tax"]
    out: dict[str, Any] = {"settings": {}}
    pruned_top10_rows = []
    gold_rows_all = []
    for setting in ("R", "G"):
        best = exp1["pooled_heldout"][setting]["frozen_best_B"]
        rows = exp1["heldout_rows"][setting]
        by_unit = {}
        for r in rows:
            by_unit.setdefault(r["unit_id"], {})[r["config"]] = r
        records = []
        for uid, recs in by_unit.items():
            a = recs["A"]
            b = recs[best]
            gold = a["gold"]
            if gold.size == 0:
                continue
            a_rank = {int(idx): rank + 1 for rank, idx in enumerate(a["ranked"].tolist())}
            b_cand = set(b["ranked"].tolist())
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                cos = data["chunk_emb"][gold] @ a["q_emb"]
            n_univ = a["n_universe"]
            routed = b["routed_node_indices"]
            if best == "B_leaf":
                routed_names = [tax.leaf_ids[i] for i in routed.tolist()]
            else:
                routed_names = [tax.pillar_ids[i] for i in routed.tolist()]
            for j, idx in enumerate(gold.tolist()):
                rel_key = (
                    ("__question__", a["question"], idx)
                    if setting == "G"
                    else (uid, idx)
                )
                rel = data["gold"]["relevance"].get(rel_key)
                if rel is None:
                    # fall back to any recorded relevance for this chunk/question
                    rel = data["gold"]["relevance"].get(("__question__", a["question"], idx))
                rank = a_rank.get(int(idx))
                pruned = int(idx) not in b_cand
                rec = {
                    "setting": setting,
                    "unit_id": uid,
                    "question": a["question"],
                    "chunk_idx": int(idx),
                    "chunk_id": data["chunks"].iloc[int(idx)]["chunk_id"],
                    "report": data["chunks"].iloc[int(idx)]["report"],
                    "text": data["chunks"].iloc[int(idx)]["text"],
                    "relevance": rel,
                    "pruned_by_B": pruned,
                    "A_rank": rank,
                    "A_rank_percentile": None if rank is None else float(rank / n_univ),
                    "cosine_to_query": float(cos[j]),
                    "B_config": best,
                    "B_routed_topics": ";".join(routed_names),
                    "n_universe": n_univ,
                }
                records.append(rec)
                gold_rows_all.append(rec)
                if pruned and rank is not None and rank <= 10:
                    pruned_top10_rows.append(rec)

        rec_df = pd.DataFrame(records)
        pruned = rec_df.loc[rec_df["pruned_by_B"]]
        kept = rec_df.loc[~rec_df["pruned_by_B"]]
        block: dict[str, Any] = {
            "frozen_best_B": best,
            "n_gold": int(len(rec_df)),
            "n_pruned": int(len(pruned)),
            "n_kept": int(len(kept)),
        }
        for name, part in (("pruned", pruned), ("kept", kept)):
            if part.empty:
                block[name] = {"n": 0, "median_A_rank": None, "median_A_rank_percentile": None}
                continue
            block[name] = {
                "n": int(len(part)),
                "median_A_rank": float(part["A_rank"].median()) if part["A_rank"].notna().any() else None,
                "median_A_rank_percentile": float(part["A_rank_percentile"].median())
                if part["A_rank_percentile"].notna().any()
                else None,
                "share_in_A_top10": float((part["A_rank"] <= 10).mean()),
                "share_in_A_top20": float((part["A_rank"] <= 20).mean()),
                "n_relevance_2": int((part["relevance"] == 2).sum()),
                "n_relevance_3": int((part["relevance"] == 3).sum()),
                "n_relevance_other_or_missing": int((~part["relevance"].isin([2, 3])).sum()),
            }
        if len(pruned) and len(kept) and pruned["A_rank_percentile"].notna().any() and kept["A_rank_percentile"].notna().any():
            from scipy.stats import mannwhitneyu

            u_stat, p = mannwhitneyu(
                pruned["A_rank_percentile"].dropna().to_numpy(),
                kept["A_rank_percentile"].dropna().to_numpy(),
                alternative="greater",
            )
            block["mannwhitneyu"] = {
                "alternative": "pruned_percentile > kept_percentile",
                "U": float(u_stat),
                "p_value": float(p),
                "n_pruned": int(pruned["A_rank_percentile"].notna().sum()),
                "n_kept": int(kept["A_rank_percentile"].notna().sum()),
            }
        else:
            block["mannwhitneyu"] = {
                "p_value": None,
                "note": "not enough pruned and kept gold with A-ranks to run Mann–Whitney U",
            }
        n_pruned = max(int(len(pruned)), 0)
        n_pruned_top10 = int(((pruned["A_rank"] <= 10).sum()) if len(pruned) else 0)
        block["share_of_pruned_gold_in_A_top10"] = None if n_pruned == 0 else float(n_pruned_top10 / n_pruned)
        block["n_pruned_gold_in_A_top10"] = n_pruned_top10
        out["settings"][setting] = block
        print(
            f"  {setting} best B={best}: gold={block['n_gold']} pruned={block['n_pruned']} "
            f"pruned-in-A-top10={n_pruned_top10} MW p={block['mannwhitneyu'].get('p_value')}"
        )

    top10_df = pd.DataFrame(pruned_top10_rows)
    cols = [
        "setting",
        "unit_id",
        "question",
        "chunk_id",
        "report",
        "text",
        "relevance",
        "A_rank",
        "A_rank_percentile",
        "cosine_to_query",
        "B_config",
        "B_routed_topics",
    ]
    if top10_df.empty:
        top10_df = pd.DataFrame(columns=cols)
    else:
        top10_df = top10_df[cols]
    top10_path = RESULTS / "pruned_top10_gold.csv"
    top10_df.to_csv(top10_path, index=False)
    print(f"Wrote {top10_path} rows={len(top10_df)}")
    gold_df = pd.DataFrame(gold_rows_all)
    gold_df.drop(columns=["text"]).to_csv(RESULTS / "followup_gold_ranks.csv", index=False)
    out["n_pruned_top10_listed"] = int(len(top10_df))
    out["gold_rank_rows"] = gold_rows_all
    return out


def _rule_eval(exp1: dict, exp2: dict, exp3: dict) -> dict:
    rules = {}
    # Retrieval: B vs A on capped R@50. Evaluate each B and each setting; verdict uses frozen-best B in G.
    for setting in ("R", "G"):
        best = exp1["pooled_heldout"][setting]["frozen_best_B"]
        diffs = exp1["pooled_heldout"][setting]["paired_diffs"]
        rules[f"B_beats_A_retrieval_{setting}"] = {}
        for cfg in ("B_leaf", "B_pillar"):
            d = diffs.get(f"{cfg}_minus_A", {}).get(PRIMARY_METRIC, {})
            rules[f"B_beats_A_retrieval_{setting}"][cfg] = {
                "mean_diff": d.get("mean_diff"),
                "ci95_low": d.get("ci95_low"),
                "ci95_high": d.get("ci95_high"),
                "ci_entirely_above_0": d.get("ci_entirely_above_0"),
                "MET": bool(d.get("ci_entirely_above_0")),
                "used_for_verdict": bool(setting == "G" and cfg == best),
            }
        d_best = diffs.get(f"{best}_minus_A", {}).get(PRIMARY_METRIC, {})
        rules[f"B_beats_A_retrieval_{setting}"]["frozen_best_B"] = best
        rules[f"B_beats_A_retrieval_{setting}"]["frozen_best_MET"] = bool(d_best.get("ci_entirely_above_0"))

    # Tokens at r=0.8
    for setting in ("R", "G"):
        best = exp1["pooled_heldout"][setting]["frozen_best_B"]
        rules[f"B_saves_meaningful_tokens_{setting}"] = {}
        for cfg in ("B_leaf", "B_pillar"):
            p = exp2["settings"][setting]["paired_vs_A"].get(cfg, {}).get(str(PRIMARY_R), {})
            met = bool(p.get("rule_ge_15pct") and p.get("ci_excludes_0") and (p.get("relative_saving") or {}).get("mean", 0) > 0)
            # CI excluding 0 could be a significant *increase* in tokens; require positive saving.
            rel = p.get("relative_saving") or {}
            met = bool(
                p.get("n_paired", 0) > 0
                and p.get("rule_ge_15pct")
                and rel.get("ci95_low") is not None
                and rel["ci95_low"] > 0
            )
            rules[f"B_saves_meaningful_tokens_{setting}"][cfg] = {
                "n_paired": p.get("n_paired"),
                "relative_saving": rel,
                "rule_ge_15pct": p.get("rule_ge_15pct"),
                "ci_low_above_0": bool(rel.get("ci95_low") is not None and rel["ci95_low"] > 0),
                "MET": met,
                "used_for_verdict": bool(cfg == best),
            }
        rules[f"B_saves_meaningful_tokens_{setting}"]["frozen_best_B"] = best
        rules[f"B_saves_meaningful_tokens_{setting}"]["frozen_best_MET"] = rules[f"B_saves_meaningful_tokens_{setting}"][best]["MET"]

    for setting in ("R", "G"):
        b = exp3["settings"][setting]
        mw = b.get("mannwhitneyu") or {}
        p = mw.get("p_value")
        share = b.get("share_of_pruned_gold_in_A_top10")
        met = bool(p is not None and p < 0.05 and share is not None and share <= 0.10)
        rules[f"pruning_errors_fall_on_hard_evidence_{setting}"] = {
            "mannwhitneyu_p": p,
            "share_of_pruned_gold_in_A_top10": share,
            "n_pruned_gold_in_A_top10": b.get("n_pruned_gold_in_A_top10"),
            "p_lt_0_05": bool(p is not None and p < 0.05),
            "share_le_10pct": bool(share is not None and share <= 0.10),
            "MET": met,
            "frozen_best_B": b.get("frozen_best_B"),
        }

    beats_g = rules["B_beats_A_retrieval_G"]["frozen_best_MET"]
    tokens_any = rules["B_saves_meaningful_tokens_R"]["frozen_best_MET"] or rules["B_saves_meaningful_tokens_G"]["frozen_best_MET"]
    if beats_g:
        where = f"setting G retrieval (frozen best B={rules['B_beats_A_retrieval_G']['frozen_best_B']}, capped Recall@50)"
        verdict = f"POSITIVE: the hierarchy pays off in {where}"
    elif tokens_any:
        parts = []
        if rules["B_saves_meaningful_tokens_R"]["frozen_best_MET"]:
            parts.append("setting R tokens@r=0.8")
        if rules["B_saves_meaningful_tokens_G"]["frozen_best_MET"]:
            parts.append("setting G tokens@r=0.8")
        verdict = f"POSITIVE: the hierarchy pays off in {', '.join(parts)}"
    else:
        verdict = "NEGATIVE: no retrieval or token benefit at this scale"
    rules["verdict"] = {
        "text": verdict,
        "B_beats_A_in_G": beats_g,
        "B_saves_meaningful_tokens_in_either_setting": tokens_any,
        "pruning_hard_evidence_R": rules["pruning_errors_fall_on_hard_evidence_R"]["MET"],
        "pruning_hard_evidence_G": rules["pruning_errors_fall_on_hard_evidence_G"]["MET"],
    }
    return rules


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = _load_data()
    folds = core_folds(data["queries"], SEED)
    g_stats, g_per_report = compute_setting_g_question_stats(data["queries"], data["qrels"])
    enc, tok_name = load_tokenizer()
    costs = chunk_token_costs(data["chunks"]["text"].astype(str).tolist(), enc, TOKEN_OVERHEAD)
    np.save(PROCESSED / "chunk_token_costs.npy", costs)

    exp1 = experiment_1(data, folds)
    exp2 = experiment_2(data, exp1, costs, tok_name)
    exp3 = experiment_3(data, exp1)
    rules = _rule_eval(exp1, exp2, exp3)

    # Persist per-unit metrics without huge arrays
    slim_rows = []
    for setting in ("R", "G"):
        for r in exp1["heldout_rows"][setting]:
            slim_rows.append({k: v for k, v in r.items() if k not in {"ranked", "gold", "universe", "q_emb", "routed_node_indices"}})
    pd.DataFrame(slim_rows).to_csv(RESULTS / "followup_heldout_metrics.csv", index=False)

    env = collect_environment("BAAI/bge-small-en-v1.5")
    assign_stats = {}
    for setting in ("R", "G"):
        tau = exp1["fold1_frozen"][setting]["tau_pillar"]
        leaf_mem, _ = _mem_for_tau(data["leaf_sim"], data["tax"], tau)
        assign_stats[setting] = chunk_assignment_stats(leaf_mem, data["tax"])
        assign_stats[setting]["tau"] = tau

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT_FOLLOWUP,
        "gold_threshold": GOLD_THRESHOLD,
        "primary_metric": PRIMARY_METRIC,
        "primary_r": PRIMARY_R,
        "environment": env,
        "tokenizer": {"name": tok_name, "per_chunk_overhead": TOKEN_OVERHEAD},
        "folds": {k: v for k, v in folds.items() if k != "folds"} | {
            "fold_tune_test": [{"fold_id": f["fold_id"], "tune": f["tune"], "test": f["test"], "label": f["label"]} for f in folds["folds"]]
        },
        "setting_g_question_stats": g_stats.to_dict(orient="records"),
        "tau_selection": data["tau_dist"],
        "taxonomy": {"leaf_ids": data["tax"].leaf_ids, "pillar_ids": data["tax"].pillar_ids},
        "chunk_assignment": assign_stats,
        "experiment_1": {
            "pooled_heldout": exp1["pooled_heldout"],
            "original_core_test": exp1["original_core_test"],
            "extra": exp1["extra"],
            "fold_frozen": {
                str(fid): {s: {k: v for k, v in exp1["folds"][fid][s]["frozen"].items() if k != "sweep"} | {"sweep": exp1["folds"][fid][s]["frozen"]["sweep"]} for s in ("R", "G")}
                for fid in exp1["folds"]
            },
        },
        "experiment_2": {k: v for k, v in exp2.items() if k != "token_rows"},
        "experiment_3": {k: v for k, v in exp3.items() if k != "gold_rank_rows"},
        "rules": rules,
        "e_human": {
            "ran": False,
            "reason": "data/question_topics.csv was not present; run_e_human.py was not applied. Not used in rule evaluation.",
        },
        "notes": [
            "E_human skipped. Human labels were not invented.",
            "Setting G treats chunks in reports not annotated for a question as UNJUDGED. Recall is valid; precision-type numbers would be lower bounds and are not reported as precision.",
            "In setting G, tokens on unjudged chunks count as cost and are an upper bound.",
            "B_leaf and B_pillar are tuned separately on each fold. D_matched is matched to B_pillar candidate count.",
        ],
        "dataset_preprocessing": {
            "n_report_level_rows_raw": 47941,
            "n_empty_paragraph_rows_dropped": 15,
            "n_chunks": int(len(data["chunks"])),
            "n_reports": int(data["chunks"]["report"].nunique()),
            "exact_text_dedup_collapsed": 1,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
    }
    # strip numpy arrays from nested frozen sweeps — already JSON-safe
    path = RESULTS / "followup_results.json"
    path.write_text(json.dumps(json_safe(payload), indent=2))
    print(f"\nWrote {path}")

    write_figures(payload, exp1["heldout_rows"], exp3.get("gold_rank_rows") or [], exp2)
    write_report(path)
    print("Follow-up complete.")


if __name__ == "__main__":
    main()
