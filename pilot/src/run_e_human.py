"""Human-labeled oracle routing. Runs only if data/question_topics.csv exists.

Writes results/e_human_results.json. Does not update REPORT.md or rule evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, EMB_DIR, GOLD_THRESHOLD, PROCESSED, RESULTS, SEED
from followup_common import (
    N_BOOT_FOLLOWUP,
    cluster_bootstrap_mean,
    core_folds,
    gold_chunk_indices,
)
from followup_common import capped_recall, miss_rate, raw_recall
from followup_retrieve import candidates_from_nodes, rank_by_cosine
from taxonomy import assign_leaves, build_taxonomy, choose_taus, leaf_to_pillar_membership, node_matrices
from util import json_safe


TOPICS_PATH = DATA / "question_topics.csv"


def _parse_topics(path: Path, tax) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"question", "set", "human_pillar", "human_leaves"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["human_pillar"] = df["human_pillar"].fillna("").astype(str).str.strip()
    df["human_leaves"] = df["human_leaves"].fillna("").astype(str).str.strip()
    filled = df.loc[(df["human_pillar"] != "") | (df["human_leaves"] != "")].copy()
    if filled.empty:
        raise ValueError(f"{path} exists but human_pillar and human_leaves are empty on every row")
    bad_p = sorted({p for p in filled["human_pillar"] if p and p not in tax.pillar_ids})
    leaves_ok = set(tax.leaf_ids)
    bad_l = []
    for s in filled["human_leaves"]:
        for leaf in [x.strip() for x in s.split(";") if x.strip()]:
            if leaf not in leaves_ok:
                bad_l.append(leaf)
    if bad_p or bad_l:
        raise ValueError(f"Unknown human labels: pillars={bad_p} leaves={sorted(set(bad_l))}")
    return filled


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not TOPICS_PATH.exists():
        print(f"SKIP E_human: {TOPICS_PATH} does not exist. Fill data/question_topics_TEMPLATE.csv and save as question_topics.csv.")
        return
    tax = build_taxonomy()
    try:
        topics = _parse_topics(TOPICS_PATH, tax)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ran": False,
            "error": f"{type(exc).__name__}: {exc}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (RESULTS / "e_human_results.json").write_text(json.dumps(payload, indent=2))
        print(f"E_human failed: {payload['error']}")
        raise

    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    qrels = pd.read_parquet(PROCESSED / "qrels.parquet")
    chunk_emb = np.load(EMB_DIR / "chunk_embeddings.npy")
    question_emb = np.load(EMB_DIR / "question_embeddings.npy")
    question_texts = json.loads((EMB_DIR / "question_texts.json").read_text())
    qmap = {q: i for i, q in enumerate(question_texts)}
    node_emb = np.load(EMB_DIR / "node_embeddings.npy")
    mats = node_matrices(tax, node_emb)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        leaf_sim = chunk_emb @ mats["leaf"].T
    taus, _ = choose_taus(leaf_sim)
    # Use the middle τ (same distribution rule as the pilot / follow-up).
    tau = float(taus[1])
    leaf_mem = assign_leaves(leaf_sim, tau)
    pillar_mem = leaf_to_pillar_membership(leaf_mem, tax)
    chunk_pos = {cid: i for i, cid in enumerate(chunks["chunk_id"])}
    gold = gold_chunk_indices(queries, qrels, chunk_pos)
    universe_all = np.arange(len(chunks), dtype=int)

    pillar_index = {p: i for i, p in enumerate(tax.pillar_ids)}
    leaf_index = {leaf: i for i, leaf in enumerate(tax.leaf_ids)}

    rows = []
    for row in topics.itertuples(index=False):
        q = row.question
        if q not in gold["by_question"] or gold["by_question"][q].size == 0:
            continue
        if q not in qmap:
            continue
        q_emb = question_emb[qmap[q]]
        gidx = gold["by_question"][q]
        if row.human_leaves:
            nodes = np.array(
                [leaf_index[x.strip()] for x in row.human_leaves.split(";") if x.strip()],
                dtype=int,
            )
            cand = candidates_from_nodes(universe_all, leaf_mem, nodes)
            kind = "human_leaves"
        else:
            nodes = np.array([pillar_index[row.human_pillar]], dtype=int)
            cand = candidates_from_nodes(universe_all, pillar_mem, nodes)
            kind = "human_pillar"
        ranked = rank_by_cosine(q_emb, chunk_emb, cand)
        rows.append(
            {
                "question": q,
                "set": row.set,
                "kind": kind,
                "human_pillar": row.human_pillar,
                "human_leaves": row.human_leaves,
                "n_gold": int(gidx.size),
                "n_candidates": int(ranked.size),
                "selectivity": float(ranked.size / universe_all.size),
                "miss_rate": miss_rate(gidx, ranked),
                "capped_recall@50": capped_recall(gidx, ranked, 50),
                "recall@50": raw_recall(gidx, ranked, 50),
                "recall@20": raw_recall(gidx, ranked, 20),
                "recall@100": raw_recall(gidx, ranked, 100),
            }
        )

    core_rows = [r for r in rows if r["set"] == "core"]
    extra_rows = [r for r in rows if r["set"] == "extra"]

    def _agg(sub: list[dict]) -> dict:
        if not sub:
            return {"n_questions": 0, "note": "no labeled questions with gold"}
        qs = [r["question"] for r in sub]
        out = {"n_questions": len(sub)}
        for key in ("capped_recall@50", "recall@20", "recall@50", "recall@100", "miss_rate", "selectivity"):
            out[key] = cluster_bootstrap_mean(
                qs, np.array([r[key] for r in sub], dtype=float), N_BOOT_FOLLOWUP, SEED, "question"
            )
        return out

    payload = {
        "ran": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(TOPICS_PATH),
        "tau_used_for_chunk_assignment": tau,
        "gold_threshold": GOLD_THRESHOLD,
        "n_labeled_rows_in_csv": int(len(topics)),
        "n_evaluated_core": len(core_rows),
        "n_evaluated_extra": len(extra_rows),
        "core_macro": _agg(core_rows),
        "extra_macro": _agg(extra_rows),
        "per_question": rows,
        "note": (
            "Standalone human-oracle run. Not merged into followup_results.json rule evaluation "
            "and not cited by REPORT.md decision rules."
        ),
    }
    out = RESULTS / "e_human_results.json"
    out.write_text(json.dumps(json_safe(payload), indent=2))
    print(f"Wrote {out} core={len(core_rows)} extra={len(extra_rows)}")


if __name__ == "__main__":
    main()
