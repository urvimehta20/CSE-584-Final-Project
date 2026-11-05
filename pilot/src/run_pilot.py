"""Orchestrate the ClimRetrieve hierarchy pilot and write results JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import EMB_DIR, PROCESSED, RESULTS, SEED, SRC
from embed import embed_corpus
from evaluate import assign_question_split, run_evaluation
from load import main as load_main
from report import export_label_sample, routing_diagnostic, write_figures, write_summary_from_json
from retrieve import build_index_maps
from taxonomy import (
    build_taxonomy,
    chunk_assignment_stats,
    embed_nodes,
    node_matrices,
    choose_taus,
)
from util import collect_environment, json_safe


def _run_timing_subprocess() -> dict:
    cmd = [sys.executable, str(SRC / "timing.py")]
    print("Launching", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(SRC.parent), env={**os.environ, "PYTHONPATH": str(SRC)})
    timing_path = RESULTS / "timing_results.json"
    partial_path = RESULTS / "timing_partial.json"
    if proc.returncode == 0 and timing_path.exists():
        return json.loads(timing_path.read_text())
    note = (
        f"Timing subprocess exited with code {proc.returncode}. "
        "A native crash here is usually FAISS/HNSW or the 1e6 synthetic corpus."
    )
    print(note)
    if partial_path.exists():
        data = json.loads(partial_path.read_text())
        data.setdefault("failures", []).append(note)
        data["where_saving_exceeds_1ms"] = _savings_threshold(data.get("savings_filtered_vs_unfiltered_exact") or [])
        data["incomplete"] = True
        return data
    return {
        "n_queries_timed": None,
        "warmup": None,
        "real_corpus_size": None,
        "settings": [],
        "savings_filtered_vs_unfiltered_exact": [],
        "where_saving_exceeds_1ms": {
            "min_N": None,
            "examples": [],
            "note": note,
        },
        "failures": [note],
        "incomplete": True,
    }


def _savings_threshold(savings: list[dict]) -> dict:
    over = [s for s in savings if s.get("exceeds_1ms")]
    if over:
        return {
            "min_N": min(s["N"] for s in over),
            "examples": over,
            "note": "N values where median filtered exact search was >1 ms faster than unfiltered exact.",
        }
    return {
        "min_N": None,
        "examples": [],
        "note": "In the completed timing settings, filtered exact search was never more than 1 ms faster than unfiltered exact.",
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    inspect = os.environ.get("PILOT_SKIP_INSPECT") != "1"
    chunks, queries, qrels, dataset_stats = load_main(inspect=inspect)
    queries, split_info = assign_question_split(queries, SEED)
    queries.to_parquet(PROCESSED / "queries.parquet", index=False)
    print(
        "Question split (by question, seed="
        f"{SEED}): core_dev={split_info['core_dev']} core_test={split_info['core_test']}"
    )
    print(
        "Eligible queries by split:",
        queries.loc[queries["eval_eligible"]]
        .groupby("question_split")
        .size()
        .to_dict(),
    )

    emb = embed_corpus(chunks, queries)
    tax = build_taxonomy()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(emb["model_name"])
    node_emb = embed_nodes(model, emb["query_prefix"], tax)
    mats = node_matrices(tax, node_emb)
    np.save(EMB_DIR / "node_embeddings.npy", node_emb)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        leaf_sim = emb["chunk_embeddings"] @ mats["leaf"].T
    taus, tau_dist = choose_taus(leaf_sim)
    print(f"τ values from second-highest leaf cosine: {taus}")

    pack = build_index_maps(chunks, queries, qrels, emb["query_embeddings"])
    extra_dropped = list(dataset_stats.get("dropped_extra_questions") or [])
    ev = run_evaluation(
        pack=pack,
        chunks=chunks,
        chunk_emb=emb["chunk_embeddings"],
        tax=tax,
        node_mats=mats,
        taus=taus,
        extra_dropped=extra_dropped,
    )

    qtype = {
        str(q): str(t)
        for q, t in queries.drop_duplicates("question")[["question", "set_type"]].itertuples(
            index=False, name=None
        )
    }
    routing = routing_diagnostic(
        emb["question_texts"], emb["question_embeddings"], qtype, tax, mats
    )
    print("\n========== ROUTING (all questions) ==========")
    for row in routing["per_question"]:
        print(
            f"  [{row['set_type']}] {row['question'][:70]} -> "
            f"{row['top_pillar']}/{row['top_leaf_name']}"
        )
    print(
        f"Core distinct pillars: {routing['core']['n_distinct_pillars']} {routing['core']['distinct_pillars']}"
    )
    print(
        f"Extra distinct pillars: {routing['extra']['n_distinct_pillars']} {routing['extra']['distinct_pillars']}"
    )

    leaf_mem = ev["leaf_membership_chosen"]
    assign_stats = chunk_assignment_stats(leaf_mem, tax)
    export_label_sample(chunks, leaf_mem, tax)

    env = collect_environment(emb["model_name"])
    print("\n========== TIMING PILOT (subprocess) ==========")
    timing = _run_timing_subprocess()

    caveats = [
        "Synthetic scale-up embeddings are real vectors plus Gaussian noise, then re-normalized; those N values are labeled synthetic in the JSON.",
        "Primary test set is 8 core questions only (queries are those questions asked on reports with gold at relevance ≥ 2).",
        "All ClimRetrieve questions are climate-related, so topic routing may collapse onto a small number of pillars/leaves.",
        "Chunk topic labels are automatic cosine assignments and have not been human-verified (config C is out of scope).",
        "λ and τ were chosen on core-dev; core-dev tables are tuning data, not a held-out estimate.",
        "Core-test and extra numbers are never pooled.",
    ]
    notes.extend(dataset_stats.get("notes") or [])

    per_query_df = pd.DataFrame(ev["per_query"])
    per_query_df.to_csv(RESULTS / "per_query_metrics.csv", index=False)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "environment": env,
        "dataset": dataset_stats,
        "splits": {
            "core_dev": split_info["core_dev"],
            "core_test": split_info["core_test"],
            "extra_retained": [
                q
                for q in dataset_stats.get("extra_question_list", [])
                if q not in extra_dropped
            ],
            "extra_dropped": extra_dropped,
            "rule": "Split the 16 core questions 8/8 by question string, not by query. Extra questions are a secondary held-out set.",
        },
        "embeddings": {
            "model_name": emb["model_name"],
            "query_prefix": emb["query_prefix"],
            "dim": emb["dim"],
            "normalized": True,
            "n_chunks": int(emb["chunk_embeddings"].shape[0]),
            "n_unique_questions": int(emb["question_embeddings"].shape[0]),
        },
        "taxonomy": {
            "n_leaves": len(tax.leaf_ids),
            "leaf_ids": tax.leaf_ids,
            "pillar_ids": tax.pillar_ids,
            "node_ids": tax.node_ids,
        },
        "tau_selection": tau_dist,
        "retrieval": {
            "chosen_setting": ev["chosen_setting"],
            "sweep_points": ev["sweep_points"],
            "frozen_metrics": ev["frozen_metrics"],
            "miss_attribution": ev["miss_attribution"],
            "split_counts_eligible_queries": ev["split_counts"],
        },
        "timing": timing,
        "diagnostics": {
            "routing": routing,
            "chunk_assignment": assign_stats,
        },
        "notes": notes,
        "caveats": caveats,
    }
    # Drop bulky arrays before JSON.
    json_path = RESULTS / "pilot_results.json"
    json_path.write_text(json.dumps(json_safe(results), indent=2))
    print(f"\nWrote {json_path}")

    figs = write_figures(results)
    print(f"Wrote figures: {figs}")
    write_summary_from_json(json_path)
    print("Pilot complete.")


if __name__ == "__main__":
    main()
