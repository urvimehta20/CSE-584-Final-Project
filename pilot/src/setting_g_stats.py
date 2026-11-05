"""Setting G (corpus-wide) per-question gold coverage. Checkpoint before Experiment 1."""

from __future__ import annotations

import json

import pandas as pd

from config import GOLD_THRESHOLD, PROCESSED, RESULTS


def compute_setting_g_question_stats(
    queries: pd.DataFrame, qrels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = qrels.loc[qrels["relevance"] >= GOLD_THRESHOLD].copy()
    gold = gold.merge(
        queries[["query_id", "question", "report", "set_type"]],
        on="query_id",
        how="left",
    )

    per_report_rows = []
    summary_rows = []
    questions = (
        queries[["question", "set_type"]]
        .drop_duplicates()
        .sort_values(["set_type", "question"], kind="mergesort")
    )
    for question, set_type in questions.itertuples(index=False, name=None):
        q_queries = queries.loc[queries["question"] == question]
        annotated_reports = sorted(q_queries["report"].unique())
        q_gold = gold.loc[gold["question"] == question]
        n_gold = int(q_gold["chunk_id"].nunique())
        gold_by_report = (
            q_gold.groupby("report")["chunk_id"].nunique() if len(q_gold) else pd.Series(dtype=int)
        )
        per_report_counts = []
        for report in annotated_reports:
            n = int(gold_by_report.get(report, 0))
            per_report_counts.append(n)
            per_report_rows.append(
                {
                    "question": question,
                    "set": set_type,
                    "report": report,
                    "n_gold_chunks": n,
                }
            )
        s = pd.Series(per_report_counts, dtype=float)
        summary_rows.append(
            {
                "question": question,
                "set": set_type,
                "n_annotated_reports": int(len(annotated_reports)),
                "n_gold_chunks": n_gold,
                "gold_chunks_per_report_min": int(s.min()) if len(s) else 0,
                "gold_chunks_per_report_median": float(s.median()) if len(s) else 0.0,
                "gold_chunks_per_report_max": int(s.max()) if len(s) else 0,
                "n_annotated_reports_with_zero_gold": int((s == 0).sum()) if len(s) else 0,
                "gold_chunks_per_report": per_report_counts,
                "annotated_reports": annotated_reports,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_report_rows)


def main() -> dict:
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    qrels = pd.read_parquet(PROCESSED / "qrels.parquet")
    summary, per_report = compute_setting_g_question_stats(queries, qrels)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "setting_g_question_stats.csv"
    per_report_path = RESULTS / "setting_g_gold_per_report.csv"
    json_path = RESULTS / "setting_g_question_stats.json"
    summary.drop(columns=["gold_chunks_per_report", "annotated_reports"]).to_csv(
        summary_path, index=False
    )
    per_report.to_csv(per_report_path, index=False)
    payload = {
        "gold_threshold": GOLD_THRESHOLD,
        "n_chunks_in_corpus": int(pd.read_parquet(PROCESSED / "chunks.parquet").shape[0]),
        "definition": (
            "Setting G evaluates a QUESTION, not a (question, report) pair. "
            "Annotated reports = reports where that question appears in the report-level file. "
            "Gold = unique chunk_ids with relevance >= 2 across those reports. "
            "Chunks in reports not annotated for the question are UNJUDGED."
        ),
        "questions": summary.to_dict(orient="records"),
        "n_core_questions": int((summary["set"] == "core").sum()),
        "n_extra_questions": int((summary["set"] == "extra").sum()),
        "n_core_questions_with_zero_gold": int(
            ((summary["set"] == "core") & (summary["n_gold_chunks"] == 0)).sum()
        ),
        "n_extra_questions_with_zero_gold": int(
            ((summary["set"] == "extra") & (summary["n_gold_chunks"] == 0)).sum()
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print("\n========== SETTING G PER-QUESTION STATS (checkpoint) ==========")
    print(payload["definition"])
    print(
        f"\nCorpus chunks: {payload['n_chunks_in_corpus']}. "
        f"Core questions: {payload['n_core_questions']} "
        f"({payload['n_core_questions_with_zero_gold']} with zero gold). "
        f"Extra questions: {payload['n_extra_questions']} "
        f"({payload['n_extra_questions_with_zero_gold']} with zero gold)."
    )
    show = summary[
        [
            "question",
            "set",
            "n_annotated_reports",
            "n_gold_chunks",
            "gold_chunks_per_report_min",
            "gold_chunks_per_report_median",
            "gold_chunks_per_report_max",
            "n_annotated_reports_with_zero_gold",
        ]
    ]
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.width", 220)
    print("\n" + show.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {per_report_path}")
    print(f"Wrote {json_path}")
    return payload


if __name__ == "__main__":
    main()
