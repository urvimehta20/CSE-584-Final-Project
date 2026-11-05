"""Load ClimRetrieve, inspect source files, and write chunks/queries/qrels parquet."""

from __future__ import annotations

import json

import pandas as pd

from config import (
    BASE_PATH,
    CORE_PATH,
    GOLD_THRESHOLD,
    PROCESSED,
    REPORT_LEVEL_PATH,
    RESULTS,
)
from inspect_data import inspect_one


def _series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    if s.empty:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": int(s.min()),
        "median": float(s.median()),
        "max": int(s.max()),
        "mean": float(s.mean()),
    }


def load_sources(inspect: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if inspect:
        print("\n========== SOURCE FILE INSPECTION ==========")
        inspect_one(REPORT_LEVEL_PATH)
        inspect_one(BASE_PATH)
        inspect_one(CORE_PATH)
    else:
        print("Skipping source-file reprint (tables already built).")
    return pd.read_csv(REPORT_LEVEL_PATH), pd.read_excel(BASE_PATH), pd.read_excel(CORE_PATH)


def _core_flag_by_question(base: pd.DataFrame) -> dict[str, int]:
    tmp = base.copy()
    tmp["Question"] = tmp["Question"].astype(str)
    return tmp.groupby("Question")["Core 16 Question"].max().astype(int).to_dict()


def extra_coverage_table(queries: pd.DataFrame) -> pd.DataFrame:
    extra = queries.loc[queries["set_type"] == "extra"]
    rows = []
    for question, grp in extra.groupby("question", sort=True):
        n_gold = int(grp["n_gold_ge2"].sum())
        rows.append(
            {
                "question": question,
                "n_reports": int(grp["report"].nunique()),
                "n_queries": int(len(grp)),
                "n_gold_chunks_ge2": n_gold,
                "n_queries_zero_gold": int((grp["n_gold_ge2"] == 0).sum()),
                "dropped": bool(n_gold == 0),
            }
        )
    return pd.DataFrame(rows)


def print_extra_coverage(coverage: pd.DataFrame) -> None:
    print("\n========== EXTRA-QUESTION COVERAGE (before extra evaluation) ==========")
    if coverage.empty:
        print("No extra questions found.")
        return
    print(coverage.to_string(index=False))
    dropped = coverage.loc[coverage["dropped"], "question"].tolist()
    print(f"\nDropped extra questions with zero gold chunks across all reports: {dropped}")


def build_tables(
    report_level: pd.DataFrame, base: pd.DataFrame, core: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    notes: list[str] = []
    n_rows_raw = int(len(report_level))
    n_rel_na = int(report_level["relevance"].isna().sum())

    df = report_level.copy()
    df["paragraph"] = df["paragraph"].astype("string")
    empty_mask = df["paragraph"].isna() | df["paragraph"].str.strip().eq("")
    n_empty = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()
    notes.append(
        f"Dropped {n_empty} report-level rows with null/empty paragraph (raw rows={n_rows_raw})."
    )
    if n_rel_na:
        notes.append(
            f"Report-level relevance has {n_rel_na} NaN values. "
            "After max-aggregating per (report, question, paragraph), "
            "pairs whose only values are NaN are omitted from qrels."
        )

    flag_by_q = _core_flag_by_question(base)
    expl_core = set(core["question"].dropna().astype(str))
    flag_core = {q for q, flag in flag_by_q.items() if flag == 1}
    if expl_core != flag_core:
        notes.append(
            "Core_Questions_with_Explanations.xlsx and the base 'Core 16 Question' flag "
            f"disagree. Tagging uses the base flag. only_expl={sorted(expl_core-flag_core)}; "
            f"only_flag={sorted(flag_core-expl_core)}"
        )
    else:
        notes.append(
            "Query set_type is tagged from ClimRetrieve_base 'Core 16 Question' "
            "(max flag per Question text). It matches Core_Questions_with_Explanations.xlsx."
        )

    chunks = (
        df.groupby(["report", "paragraph"], as_index=False)
        .size()
        .drop(columns="size")
        .rename(columns={"paragraph": "text"})
        .sort_values(["report", "text"], kind="mergesort")
        .reset_index(drop=True)
    )
    chunks.insert(0, "chunk_id", [f"c_{i:05d}" for i in range(len(chunks))])

    n_same_text_multi_idx = int(
        (df.groupby(["report", "paragraph"])["Unnamed: 0"].nunique() > 1).sum()
    )
    if n_same_text_multi_idx:
        notes.append(
            f"Exact-text dedup within a report collapsed {n_same_text_multi_idx} "
            "paragraph text(s) that had more than one original row index."
        )

    queries = (
        df[["report", "question"]]
        .drop_duplicates()
        .sort_values(["report", "question"], kind="mergesort")
        .reset_index(drop=True)
    )
    queries.insert(0, "query_id", [f"q_{i:04d}" for i in range(len(queries))])
    missing_flag = queries["question"].map(lambda q: q not in flag_by_q)
    if missing_flag.any():
        missing_qs = sorted(queries.loc[missing_flag, "question"].unique())
        notes.append(
            "Questions present in report-level but absent from ClimRetrieve_base; "
            f"tagged extra: {missing_qs}"
        )
    queries["set_type"] = queries["question"].map(
        lambda q: "core" if flag_by_q.get(q, 0) == 1 else "extra"
    )
    queries["is_core16"] = queries["set_type"] == "core"

    labeled = df[["report", "question", "paragraph", "relevance"]].copy()
    labeled = labeled.merge(queries[["query_id", "report", "question", "set_type"]], on=["report", "question"], how="left")
    labeled = labeled.merge(
        chunks.rename(columns={"text": "paragraph"}),
        on=["report", "paragraph"],
        how="left",
    )
    qrels = (
        labeled.groupby(["query_id", "chunk_id", "set_type"], as_index=False)["relevance"]
        .max()
        .dropna(subset=["relevance"])
        .sort_values(["query_id", "chunk_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    report_pairs = set(zip(queries["report"], queries["question"]))
    base_pairs = set(zip(base["Document"].astype(str), base["Question"].astype(str)))
    missing_from_report = sorted(base_pairs - report_pairs)
    extra_in_report = sorted(report_pairs - base_pairs)
    if missing_from_report:
        notes.append(
            "Expert-base (Document, Question) pairs absent from report-level "
            f"({len(missing_from_report)}): {missing_from_report}."
        )
    if extra_in_report:
        notes.append(
            "Report-level (report, question) pairs absent from expert base "
            f"({len(extra_in_report)}): {extra_in_report}."
        )

    gold_by_query = _gold_counts(qrels)
    queries = queries.merge(gold_by_query, on="query_id", how="left")
    for col in ["n_gold_ge1", "n_gold_ge2", "n_gold_eq3"]:
        queries[col] = queries[col].fillna(0).astype(int)
    queries["eval_eligible"] = queries["n_gold_ge2"] > 0

    extra_cov = extra_coverage_table(queries)
    print_extra_coverage(extra_cov)
    dropped_extra = extra_cov.loc[extra_cov["dropped"], "question"].tolist()
    queries["extra_question_dropped"] = queries["question"].isin(dropped_extra)

    stats = compute_stats(
        chunks=chunks,
        queries=queries,
        qrels=qrels,
        extra_cov=extra_cov,
        notes=notes,
        n_rows_raw=n_rows_raw,
        n_empty_paragraph_rows=n_empty,
        n_relevance_nan_rows=n_rel_na,
        missing_from_report=missing_from_report,
        flag_core=sorted(flag_core),
        extra_questions=sorted(queries.loc[queries["set_type"] == "extra", "question"].unique()),
    )
    return chunks, queries, qrels, stats


def _gold_counts(qrels: pd.DataFrame) -> pd.DataFrame:
    if qrels.empty:
        return pd.DataFrame(columns=["query_id", "n_gold_ge1", "n_gold_ge2", "n_gold_eq3"])
    return qrels.groupby("query_id", as_index=False).agg(
        n_gold_ge1=("relevance", lambda s: int((s >= 1).sum())),
        n_gold_ge2=("relevance", lambda s: int((s >= GOLD_THRESHOLD).sum())),
        n_gold_eq3=("relevance", lambda s: int((s == 3).sum())),
    )


def compute_stats(
    chunks: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    extra_cov: pd.DataFrame,
    notes: list[str],
    n_rows_raw: int,
    n_empty_paragraph_rows: int,
    n_relevance_nan_rows: int,
    missing_from_report: list[tuple[str, str]],
    flag_core: list[str],
    extra_questions: list[str],
) -> dict:
    chunks_per_report = chunks.groupby("report").size()

    def _subset_stats(qsub: pd.DataFrame, label: str) -> dict:
        eval_q = qsub.loc[qsub["eval_eligible"]]
        return {
            "label": label,
            "n_unique_questions": int(qsub["question"].nunique()),
            "n_queries": int(len(qsub)),
            "n_queries_zero_gold_ge2": int((~qsub["eval_eligible"]).sum()),
            "n_queries_excluded_from_eval": int((~qsub["eval_eligible"]).sum()),
            "n_eval_queries_after_exclusion": int(len(eval_q)),
            "gold_chunks_per_query_ge2_all_queries": _series_stats(qsub["n_gold_ge2"]),
            "gold_chunks_per_query_ge2_eval_only": _series_stats(eval_q["n_gold_ge2"]),
            "gold_chunks_per_query_ge1_all_queries": _series_stats(qsub["n_gold_ge1"]),
            "gold_chunks_per_query_eq3_all_queries": _series_stats(qsub["n_gold_eq3"]),
            "n_queries_zero_gold_ge1": int((qsub["n_gold_ge1"] == 0).sum()),
            "n_queries_zero_gold_eq3": int((qsub["n_gold_eq3"] == 0).sum()),
        }

    qrels_pos = qrels.loc[qrels["relevance"] > 0]
    return {
        "source_files": {
            "report_level": str(REPORT_LEVEL_PATH),
            "expert_base": str(BASE_PATH),
            "core_questions": str(CORE_PATH),
        },
        "n_report_level_rows_raw": n_rows_raw,
        "n_empty_paragraph_rows_dropped": n_empty_paragraph_rows,
        "n_relevance_nan_rows_in_report_level": n_relevance_nan_rows,
        "n_reports": int(chunks["report"].nunique()),
        "n_chunks": int(len(chunks)),
        "chunks_per_report": _series_stats(chunks_per_report),
        "n_unique_questions_all": int(queries["question"].nunique()),
        "n_unique_core_questions": int((queries["set_type"] == "core").sum() and queries.loc[queries["set_type"] == "core", "question"].nunique()),
        "n_unique_extra_questions": int(queries.loc[queries["set_type"] == "extra", "question"].nunique()),
        "n_queries_all": int(len(queries)),
        "n_queries_core": int((queries["set_type"] == "core").sum()),
        "n_queries_extra": int((queries["set_type"] == "extra").sum()),
        "n_qrels_rows": int(len(qrels)),
        "n_qrels_positive_relevance": int(len(qrels_pos)),
        "relevance_value_counts": {
            str(k): int(v)
            for k, v in qrels["relevance"].value_counts(dropna=False).sort_index().items()
        },
        "all_questions": _subset_stats(queries, "all_report_level_questions"),
        "core_questions": _subset_stats(queries.loc[queries["set_type"] == "core"].copy(), "core_questions"),
        "extra_questions": _subset_stats(queries.loc[queries["set_type"] == "extra"].copy(), "extra_questions"),
        "extra_coverage": extra_cov.to_dict(orient="records"),
        "dropped_extra_questions": extra_cov.loc[extra_cov["dropped"], "question"].tolist(),
        "expert_base_pairs_missing_from_report_level": [
            {"report": r, "question": q} for r, q in missing_from_report
        ],
        "core_question_list": flag_core,
        "extra_question_list": extra_questions,
        "notes": notes,
        "gold_threshold_main": GOLD_THRESHOLD,
        "column_mapping": {
            "chunks.text": "report-level.paragraph (exact-deduped within report)",
            "chunks.report": "report-level.report",
            "queries.question": "report-level.question",
            "queries.report": "report-level.report",
            "queries.set_type": "core if ClimRetrieve_base['Core 16 Question']==1 else extra",
            "qrels.relevance": "report-level.relevance (max over duplicate rows)",
            "qrels.set_type": "copied from queries.set_type",
        },
    }


def save_outputs(
    chunks: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame, stats: dict
) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(PROCESSED / "chunks.parquet", index=False)
    queries.to_parquet(PROCESSED / "queries.parquet", index=False)
    qrels.to_parquet(PROCESSED / "qrels.parquet", index=False)
    extra_cov = pd.DataFrame(stats["extra_coverage"])
    extra_cov.to_csv(RESULTS / "extra_coverage.csv", index=False)
    stats_path = RESULTS / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {PROCESSED / 'chunks.parquet'} rows={len(chunks)}")
    print(f"Wrote {PROCESSED / 'queries.parquet'} rows={len(queries)}")
    print(f"Wrote {PROCESSED / 'qrels.parquet'} rows={len(qrels)}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {RESULTS / 'extra_coverage.csv'}")


def main(inspect: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    report_level, base, core = load_sources(inspect=inspect)
    chunks, queries, qrels, stats = build_tables(report_level, base, core)
    save_outputs(chunks, queries, qrels, stats)
    print("\n========== DATASET STATS ==========")
    print(json.dumps(stats, indent=2))
    return chunks, queries, qrels, stats


if __name__ == "__main__":
    main()
