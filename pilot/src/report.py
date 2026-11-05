"""Write figures and generate summary.md strictly from pilot_results.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import FIGURES, LABEL_SAMPLE_N, RESULTS, SEED
from taxonomy import Taxonomy


def routing_diagnostic(
    question_texts: list[str],
    question_emb: np.ndarray,
    question_set_type: dict[str, str],
    tax: Taxonomy,
    node_mats: dict,
) -> dict[str, Any]:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        leaf_sim = question_emb @ node_mats["leaf"].T
        pillar_sim = question_emb @ node_mats["pillar"].T
    rows = []
    for i, q in enumerate(question_texts):
        li = int(np.argmax(leaf_sim[i]))
        pi = int(np.argmax(pillar_sim[i]))
        rows.append(
            {
                "question": q,
                "set_type": question_set_type.get(q, "unknown"),
                "top_leaf": tax.leaf_ids[li],
                "top_leaf_name": tax.node_names[tax.leaf_ids[li]],
                "top_leaf_cosine": float(leaf_sim[i, li]),
                "top_pillar": tax.pillar_ids[pi],
                "top_pillar_name": tax.node_names[tax.pillar_ids[pi]],
                "top_pillar_cosine": float(pillar_sim[i, pi]),
            }
        )
    df = pd.DataFrame(rows)

    def _group(name: str) -> dict[str, Any]:
        sub = df.loc[df["set_type"] == name]
        pillars = sub["top_pillar"].tolist()
        return {
            "n_questions": int(len(sub)),
            "distinct_pillars": sorted(set(pillars)),
            "n_distinct_pillars": int(sub["top_pillar"].nunique()) if len(sub) else 0,
            "pillar_counts": {k: int(v) for k, v in sub["top_pillar"].value_counts().to_dict().items()},
            "n_questions_sharing_the_modal_pillar": (
                int(sub["top_pillar"].value_counts().iloc[0]) if len(sub) else 0
            ),
            "modal_pillar": (
                str(sub["top_pillar"].value_counts().index[0]) if len(sub) else None
            ),
        }

    return {
        "per_question": rows,
        "core": _group("core"),
        "extra": _group("extra"),
        "all": _group("core") if False else {
            "n_questions": int(len(df)),
            "n_distinct_pillars": int(df["top_pillar"].nunique()) if len(df) else 0,
            "pillar_counts": {k: int(v) for k, v in df["top_pillar"].value_counts().to_dict().items()},
        },
    }


def export_label_sample(
    chunks: pd.DataFrame,
    leaf_mem: np.ndarray,
    tax: Taxonomy,
    n: int = LABEL_SAMPLE_N,
    seed: int = SEED,
) -> pd.DataFrame:
    assigned = []
    for i in range(len(chunks)):
        leaves = [tax.leaf_ids[j] for j in np.flatnonzero(leaf_mem[i])]
        assigned.append("|".join(leaves))
    tmp = chunks.copy()
    tmp["auto_leaves"] = assigned
    tmp["primary_leaf"] = [tax.leaf_ids[int(np.argmax(leaf_mem[i]))] if leaf_mem[i].any() else "" for i in range(len(tmp))]
    rng = np.random.default_rng(seed)
    parts = []
    leaves = sorted(tmp["primary_leaf"].unique())
    if not leaves:
        sample = tmp.head(0)
    else:
        base = n // len(leaves)
        rem = n % len(leaves)
        for k, leaf in enumerate(leaves):
            take = base + (1 if k < rem else 0)
            grp = tmp.loc[tmp["primary_leaf"] == leaf]
            if len(grp) <= take:
                parts.append(grp)
            else:
                idx = rng.choice(grp.index.to_numpy(), size=take, replace=False)
                parts.append(grp.loc[idx])
        sample = pd.concat(parts, axis=0)
        if len(sample) < n:
            leftover = tmp.drop(index=sample.index, errors="ignore")
            need = n - len(sample)
            if len(leftover) and need > 0:
                idx = rng.choice(
                    leftover.index.to_numpy(),
                    size=min(need, len(leftover)),
                    replace=False,
                )
                sample = pd.concat([sample, leftover.loc[idx]], axis=0)
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=seed)
    sample = sample.sort_values(["primary_leaf", "chunk_id"]).reset_index(drop=True)
    out = sample[["chunk_id", "report", "text", "auto_leaves"]].copy()
    out["annotator_1"] = ""
    out["annotator_2"] = ""
    out_path = RESULTS / "label_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)}")
    return out


def write_figures(results: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # (1) Recall@10 vs selectivity, one point per sweep setting
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"A": "o", "B": "s", "D": "D", "E": "^"}
    colors = {"A": "#4C72B0", "B": "#DD8452", "D": "#55A868", "E": "#C44E52"}
    plotted_a = False
    for cfg in ("A", "B", "D", "E"):
        xs, ys, labels = [], [], []
        seen = set()
        for p in results["retrieval"]["sweep_points"]:
            m = p["metrics"][cfg]
            rec = (m.get("recall@10") or {}).get("mean")
            sel = (m.get("selectivity") or {}).get("mean")
            if rec is None or sel is None:
                continue
            if cfg == "A":
                if plotted_a:
                    continue
                plotted_a = True
                xs.append(sel)
                ys.append(rec)
                labels.append("A (no topic filter)")
                break
            key = (round(sel, 6), round(rec, 6), p["level"], p["lambda"], p["tau"])
            if key in seen:
                continue
            seen.add(key)
            xs.append(sel)
            ys.append(rec)
            labels.append(f"{cfg} {p['level']} λ={p['lambda']} τ={p['tau']:.2f}")
        if not xs:
            continue
        ax.scatter(xs, ys, marker=markers[cfg], c=colors[cfg], label=cfg, s=45, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=6, alpha=0.85)
    ax.set_xlabel("Selectivity (mean |candidates| / |report chunks|)")
    ax.set_ylabel("Recall@10 (mean, core-dev)")
    ax.set_title("Recall@10 vs selectivity on core-dev sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p1 = FIGURES / "recall_vs_selectivity.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    written.append(str(p1))

    # (2) latency vs N log-log, unfiltered vs filtered
    fig, ax = plt.subplots(figsize=(8, 6))
    timing = results["timing"]["settings"]
    series: dict[str, list[tuple[float, float]]] = {}
    for rec in timing:
        if rec.get("status") != "ok":
            continue
        if rec.get("method") != "exact_numpy_matmul":
            continue
        if rec.get("threads") != "all_threads":
            continue
        if rec.get("N") is None:
            continue
        med = (rec.get("latency_ms") or {}).get("median")
        if med is None:
            continue
        if rec.get("filtered"):
            key = f"filtered s={rec['selectivity']}"
        else:
            key = "unfiltered"
        series.setdefault(key, []).append((float(rec["N"]), float(med)))
    for key, pts in series.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=key)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (vectors)")
    ax.set_ylabel("Median per-query latency (ms), all threads")
    ax.set_title("Exact-search latency vs N (synthetic N labeled in JSON)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    p2 = FIGURES / "latency_vs_n.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(str(p2))

    # (3) routed pillar per question
    fig, ax = plt.subplots(figsize=(10, 6))
    rows = results["diagnostics"]["routing"]["per_question"]
    if rows:
        qs = [r["question"][:48] + ("…" if len(r["question"]) > 48 else "") for r in rows]
        pillars = [r["top_pillar"] for r in rows]
        colors_p = {"E": "#2ca02c", "S": "#1f77b4", "G": "#ff7f0e"}
        bar_colors = [colors_p.get(p, "gray") for p in pillars]
        hatches = ["//" if r["set_type"] == "extra" else "" for r in rows]
        bars = ax.barh(range(len(rows)), [1] * len(rows), color=bar_colors)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(qs, fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel("Routed pillar (solid=core, hatched=extra)")
        ax.invert_yaxis()
        ax.set_title("Top routed pillar per question")
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=colors_p["E"], label="E"),
                Patch(facecolor=colors_p["S"], label="S"),
                Patch(facecolor=colors_p["G"], label="G"),
                Patch(facecolor="white", edgecolor="black", hatch="//", label="extra"),
            ],
            loc="lower right",
        )
    p3 = FIGURES / "routed_pillar_per_question.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    written.append(str(p3))
    return written


def _fmt_ci(cell: dict | None, digits: int = 3) -> str:
    if not cell or cell.get("mean") is None:
        return "NA"
    mean = cell["mean"]
    lo = cell.get("ci95_low")
    hi = cell.get("ci95_high")
    if lo is None or hi is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _get(d: dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_summary_from_json(json_path: Path) -> Path:
    data = json.loads(json_path.read_text())
    lines: list[str] = []
    ds = data["dataset"]
    chosen = data["retrieval"]["chosen_setting"]
    frozen = data["retrieval"]["frozen_metrics"]

    lines.append("# ClimRetrieve hierarchy-guided retrieval pilot")
    lines.append("")
    lines.append("This file is generated from `results/pilot_results.json`. Every number is copied from that JSON.")
    lines.append("")
    lines.append("## Dataset stats")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| reports | {ds['n_reports']} |")
    lines.append(f"| chunks | {ds['n_chunks']} |")
    lines.append(
        f"| chunks/report min/median/max | {ds['chunks_per_report']['min']} / "
        f"{ds['chunks_per_report']['median']} / {ds['chunks_per_report']['max']} |"
    )
    lines.append(f"| unique questions (all) | {ds['n_unique_questions_all']} |")
    lines.append(f"| unique core questions | {ds['n_unique_core_questions']} |")
    lines.append(f"| unique extra questions | {ds['n_unique_extra_questions']} |")
    lines.append(f"| queries (all) | {ds['n_queries_all']} |")
    lines.append(f"| queries (core) | {ds['n_queries_core']} |")
    lines.append(f"| queries (extra) | {ds['n_queries_extra']} |")
    lines.append(
        f"| gold chunks/query (core, rel≥2) min/median/max | "
        f"{ds['core_questions']['gold_chunks_per_query_ge2_all_queries']['min']} / "
        f"{ds['core_questions']['gold_chunks_per_query_ge2_all_queries']['median']} / "
        f"{ds['core_questions']['gold_chunks_per_query_ge2_all_queries']['max']} |"
    )
    lines.append(
        f"| core queries excluded (zero gold, rel≥2) | "
        f"{ds['core_questions']['n_queries_excluded_from_eval']} |"
    )
    lines.append(
        f"| extra queries excluded (zero gold, rel≥2) | "
        f"{ds['extra_questions']['n_queries_excluded_from_eval']} |"
    )
    lines.append(
        f"| core-dev questions | {len(data['splits']['core_dev'])} |"
    )
    lines.append(
        f"| core-test questions | {len(data['splits']['core_test'])} |"
    )
    lines.append(
        f"| frozen setting (tuned on core-dev) | level={chosen['level']}, λ={chosen['lambda']}, τ={chosen['tau']} |"
    )
    lines.append("")
    lines.append("## Extra-question coverage")
    lines.append("")
    lines.append("| question | #reports | #queries | #gold chunks (rel≥2) | #zero-gold queries | dropped |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in ds["extra_coverage"]:
        lines.append(
            f"| {row['question']} | {row['n_reports']} | {row['n_queries']} | "
            f"{row['n_gold_chunks_ge2']} | {row['n_queries_zero_gold']} | {row['dropped']} |"
        )
    dropped = ds.get("dropped_extra_questions") or []
    lines.append("")
    if dropped:
        lines.append("Dropped extra questions (zero gold chunks across all reports):")
        for q in dropped:
            lines.append(f"- {q}")
    else:
        lines.append(
            f"Dropped extra questions (zero gold chunks across all reports): {dropped}"
        )
    lines.append("")

    def _table(split_key: str, title: str, note: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(note)
        lines.append("")
        lines.append(
            "| config | Recall@5 | Recall@10 | Recall@20 | miss rate | selectivity | latency median / p95 (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        block = frozen.get(split_key, {})
        for cfg in ("A", "B", "D", "E"):
            m = block.get(cfg, {})
            if not m or m.get("n_queries") == 0:
                lines.append(f"| {cfg} | NA | NA | NA | NA | NA | NA |")
                continue
            lat = m.get("latency_ms") or {}
            lines.append(
                f"| {cfg} | {_fmt_ci(m.get('recall@5'))} | {_fmt_ci(m.get('recall@10'))} | "
                f"{_fmt_ci(m.get('recall@20'))} | {_fmt_ci(m.get('miss_rate'))} | "
                f"{_fmt_ci(m.get('selectivity'))} | {_fmt(lat.get('median'), 3)} / {_fmt(lat.get('p95'), 3)} |"
            )
        nq = _get(block, "B", "n_queries", default=_get(block, "A", "n_queries", default="NA"))
        lines.append("")
        lines.append(f"Eligible queries in this split: {nq}.")
        lines.append("")

    _table(
        "core_test",
        "Main results (PRIMARY: core-test)",
        "Frozen λ/τ/level from core-dev. Means with 95% bootstrap CIs over queries. Do not pool with extra.",
    )
    _table(
        "extra",
        "Secondary results (extra questions)",
        "Same frozen settings. Extra questions only; not pooled with core-test.",
    )
    _table(
        "core_dev",
        "Tuning-data results (core-dev)",
        "These numbers were used to pick λ/τ/level. They are not a test estimate.",
    )

    lines.append("## Miss attribution (gold missed by B)")
    lines.append("")
    attr = data["retrieval"]["miss_attribution"]
    lines.append("| split | gold missed by B | routing error (in E) | chunk-labeling error (not in E) | queries with any miss |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in ("core_test", "extra", "core_dev"):
        a = attr.get(split, {})
        lines.append(
            f"| {split} | {a.get('n_gold_missed_by_B', 'NA')} | "
            f"{a.get('routing_error_gold_in_E', 'NA')} | "
            f"{a.get('chunk_labeling_error_gold_missing_from_E', 'NA')} | "
            f"{a.get('n_queries_where_B_missed_any_gold', 'NA')} |"
        )
    lines.append("")
    if attr.get("core_test", {}).get("definition"):
        lines.append(attr["core_test"]["definition"])
        lines.append("")

    lines.append("## Timing pilot")
    lines.append("")
    timing = data["timing"]
    lines.append(
        f"Queries timed per setting: {timing['n_queries_timed']} after {timing['warmup']} warm-up. "
        f"Real corpus N={timing['real_corpus_size']}."
    )
    lines.append("")
    lines.append("| threads | N | synthetic | method | efSearch | filtered | selectivity | median ms | p95 ms | status |")
    lines.append("|---|---:|---|---|---:|---|---:|---:|---:|---|")
    for rec in timing["settings"]:
        lat = rec.get("latency_ms") or {}
        lines.append(
            f"| {rec.get('threads')} | {rec.get('N')} | {rec.get('synthetic')} | "
            f"{rec.get('method')} | {rec.get('efSearch')} | {rec.get('filtered')} | {rec.get('selectivity')} | "
            f"{_fmt(lat.get('median'), 4)} | {_fmt(lat.get('p95'), 4)} | {rec.get('status')} |"
        )
    lines.append("")
    lines.append("Per-query saving (unfiltered exact median − filtered exact median):")
    lines.append("")
    lines.append("| threads | N | synthetic | s | unfiltered ms | filtered ms | saving ms | exceeds 1 ms |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for s in timing["savings_filtered_vs_unfiltered_exact"]:
        lines.append(
            f"| {s['threads']} | {s['N']} | {s['synthetic']} | {s['selectivity']} | "
            f"{_fmt(s['unfiltered_median_ms'], 4)} | {_fmt(s['filtered_median_ms'], 4)} | "
            f"{_fmt(s['saving_ms'], 4)} | {s['exceeds_1ms']} |"
        )
    where = timing["where_saving_exceeds_1ms"]
    lines.append("")
    lines.append(
        f"Smallest N where any filtered-vs-unfiltered exact saving exceeded 1 ms: "
        f"{where.get('min_N')}. {where.get('note')}"
    )
    if timing.get("failures"):
        lines.append("")
        lines.append("Timing failures / skipped settings:")
        for f in timing["failures"]:
            lines.append(f"- {f}")
    lines.append("")

    lines.append("## Routing diagnostic")
    lines.append("")
    route = data["diagnostics"]["routing"]
    lines.append("| question | set | top pillar | top leaf |")
    lines.append("|---|---|---|---|")
    for r in route["per_question"]:
        lines.append(
            f"| {r['question']} | {r['set_type']} | {r['top_pillar']} ({r['top_pillar_name']}) | "
            f"{r['top_leaf_name']} |"
        )
    lines.append("")
    lines.append(
        f"Core questions route to {route['core']['n_distinct_pillars']} distinct pillar(s) "
        f"{route['core']['distinct_pillars']}; "
        f"{route['core']['n_questions_sharing_the_modal_pillar']} of "
        f"{route['core']['n_questions']} share modal pillar {route['core']['modal_pillar']}."
    )
    lines.append(
        f"Extra questions route to {route['extra']['n_distinct_pillars']} distinct pillar(s) "
        f"{route['extra']['distinct_pillars']}; "
        f"{route['extra']['n_questions_sharing_the_modal_pillar']} of "
        f"{route['extra']['n_questions']} share modal pillar {route['extra']['modal_pillar']}."
    )
    lines.append("")
    assign = data["diagnostics"]["chunk_assignment"]
    lines.append(
        f"At frozen τ={chosen['tau']}, mean leaves per chunk = {assign['mean_leaves_per_chunk']} "
        f"(min {assign['min_leaves_per_chunk']}, max {assign['max_leaves_per_chunk']})."
    )
    lines.append("")
    lines.append("Chunks per leaf:")
    for leaf, n in assign["chunks_per_leaf"].items():
        lines.append(f"- {leaf}: {n}")
    lines.append("")

    # Slide-ready findings: core-test only, quote JSON numbers.
    ct = frozen.get("core_test", {})
    a = ct.get("A", {})
    b = ct.get("B", {})
    d = ct.get("D", {})
    e = ct.get("E", {})
    attr_ct = attr.get("core_test", {})
    lines.append("## Slide-ready findings")
    lines.append("")
    lines.append("Quoted from `retrieval.frozen_metrics.core_test` unless noted.")
    lines.append("")

    def _mean(block: dict, key: str) -> Any:
        return _get(block, key, "mean")

    b_rec10 = _mean(b, "recall@10")
    a_rec10 = _mean(a, "recall@10")
    b_sel = _mean(b, "selectivity")
    b_miss = _mean(b, "miss_rate")
    if b_rec10 is None or a_rec10 is None:
        lines.append(
            f"- Core-test Recall@10 was not available for a B vs A comparison "
            f"(B={_fmt(b_rec10)}, A={_fmt(a_rec10)})."
        )
    elif b_rec10 > a_rec10:
        lines.append(
            f"- On core-test, B Recall@10 was {_fmt(b_rec10)} vs A {_fmt(a_rec10)} "
            f"at B selectivity {_fmt(b_sel)}."
        )
    elif b_rec10 < a_rec10:
        lines.append(
            f"- On core-test, B did not beat A on Recall@10: B {_fmt(b_rec10)} vs A {_fmt(a_rec10)} "
            f"at B selectivity {_fmt(b_sel)}."
        )
    else:
        lines.append(
            f"- On core-test, B matched A on Recall@10 at {_fmt(b_rec10)} "
            f"with B selectivity {_fmt(b_sel)}."
        )
    lines.append(
        f"- B kept { _fmt(None if b_miss is None else (1.0 - b_miss), 3) } of gold chunks in its "
        f"candidate set on core-test (miss rate {_fmt(b_miss)}) at selectivity {_fmt(b_sel)}."
    )
    lines.append(
        f"- Oracle E core-test Recall@10 was {_fmt(_mean(e, 'recall@10'))} with miss rate "
        f"{_fmt(_mean(e, 'miss_rate'))}; D (flat leaves) Recall@10 was {_fmt(_mean(d, 'recall@10'))}."
    )
    lines.append(
        f"- Of {attr_ct.get('n_gold_missed_by_B', 'NA')} gold chunks missed by B on core-test, "
        f"{attr_ct.get('routing_error_gold_in_E', 'NA')} were in E (routing error) and "
        f"{attr_ct.get('chunk_labeling_error_gold_missing_from_E', 'NA')} were missing from E "
        f"(chunk-labeling error)."
    )
    where = timing["where_saving_exceeds_1ms"]
    lines.append(
        f"- Filtered vs unfiltered exact search first exceeded a 1 ms median saving at N="
        f"{where.get('min_N')} (see timing.where_saving_exceeds_1ms)."
    )
    lines.append(
        f"- Core questions used {route['core']['n_distinct_pillars']} distinct routed pillar(s); "
        f"{route['core']['n_questions_sharing_the_modal_pillar']} of {route['core']['n_questions']} "
        f"routed to {route['core']['modal_pillar']}."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in data.get("caveats", []):
        lines.append(f"- {c}")
    hw = data.get("environment", {})
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- embedding model: {hw.get('embed_model_name')}")
    lines.append(f"- seed: {data.get('seed')}")
    lines.append(f"- CPU: {hw.get('cpu_model')} ({hw.get('cpu_cores')} cores), RAM {hw.get('ram_gb')} GB")
    lines.append(f"- Python {hw.get('python')} on {hw.get('platform')}")
    lines.append(f"- library versions: {hw.get('library_versions')}")
    if data.get("notes"):
        lines.append("")
        lines.append("## Notes from the run")
        lines.append("")
        for n in data["notes"]:
            lines.append(f"- {n}")
    lines.append("")

    out = RESULTS / "summary.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out
