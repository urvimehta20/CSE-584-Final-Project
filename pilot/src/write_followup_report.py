"""Render figures and REPORT.md strictly from follow-up numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import FIGURES, RESULTS


def _fmt(v: Any, d: int = 3) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _ci(cell: dict | None, d: int = 3) -> str:
    if not cell or cell.get("mean") is None:
        return "NA"
    lo, hi = cell.get("ci95_low"), cell.get("ci95_high")
    if lo is None or hi is None:
        return _fmt(cell["mean"], d)
    return f"{cell['mean']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_figures(payload: dict, heldout_rows: dict, gold_rows: list[dict], exp2: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    markers = {"A": "o", "B_leaf": "s", "B_pillar": "^", "D_matched": "D"}
    colors = {"A": "#4C72B0", "B_leaf": "#DD8452", "B_pillar": "#C44E52", "D_matched": "#55A868"}

    # 1. Recall@50 vs selectivity, setting G (and R as labeled extras)
    fig, ax = plt.subplots(figsize=(8, 6))
    for setting, mk_edge in (("G", "black"), ("R", "gray")):
        cfgs = _get(payload, "experiment_1", "pooled_heldout", setting, "configs") or {}
        for cfg, rec in cfgs.items():
            rec50 = _get(rec, "recall@50", "mean")
            sel = _get(rec, "selectivity", "mean")
            if rec50 is None or sel is None:
                continue
            ax.scatter(
                [sel],
                [rec50],
                marker=markers.get(cfg, "x"),
                c=colors.get(cfg, "gray"),
                edgecolors=mk_edge,
                s=80 if setting == "G" else 50,
                label=f"{cfg} ({setting})",
                zorder=3,
            )
            ax.annotate(f"{cfg}/{setting}", (sel, rec50), fontsize=7)
    ax.set_xlabel("Selectivity (mean |candidates| / |universe|)")
    ax.set_ylabel("Recall@50 (pooled held-out)")
    ax.set_title("Recall@50 vs selectivity (G primary; R outlined in gray)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    p1 = FIGURES / "followup_recall50_vs_selectivity.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    written.append(str(p1))

    # 2. Recall vs cumulative prompt tokens
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    budgets = [500, 1000, 2000, 4000, 8000]
    for ax, setting in zip(axes, ("R", "G")):
        br = _get(payload, "experiment_2", "settings", setting, "budget_recall") or {}
        for cfg in ("A", "B_pillar", "D_matched"):
            xs, ys = [], []
            for b in budgets:
                m = _get(br, cfg, str(b), "mean")
                if m is None:
                    continue
                xs.append(b)
                ys.append(m)
            if xs:
                ax.plot(xs, ys, marker=markers.get(cfg, "o"), color=colors.get(cfg), label=cfg)
        ax.set_title(f"Setting {setting}")
        ax.set_xlabel("Token budget")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Recall (pooled held-out)")
    fig.suptitle("Recall vs cumulative prompt tokens")
    p2 = FIGURES / "followup_recall_vs_tokens.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    written.append(str(p2))

    # 3. A-rank percentile pruned vs kept
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, setting in zip(axes, ("R", "G")):
        pruned = [r["A_rank_percentile"] for r in gold_rows if r["setting"] == setting and r["pruned_by_B"] and r["A_rank_percentile"] is not None]
        kept = [r["A_rank_percentile"] for r in gold_rows if r["setting"] == setting and (not r["pruned_by_B"]) and r["A_rank_percentile"] is not None]
        data = [kept, pruned]
        ax.boxplot(data, tick_labels=["kept", "pruned"], showfliers=False)
        ax.set_title(f"Setting {setting} (n kept={len(kept)}, pruned={len(pruned)})")
        ax.set_ylabel("A-rank percentile (1 = best)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("A-rank percentile of gold chunks pruned vs kept by frozen-best B")
    p3 = FIGURES / "followup_prank_percentile.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    written.append(str(p3))
    print(f"Wrote figures: {written}")
    return written


def write_report(json_path: Path) -> Path:
    data = json.loads(Path(json_path).read_text())
    e1 = data["experiment_1"]
    e2 = data["experiment_2"]
    e3 = data["experiment_3"]
    rules = data["rules"]
    ds = data["dataset_preprocessing"]
    lines: list[str] = []

    def table_retrieval(block: dict, title: str, note: str) -> None:
        lines.append(f"#### {title}")
        lines.append("")
        lines.append(note)
        lines.append("")
        lines.append(
            "| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        cfgs = block.get("configs") or {}
        for cfg in ("A", "B_leaf", "B_pillar", "D_matched"):
            m = cfgs.get(cfg, {})
            lat = m.get("latency_ms") or {}
            lines.append(
                f"| {cfg} | {_ci(m.get('capped_recall@50'))} | {_ci(m.get('recall@20'))} | "
                f"{_ci(m.get('recall@50'))} | {_ci(m.get('recall@100'))} | {_ci(m.get('miss_rate'))} | "
                f"{_ci(m.get('selectivity'))} | {_fmt(lat.get('median'), 3)} / {_fmt(lat.get('p95'), 3)} |"
            )
        lines.append("")
        diffs = block.get("paired_diffs") or {}
        if diffs:
            lines.append("Paired differences on capped Recall@50 (B or D minus A):")
            lines.append("")
            lines.append("| contrast | mean diff | 95% CI | CI entirely > 0 |")
            lines.append("|---|---|---|---|")
            for name, d in diffs.items():
                cell = d.get("capped_recall@50") or {}
                lines.append(
                    f"| {name} | {_fmt(cell.get('mean_diff'))} | "
                    f"[{_fmt(cell.get('ci95_low'))}, {_fmt(cell.get('ci95_high'))}] | "
                    f"{cell.get('ci_entirely_above_0')} |"
                )
            lines.append("")

    # --- Summary ---
    g = e1["pooled_heldout"]["G"]
    r = e1["pooled_heldout"]["R"]
    best_g = g["frozen_best_B"]
    best_r = r["frozen_best_B"]
    g_a = _ci(_get(g, "configs", "A", "capped_recall@50"))
    g_b = _ci(_get(g, "configs", best_g, "capped_recall@50"))
    g_diff = _get(g, "paired_diffs", f"{best_g}_minus_A", "capped_recall@50") or {}
    g_sel_b = _ci(_get(g, "configs", best_g, "selectivity"))
    g_sel_a = _ci(_get(g, "configs", "A", "selectivity"))
    tok_g = _get(e2, "settings", "G", "paired_vs_A", best_g, "0.8") or {}
    tok_r = _get(e2, "settings", "R", "paired_vs_A", best_r, "0.8") or {}
    pr_g = e3["settings"]["G"]
    pr_r = e3["settings"]["R"]

    lines.append("# Follow-up report: when is hierarchy-guided retrieval worthwhile?")
    lines.append("")
    lines.append("Rendered from `results/followup_results.json`. Every number below is copied from that file.")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(
        f"In corpus-wide setting G, frozen-best B ({best_g}) had macro-averaged capped Recall@50 "
        f"{g_b} versus A {g_a}; the paired difference was {_fmt(g_diff.get('mean_diff'))} "
        f"[{_fmt(g_diff.get('ci95_low'))}, {_fmt(g_diff.get('ci95_high'))}]."
    )
    lines.append(
        f"Still in G, {best_g} selectivity was {g_sel_b} against A's {g_sel_a}, "
        f"and D-matched (flat tags at B-pillar's candidate count) capped Recall@50 was "
        f"{_ci(_get(g, 'configs', 'D_matched', 'capped_recall@50'))}."
    )
    lines.append(
        f"At equal recall r=0.8, the paired relative token saving of {best_g} versus A was "
        f"{_ci(tok_r.get('relative_saving'))} in setting R (n_paired={tok_r.get('n_paired')}) and "
        f"{_ci(tok_g.get('relative_saving'))} in setting G (n_paired={tok_g.get('n_paired')})."
    )
    lines.append(
        f"Of gold chunks pruned by frozen-best B, the share that A had placed in its top-10 was "
        f"{_fmt(pr_r.get('share_of_pruned_gold_in_A_top10'))} in setting R "
        f"({pr_r.get('n_pruned_gold_in_A_top10')} chunks) and "
        f"{_fmt(pr_g.get('share_of_pruned_gold_in_A_top10'))} in setting G "
        f"({pr_g.get('n_pruned_gold_in_A_top10')} chunks)."
    )
    lines.append(
        f"The one-sided Mann–Whitney p-value for pruned gold having worse A-rank percentiles than kept gold "
        f"was {_fmt(_get(pr_r, 'mannwhitneyu', 'p_value'), 4)} in R and "
        f"{_fmt(_get(pr_g, 'mannwhitneyu', 'p_value'), 4)} in G."
    )
    lines.append(f"Mechanical verdict: {rules['verdict']['text']}.")
    lines.append("")

    lines.append("## 2. Approach")
    lines.append("")
    lines.append(
        "Experiment 1 asks whether routing a climate question to ESG topic nodes, then searching only those "
        "nodes, recovers the same gold evidence as searching the whole collection. Setting R keeps the pilot’s "
        "per-report search. Setting G removes the report filter so the corpus is all 8,464 paragraphs; gold is "
        "the union of relevance≥2 chunks across reports annotated for that question, and other chunks are unjudged."
    )
    lines.append(
        "B-pillar uses the tree: a selected pillar’s candidates are the union of its children’s chunks. "
        "B-leaf repeats the pilot’s leaf routing. D-matched is the fair flat-tag control: leaves are added in "
        "query-score order until the candidate count first reaches B-pillar’s count for that unit."
    )
    lines.append(
        "Experiment 2 converts each ranked list into prompt tokens (chunk tokens plus 8 overhead tokens) and "
        "asks whether B reaches a target recall with fewer tokens than A. Experiment 3 checks whether the gold "
        "chunks B drops were already hard for A (low A-rank) or were evidence A would have shown in its top-10."
    )
    lines.append("")

    lines.append("## 3. Technical details")
    lines.append("")
    lines.append(
        f"ClimRetrieve report-level data: {ds['n_report_level_rows_raw']} raw rows, "
        f"{ds['n_empty_paragraph_rows_dropped']} empty-paragraph rows dropped, "
        f"{ds['exact_text_dedup_collapsed']} exact-text within-report collapse, "
        f"{ds['n_chunks']} chunks in {ds['n_reports']} reports. "
        f"Embeddings: {ds['embedding_model']}, L2-normalized, documents unprefixed, queries prefixed with the "
        f"BGE instruction. Cached under `data/processed/embeddings/`."
    )
    tax = data["taxonomy"]
    ca = data["chunk_assignment"]
    lines.append(
        f"Taxonomy: root → pillars {tax['pillar_ids']} → {len(tax['leaf_ids'])} leaves {tax['leaf_ids']}. "
        f"Soft assignment: cosine ≥ τ plus forced top-1 leaf; parents inherit children. "
        f"At fold-0 B-pillar τ={_fmt(_get(ca, 'G', 'tau'), 4)}, mean leaves per chunk in G assignment = "
        f"{_fmt(_get(ca, 'G', 'mean_leaves_per_chunk'))}."
    )
    lines.append(
        "Configs: A = exact search over the setting’s universe (report chunks in R; all 8,464 in G). "
        "B_leaf = top-λ leaves, λ ∈ {1,2,3}. B_pillar = top-λ_p pillars, λ_p ∈ {1,2}. "
        "D_matched = greedy leaves until |candidates| ≥ |B_pillar candidates|. "
        "E_human was not run (no `data/question_topics.csv`)."
    )
    tok = data["tokenizer"]
    lines.append(
        f"Tokenizer: tiktoken `{tok['name']}`; per-chunk overhead = {tok['per_chunk_overhead']} tokens. "
        f"Two-fold cross-fit over the 16 core questions (seed {data['seed']}), tune then swap. "
        f"Cluster bootstrap resamples questions ({data['n_bootstrap']} replicates), never lone queries. "
        "Setting G averages are macro over questions; setting R point estimates average over queries. "
        "Paired differences use the same resampled questions. "
        "Mann–Whitney U is one-sided on A-rank percentiles (pruned > kept). "
        f"Hardware: {data['environment'].get('cpu_model')}, {data['environment'].get('cpu_cores')} cores, "
        f"{data['environment'].get('ram_gb')} GB RAM, Python {data['environment'].get('python')}."
    )
    lines.append(
        "Primary retrieval metric: capped Recall@50 = |gold ∩ top-50| / min(50, |gold|). "
        "Primary token target: r=0.8. Small-gold questions (3, 4, and 8 gold chunks) are kept and also reported separately."
    )
    lines.append("")

    lines.append("## 4. Results")
    lines.append("")
    lines.append("### 4.1 Experiment 1 — retrieval")
    lines.append("")
    lines.append(
        f"Folds: fold 0 tunes on the original 8 core-dev questions and tests the original 8 core-test questions; "
        f"fold 1 swaps. Frozen-best B on pooled held-out capped Recall@50 was {best_g} in G and {best_r} in R."
    )
    lines.append("")
    table_retrieval(
        e1["pooled_heldout"]["G"],
        "Setting G pooled held-out (primary, 16 core questions, macro over questions)",
        "Universe = 8,464 chunks. Unjudged chunks are not treated as non-relevant for precision.",
    )
    table_retrieval(
        e1["pooled_heldout"]["R"],
        "Setting R pooled held-out (per-report queries; cluster bootstrap over questions)",
        "Each query searches only its report. Queries with zero gold at relevance ≥ 2 are excluded.",
    )
    table_retrieval(
        e1["original_core_test"]["G"],
        "Setting G original core-test (continuity with the pilot split)",
        "Fold 0 test questions only.",
    )
    table_retrieval(
        e1["original_core_test"]["R"],
        "Setting R original core-test (continuity with the pilot split)",
        "Fold 0 test queries only.",
    )
    table_retrieval(
        e1["pooled_heldout"]["G"]["small_gold_questions"],
        "Setting G small-gold slice (3 questions with <10 gold; not dropped from the main table)",
        f"Questions: {e1['pooled_heldout']['G']['small_gold_question_list']}",
    )
    table_retrieval(
        e1["pooled_heldout"]["R"]["small_gold_questions"],
        "Setting R small-gold slice (queries of those same 3 questions)",
        "Same three questions; still included in the full R table.",
    )
    table_retrieval(
        e1["extra"]["G"],
        "Setting G extra questions (secondary, never pooled)",
        e1["extra"]["G"].get("note", ""),
    )
    table_retrieval(
        e1["extra"]["R"],
        "Setting R extra questions (secondary, never pooled)",
        e1["extra"]["R"].get("note", ""),
    )
    lines.append("Figure: `results/figures/followup_recall50_vs_selectivity.png`.")
    lines.append("")

    lines.append("### 4.2 Experiment 2 — prompt tokens at equal recall")
    lines.append("")
    lines.append(
        f"Tokenizer `{tok['name']}`; +{tok['per_chunk_overhead']} tokens per chunk. "
        f"Primary r={data['primary_r']}. Token costs in G include unjudged chunks and are an upper bound."
    )
    lines.append("")
    for setting in ("R", "G"):
        lines.append(f"#### Setting {setting}")
        lines.append("")
        lines.append(_get(e2, "settings", setting, "note_unjudged") or "")
        lines.append("")
        lines.append("| config | reachable r=0.5 | reachable r=0.8 | reachable r=1.0 | tokens@0.8 (reachable only) |")
        lines.append("|---|---|---|---|---|")
        for cfg in ("A", "B_leaf", "B_pillar", "D_matched"):
            lines.append(
                f"| {cfg} | {_ci(_get(e2, 'settings', setting, 'reachable', cfg, '0.5'))} | "
                f"{_ci(_get(e2, 'settings', setting, 'reachable', cfg, '0.8'))} | "
                f"{_ci(_get(e2, 'settings', setting, 'reachable', cfg, '1.0'))} | "
                f"{_ci(_get(e2, 'settings', setting, 'configs', cfg, 'tokens_r0.8_reachable_only'))} |"
            )
        lines.append("")
        lines.append("Paired token comparisons (both configs reach r):")
        lines.append("")
        lines.append("| contrast | r | n_paired | tokens A | tokens B | relative saving | ≥15% and CI low>0 |")
        lines.append("|---|---:|---:|---|---|---|---|")
        for cfg in ("B_leaf", "B_pillar", "D_matched"):
            for rtarget in ("0.5", "0.8", "1.0"):
                p = _get(e2, "settings", setting, "paired_vs_A", cfg, rtarget) or {}
                rel = p.get("relative_saving") or {}
                flag = bool(p.get("rule_ge_15pct") and rel.get("ci95_low") is not None and rel["ci95_low"] > 0)
                lines.append(
                    f"| {cfg} − A | {rtarget} | {p.get('n_paired', 'NA')} | {_ci(p.get('tokens_A'))} | "
                    f"{_ci(p.get('tokens_B'))} | {_ci(rel)} | {flag} |"
                )
        lines.append("")
        lines.append("Recall at fixed token budgets:")
        lines.append("")
        lines.append("| config | 500 | 1000 | 2000 | 4000 | 8000 |")
        lines.append("|---|---|---|---|---|---|")
        for cfg in ("A", "B_leaf", "B_pillar", "D_matched"):
            cells = [_ci(_get(e2, "settings", setting, "budget_recall", cfg, str(b))) for b in (500, 1000, 2000, 4000, 8000)]
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Figure: `results/figures/followup_recall_vs_tokens.png`.")
    lines.append("")

    lines.append("### 4.3 Experiment 3 — where pruned gold ranks under A")
    lines.append("")
    lines.append("| setting | best B | n gold | n pruned | pruned median A-rank (pct) | kept median A-rank (pct) | pruned in A top-10 | share of pruned in A top-10 | MW p |")
    lines.append("|---|---|---:|---:|---|---|---:|---|---|")
    for setting in ("R", "G"):
        b = e3["settings"][setting]
        pr, ke = b.get("pruned") or {}, b.get("kept") or {}
        lines.append(
            f"| {setting} | {b.get('frozen_best_B')} | {b.get('n_gold')} | {b.get('n_pruned')} | "
            f"{_fmt(pr.get('median_A_rank'))} ({_fmt(pr.get('median_A_rank_percentile'))}) | "
            f"{_fmt(ke.get('median_A_rank'))} ({_fmt(ke.get('median_A_rank_percentile'))}) | "
            f"{b.get('n_pruned_gold_in_A_top10')} | {_fmt(b.get('share_of_pruned_gold_in_A_top10'))} | "
            f"{_fmt(_get(b, 'mannwhitneyu', 'p_value'), 4)} |"
        )
    lines.append("")
    for setting in ("R", "G"):
        pr = e3["settings"][setting].get("pruned") or {}
        lines.append(
            f"Setting {setting} pruned gold by relevance: "
            f"rel=2 → {pr.get('n_relevance_2', 'NA')}, rel=3 → {pr.get('n_relevance_3', 'NA')}, "
            f"other/missing → {pr.get('n_relevance_other_or_missing', 'NA')}."
        )
    lines.append("")
    lines.append(
        f"Gold chunks that A ranked in its top-10 and B pruned are listed in "
        f"`results/pruned_top10_gold.csv` ({e3.get('n_pruned_top10_listed')} rows)."
    )
    lines.append("")
    lines.append("Figure: `results/figures/followup_prank_percentile.png`.")
    lines.append("")

    lines.append("## 5. Insights")
    lines.append("")
    lines.append("### Pre-registered rules")
    lines.append("")
    for setting in ("G", "R"):
        blk = rules[f"B_beats_A_retrieval_{setting}"]
        lines.append(
            f"- **B beats A on retrieval ({setting})**: frozen-best B={blk['frozen_best_B']} is "
            f"{'MET' if blk['frozen_best_MET'] else 'NOT MET'} "
            f"(B_leaf MET={blk['B_leaf']['MET']}, B_pillar MET={blk['B_pillar']['MET']}; "
            f"frozen-best Δ capped R@50 = {_fmt(blk[blk['frozen_best_B']]['mean_diff'])} "
            f"[{_fmt(blk[blk['frozen_best_B']]['ci95_low'])}, {_fmt(blk[blk['frozen_best_B']]['ci95_high'])}])."
        )
    for setting in ("R", "G"):
        blk = rules[f"B_saves_meaningful_tokens_{setting}"]
        rel = blk[blk["frozen_best_B"]]["relative_saving"]
        lines.append(
            f"- **B saves meaningful tokens ({setting}, r=0.8)**: frozen-best B={blk['frozen_best_B']} is "
            f"{'MET' if blk['frozen_best_MET'] else 'NOT MET'} "
            f"(relative saving {_ci(rel)}, n_paired={blk[blk['frozen_best_B']].get('n_paired')})."
        )
    for setting in ("R", "G"):
        blk = rules[f"pruning_errors_fall_on_hard_evidence_{setting}"]
        lines.append(
            f"- **Pruning errors fall on hard evidence ({setting})**: "
            f"{'MET' if blk['MET'] else 'NOT MET'} "
            f"(MW p={_fmt(blk['mannwhitneyu_p'], 4)}, share of pruned gold in A top-10="
            f"{_fmt(blk['share_of_pruned_gold_in_A_top10'])})."
        )
    lines.append("")
    lines.append("### Numbered insights")
    lines.append("")
    # Insights must cite JSON numbers and hedge when CI includes 0.
    d_g = g_diff
    hedge_g = "no evidence that" if not (d_g.get("ci_entirely_above_0")) else "consistent with"
    lines.append(
        f"1. There is {hedge_g} B beating A on the primary retrieval metric in setting G: "
        f"paired Δ capped Recall@50 = {_fmt(d_g.get('mean_diff'))} "
        f"[{_fmt(d_g.get('ci95_low'))}, {_fmt(d_g.get('ci95_high'))}] "
        f"(experiment_1.pooled_heldout.G.paired_diffs)."
    )
    d_r = _get(r, "paired_diffs", f"{best_r}_minus_A", "capped_recall@50") or {}
    hedge_r = "no evidence that" if not d_r.get("ci_entirely_above_0") else "consistent with"
    lines.append(
        f"2. In the per-report setting the same comparison is {hedge_r} a retrieval win: "
        f"Δ capped Recall@50 = {_fmt(d_r.get('mean_diff'))} "
        f"[{_fmt(d_r.get('ci95_low'))}, {_fmt(d_r.get('ci95_high'))}]."
    )
    rel_r = tok_r.get("relative_saving") or {}
    hedge_tok_r = "no evidence of" if not (rel_r.get("ci95_low") is not None and rel_r["ci95_low"] > 0) else "consistent with"
    lines.append(
        f"3. At r=0.8 in setting R there is {hedge_tok_r} a token saving for frozen-best B: "
        f"relative saving {_ci(rel_r)} on n_paired={tok_r.get('n_paired')} "
        f"(experiment_2.settings.R.paired_vs_A)."
    )
    rel_g = tok_g.get("relative_saving") or {}
    hedge_tok_g = "no evidence of" if not (rel_g.get("ci95_low") is not None and rel_g["ci95_low"] > 0) else "consistent with"
    lines.append(
        f"4. At r=0.8 in setting G there is {hedge_tok_g} a token saving for frozen-best B: "
        f"relative saving {_ci(rel_g)} on n_paired={tok_g.get('n_paired')}. "
        f"Those G token totals are an upper bound because unjudged chunks count as cost."
    )
    lines.append(
        f"5. D-matched versus B-pillar tests tree structure against a matched flat budget: "
        f"in G, D-matched capped Recall@50 was {_ci(_get(g, 'configs', 'D_matched', 'capped_recall@50'))} "
        f"and B-pillar was {_ci(_get(g, 'configs', 'B_pillar', 'capped_recall@50'))} "
        f"at selectivities {_ci(_get(g, 'configs', 'D_matched', 'selectivity'))} and "
        f"{_ci(_get(g, 'configs', 'B_pillar', 'selectivity'))}."
    )
    lines.append(
        f"6. Setting G pruned {pr_g.get('n_pruned')} gold chunks, {pr_g.get('n_pruned_gold_in_A_top10')} of which "
        f"A had ranked in its top-10 (share {_fmt(pr_g.get('share_of_pruned_gold_in_A_top10'))}); "
        f"the corresponding R counts are {pr_r.get('n_pruned')} pruned and "
        f"{pr_r.get('n_pruned_gold_in_A_top10')} in A's top-10. "
        f"Inspect `results/pruned_top10_gold.csv`."
    )
    lines.append("")

    lines.append("## 6. Verdict")
    lines.append("")
    lines.append(rules["verdict"]["text"])
    lines.append("")
    lines.append(
        f"Separately, “pruning errors fall on hard evidence” is "
        f"{'MET' if rules['verdict']['pruning_hard_evidence_R'] else 'NOT MET'} in setting R and "
        f"{'MET' if rules['verdict']['pruning_hard_evidence_G'] else 'NOT MET'} in setting G."
    )
    lines.append("")

    lines.append("## 7. What this does not show")
    lines.append("")
    lines.append(
        "- Setting G leaves most of the 8,464 chunks unjudged for any given question; recall is valid, "
        "but we do not report precision, and token costs are an upper bound."
    )
    lines.append("- Only 16 core questions, all climate-related; extra questions are secondary and tiny (one report each).")
    lines.append("- Chunk-to-topic labels are automatic cosine assignments, not human-verified (config C / E_human not in this report).")
    lines.append("- Pilot synthetic timing at N=1e5/1e6 is not re-used here as evidence for retrieval quality.")
    if data.get("e_human", {}).get("ran") is False:
        lines.append(f"- E_human: {data['e_human'].get('reason')}")
    lines.append("- FAISS multi-thread crashes from the pilot timing run are irrelevant to these CPU exact-search experiments.")
    lines.append("")

    lines.append("## 8. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd pilot")
    lines.append("bash run_followup.sh")
    lines.append("```")
    lines.append("")
    lines.append(
        "Requires the existing processed parquet and cached embeddings from the pilot "
        "(`data/processed/`, `data/processed/embeddings/`). Does not re-download ClimRetrieve or re-embed "
        "unless those files are missing. Optional later: fill `data/question_topics.csv` and run "
        "`PYTHONPATH=src python src/run_e_human.py` (writes `results/e_human_results.json` only)."
    )
    lines.append("")
    lines.append(f"Generated at {data.get('generated_at_utc')}. Seed {data.get('seed')}.")
    lines.append("")

    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out
