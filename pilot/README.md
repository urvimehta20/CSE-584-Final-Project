# Hierarchy-guided retrieval pilot

Small CPU-only pilot for *When Is Hierarchy-Guided Vector Retrieval Worthwhile?*
It asks whether a hand-written ESG topic tree can pre-filter ClimRetrieve
paragraphs without losing gold evidence, and whether that filter is worth any
latency.

## Run

```bash
cd pilot
bash run_all.sh
```

Needs Python 3.10+, a network connection the first time (dataset clone +
`BAAI/bge-small-en-v1.5`), and a few GB of RAM. Everything is written under
`results/`. Re-running reuses cached embeddings in `data/processed/embeddings/`.

## Configurations

Each query searches **only that report’s chunks**.

| id | what it does |
|---|---|
| **A** | Baseline: vector-rank every paragraph in the report. |
| **B** | Route the query to the top-λ taxonomy nodes at a chosen level; keep only chunks assigned to those nodes (parents inherit children); then vector-rank. |
| **D** | Same as B, but routing and membership are **leaves only** (no parent/child). |
| **E** | Oracle: take the topic nodes that actually contain the query’s gold chunks, then vector-rank that set. Upper bound that separates routing error from chunk-labeling error. |

λ and τ are chosen on **core-dev** (8 of the 16 core questions). Frozen settings
are scored separately on **core-test** (primary) and the **extra** 10 questions
(secondary). Those two sets are never averaged together.

Config C (human-verified labels) is not run. `results/label_sample.csv` is a
150-chunk sheet for later double annotation.

## Outputs

- `results/pilot_results.json` — every measured number, seeds, versions, hardware
- `results/summary.md` — generated from that JSON (do not edit by hand)
- `results/figures/` — recall vs selectivity, latency vs N, routed pillar
- `results/label_sample.csv` — handoff for config C
