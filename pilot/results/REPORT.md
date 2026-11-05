# Follow-up report: when is hierarchy-guided retrieval worthwhile?

Rendered from `results/followup_results.json`. Every number below is copied from that file.

## 1. Summary

In corpus-wide setting G, frozen-best B (B_pillar) had macro-averaged capped Recall@50 0.103 [0.061, 0.147] versus A 0.103 [0.061, 0.147]; the paired difference was 0.000 [0.000, 0.000].
Still in G, B_pillar selectivity was 0.693 [0.646, 0.724] against A's 1.000 [1.000, 1.000], and D-matched (flat tags at B-pillar's candidate count) capped Recall@50 was 0.103 [0.061, 0.147].
At equal recall r=0.8, the paired relative token saving of B_pillar versus A was 0.050 [0.029, 0.072] in setting R (n_paired=103) and 0.020 [-0.008, 0.043] in setting G (n_paired=16).
Of gold chunks pruned by frozen-best B, the share that A had placed in its top-10 was 0.035 in setting R (2 chunks) and 0.000 in setting G (0 chunks).
The one-sided Mann–Whitney p-value for pruned gold having worse A-rank percentiles than kept gold was 0.0000 in R and 0.0000 in G.
Mechanical verdict: NEGATIVE: no retrieval or token benefit at this scale.

## 2. Approach

Experiment 1 asks whether routing a climate question to ESG topic nodes, then searching only those nodes, recovers the same gold evidence as searching the whole collection. Setting R keeps the pilot’s per-report search. Setting G removes the report filter so the corpus is all 8,464 paragraphs; gold is the union of relevance≥2 chunks across reports annotated for that question, and other chunks are unjudged.
B-pillar uses the tree: a selected pillar’s candidates are the union of its children’s chunks. B-leaf repeats the pilot’s leaf routing. D-matched is the fair flat-tag control: leaves are added in query-score order until the candidate count first reaches B-pillar’s count for that unit.
Experiment 2 converts each ranked list into prompt tokens (chunk tokens plus 8 overhead tokens) and asks whether B reaches a target recall with fewer tokens than A. Experiment 3 checks whether the gold chunks B drops were already hard for A (low A-rank) or were evidence A would have shown in its top-10.

## 3. Technical details

ClimRetrieve report-level data: 47941 raw rows, 15 empty-paragraph rows dropped, 1 exact-text within-report collapse, 8464 chunks in 32 reports. Embeddings: BAAI/bge-small-en-v1.5, L2-normalized, documents unprefixed, queries prefixed with the BGE instruction. Cached under `data/processed/embeddings/`.
Taxonomy: root → pillars ['E', 'S', 'G'] → 14 leaves ['e_physical_climate_risk', 'e_transition_risk', 'e_climate_adaptation', 'e_ghg_emissions', 'e_energy', 'e_water', 'e_biodiversity', 'e_waste', 's_workforce', 's_health_safety', 's_community', 'g_board_oversight', 'g_risk_management', 'g_ethics']. Soft assignment: cosine ≥ τ plus forced top-1 leaf; parents inherit children. At fold-0 B-pillar τ=0.5544, mean leaves per chunk in G assignment = 5.138.
Configs: A = exact search over the setting’s universe (report chunks in R; all 8,464 in G). B_leaf = top-λ leaves, λ ∈ {1,2,3}. B_pillar = top-λ_p pillars, λ_p ∈ {1,2}. D_matched = greedy leaves until |candidates| ≥ |B_pillar candidates|. E_human was not run (no `data/question_topics.csv`).
Tokenizer: tiktoken `o200k_base`; per-chunk overhead = 8 tokens. Two-fold cross-fit over the 16 core questions (seed 42), tune then swap. Cluster bootstrap resamples questions (2000 replicates), never lone queries. Setting G averages are macro over questions; setting R point estimates average over queries. Paired differences use the same resampled questions. Mann–Whitney U is one-sided on A-rank percentiles (pruned > kept). Hardware: Apple M4, 10 cores, 16.0 GB RAM, Python 3.12.13.
Primary retrieval metric: capped Recall@50 = |gold ∩ top-50| / min(50, |gold|). Primary token target: r=0.8. Small-gold questions (3, 4, and 8 gold chunks) are kept and also reported separately.

## 4. Results

### 4.1 Experiment 1 — retrieval

Folds: fold 0 tunes on the original 8 core-dev questions and tests the original 8 core-test questions; fold 1 swaps. Frozen-best B on pooled held-out capped Recall@50 was B_pillar in G and B_leaf in R.

#### Setting G pooled held-out (primary, 16 core questions, macro over questions)

Universe = 8,464 chunks. Unjudged chunks are not treated as non-relevant for precision.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.103 [0.061, 0.147] | 0.065 [0.035, 0.101] | 0.100 [0.058, 0.145] | 0.152 [0.095, 0.214] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.710 / 1.845 |
| B_leaf | 0.100 [0.058, 0.144] | 0.062 [0.034, 0.097] | 0.097 [0.056, 0.143] | 0.149 [0.093, 0.210] | 0.333 [0.227, 0.451] | 0.269 [0.213, 0.322] | 0.694 / 0.789 |
| B_pillar | 0.103 [0.061, 0.147] | 0.065 [0.035, 0.101] | 0.100 [0.058, 0.145] | 0.152 [0.095, 0.214] | 0.018 [0.005, 0.036] | 0.693 [0.646, 0.724] | 1.266 / 1.358 |
| D_matched | 0.103 [0.061, 0.147] | 0.065 [0.035, 0.101] | 0.100 [0.058, 0.145] | 0.152 [0.095, 0.214] | 0.037 [0.017, 0.059] | 0.749 [0.708, 0.781] | 1.826 / 2.193 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | -0.003 | [-0.016, 0.009] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

#### Setting R pooled held-out (per-report queries; cluster bootstrap over questions)

Each query searches only its report. Queries with zero gold at relevance ≥ 2 are excluded.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.790 [0.740, 0.837] | 0.586 [0.518, 0.658] | 0.790 [0.740, 0.837] | 0.933 [0.916, 0.954] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.025 / 0.104 |
| B_leaf | 0.762 [0.709, 0.817] | 0.577 [0.511, 0.648] | 0.762 [0.709, 0.817] | 0.881 [0.837, 0.928] | 0.095 [0.042, 0.142] | 0.629 [0.557, 0.710] | 0.024 / 0.074 |
| B_pillar | 0.732 [0.655, 0.795] | 0.551 [0.480, 0.625] | 0.732 [0.655, 0.795] | 0.872 [0.782, 0.941] | 0.092 [0.017, 0.188] | 0.711 [0.633, 0.775] | 0.025 / 0.072 |
| D_matched | 0.776 [0.720, 0.827] | 0.577 [0.509, 0.649] | 0.776 [0.720, 0.827] | 0.902 [0.874, 0.932] | 0.054 [0.025, 0.082] | 0.782 [0.715, 0.839] | 0.047 / 0.106 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | -0.028 | [-0.056, -0.001] | False |
| B_pillar_minus_A | -0.058 | [-0.142, -0.003] | False |
| D_matched_minus_A | -0.014 | [-0.035, 0.002] | False |

#### Setting G original core-test (continuity with the pilot split)

Fold 0 test questions only.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.117 [0.054, 0.184] | 0.079 [0.029, 0.140] | 0.113 [0.048, 0.181] | 0.182 [0.095, 0.278] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.744 / 1.812 |
| B_leaf | 0.117 [0.054, 0.184] | 0.079 [0.029, 0.140] | 0.113 [0.048, 0.181] | 0.183 [0.097, 0.279] | 0.158 [0.059, 0.246] | 0.359 [0.329, 0.388] | 0.730 / 0.790 |
| B_pillar | 0.117 [0.054, 0.184] | 0.079 [0.029, 0.140] | 0.113 [0.048, 0.181] | 0.182 [0.095, 0.278] | 0.012 [0.003, 0.023] | 0.693 [0.630, 0.724] | 1.269 / 1.298 |
| D_matched | 0.117 [0.054, 0.184] | 0.079 [0.029, 0.140] | 0.113 [0.048, 0.181] | 0.182 [0.095, 0.278] | 0.044 [0.009, 0.080] | 0.763 [0.726, 0.792] | 1.867 / 2.187 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | 0.000 | [0.000, 0.000] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

#### Setting R original core-test (continuity with the pilot split)

Fold 0 test queries only.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.807 [0.753, 0.868] | 0.627 [0.547, 0.713] | 0.807 [0.753, 0.868] | 0.939 [0.915, 0.968] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.022 / 0.097 |
| B_leaf | 0.791 [0.720, 0.867] | 0.623 [0.537, 0.716] | 0.791 [0.720, 0.867] | 0.913 [0.854, 0.966] | 0.060 [0.011, 0.119] | 0.680 [0.577, 0.780] | 0.025 / 0.056 |
| B_pillar | 0.807 [0.753, 0.868] | 0.627 [0.547, 0.713] | 0.807 [0.753, 0.868] | 0.952 [0.931, 0.970] | 0.009 [0.002, 0.016] | 0.781 [0.738, 0.830] | 0.027 / 0.071 |
| D_matched | 0.804 [0.744, 0.870] | 0.624 [0.540, 0.716] | 0.804 [0.744, 0.870] | 0.928 [0.903, 0.962] | 0.027 [0.004, 0.054] | 0.855 [0.825, 0.889] | 0.047 / 0.100 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | -0.016 | [-0.044, 0.000] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | -0.003 | [-0.012, 0.005] | False |

#### Setting G small-gold slice (3 questions with <10 gold; not dropped from the main table)

Questions: ['Does the company provide definitions for climate change adaptation?', 'Does the company report a plan to engage with downstream partners on water consumption or water pollution?', 'Does the company refer to any third party scenarios when identifying climate-related risks or opportunities (e.g. IPCC trajectories, NGFS scenarios, etc.)?']

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.125 [0.000, 0.375] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.679 / 1.684 |
| B_leaf | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.125 [0.000, 0.375] | 0.278 [0.000, 0.500] | 0.264 [0.090, 0.377] | 0.687 / 0.781 |
| B_pillar | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.125 [0.000, 0.375] | 0.000 [0.000, 0.000] | 0.641 [0.475, 0.724] | 1.168 / 1.228 |
| D_matched | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.083 [0.000, 0.250] | 0.125 [0.000, 0.375] | 0.000 [0.000, 0.000] | 0.730 [0.654, 0.781] | 1.674 / 2.033 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | 0.000 | [0.000, 0.000] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

#### Setting R small-gold slice (queries of those same 3 questions)

Same three questions; still included in the full R table.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.900 [0.500, 1.000] | 0.800 [0.500, 1.000] | 0.900 [0.500, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.018 / 0.060 |
| B_leaf | 0.800 [0.500, 1.000] | 0.700 [0.500, 0.833] | 0.800 [0.500, 1.000] | 0.900 [0.500, 1.000] | 0.100 [0.000, 0.500] | 0.749 [0.263, 0.882] | 0.023 / 0.055 |
| B_pillar | 0.800 [0.500, 1.000] | 0.700 [0.500, 0.833] | 0.800 [0.500, 1.000] | 0.900 [0.500, 1.000] | 0.100 [0.000, 0.500] | 0.641 [0.452, 0.813] | 0.021 / 0.050 |
| D_matched | 0.900 [0.500, 1.000] | 0.800 [0.500, 1.000] | 0.900 [0.500, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.748 [0.534, 0.877] | 0.035 / 0.089 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | -0.100 | [-0.500, 0.000] | False |
| B_pillar_minus_A | -0.100 | [-0.500, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

#### Setting G extra questions (secondary, never pooled)

Secondary. Frozen settings from fold 0 (original core-dev tune). Never pooled with core.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.054 [0.000, 0.137] | 0.048 [0.000, 0.131] | 0.054 [0.000, 0.137] | 0.085 [0.000, 0.174] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.950 / 4.069 |
| B_leaf | 0.054 [0.000, 0.137] | 0.048 [0.000, 0.131] | 0.054 [0.000, 0.137] | 0.085 [0.000, 0.174] | 0.322 [0.062, 0.649] | 0.337 [0.281, 0.393] | 0.728 / 0.826 |
| B_pillar | 0.054 [0.000, 0.137] | 0.048 [0.000, 0.131] | 0.054 [0.000, 0.137] | 0.085 [0.000, 0.174] | 0.010 [0.000, 0.031] | 0.738 [0.724, 0.768] | 1.541 / 1.742 |
| D_matched | 0.054 [0.000, 0.137] | 0.048 [0.000, 0.131] | 0.054 [0.000, 0.137] | 0.085 [0.000, 0.174] | 0.000 [0.000, 0.000] | 0.804 [0.764, 0.844] | 1.835 / 2.035 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | 0.000 | [0.000, 0.000] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

#### Setting R extra questions (secondary, never pooled)

Secondary. Frozen settings from fold 0 (original core-dev tune). Never pooled with core.

| config | capped R@50 (primary) | raw R@20 | raw R@50 | raw R@100 | miss rate | selectivity | latency med/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | 0.223 [0.072, 0.399] | 0.146 [0.042, 0.276] | 0.223 [0.072, 0.399] | 0.554 [0.307, 0.737] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.039 / 0.059 |
| B_leaf | 0.264 [0.083, 0.448] | 0.146 [0.042, 0.276] | 0.264 [0.083, 0.448] | 0.554 [0.307, 0.737] | 0.167 [0.000, 0.500] | 0.726 [0.566, 0.864] | 0.039 / 0.051 |
| B_pillar | 0.223 [0.072, 0.399] | 0.146 [0.042, 0.276] | 0.223 [0.072, 0.399] | 0.554 [0.307, 0.737] | 0.010 [0.000, 0.031] | 0.883 [0.835, 0.907] | 0.043 / 0.049 |
| D_matched | 0.223 [0.072, 0.399] | 0.146 [0.042, 0.276] | 0.223 [0.072, 0.399] | 0.554 [0.307, 0.737] | 0.000 [0.000, 0.000] | 0.912 [0.869, 0.940] | 0.066 / 0.089 |

Paired differences on capped Recall@50 (B or D minus A):

| contrast | mean diff | 95% CI | CI entirely > 0 |
|---|---|---|---|
| B_leaf_minus_A | 0.042 | [0.000, 0.125] | False |
| B_pillar_minus_A | 0.000 | [0.000, 0.000] | False |
| D_matched_minus_A | 0.000 | [0.000, 0.000] | False |

Figure: `results/figures/followup_recall50_vs_selectivity.png`.

### 4.2 Experiment 2 — prompt tokens at equal recall

Tokenizer `o200k_base`; +8 tokens per chunk. Primary r=0.8. Token costs in G include unjudged chunks and are an upper bound.

#### Setting R

Setting R only retrieves within the annotated report.

| config | reachable r=0.5 | reachable r=0.8 | reachable r=1.0 | tokens@0.8 (reachable only) |
|---|---|---|---|---|
| A | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 12803.260 [10606.148, 14806.285] |
| B_leaf | 0.943 [0.897, 0.984] | 0.837 [0.767, 0.915] | 0.772 [0.669, 0.882] | 9654.699 [7459.866, 12269.024] |
| B_pillar | 0.943 [0.871, 1.000] | 0.854 [0.696, 0.976] | 0.797 [0.633, 0.941] | 11589.743 [9140.128, 13787.738] |
| D_matched | 0.967 [0.943, 0.992] | 0.894 [0.844, 0.944] | 0.837 [0.762, 0.918] | 10602.345 [8273.701, 12997.444] |

Paired token comparisons (both configs reach r):

| contrast | r | n_paired | tokens A | tokens B | relative saving | ≥15% and CI low>0 |
|---|---:|---:|---|---|---|---|
| B_leaf − A | 0.5 | 116 | 5437.698 [4479.047, 6814.933] | 5148.043 [4287.193, 6258.051] | 0.021 [-0.000, 0.045] | False |
| B_leaf − A | 0.8 | 103 | 10508.709 [8167.448, 13203.668] | 9654.699 [7459.866, 12269.024] | 0.050 [0.029, 0.072] | False |
| B_leaf − A | 1.0 | 95 | 12573.221 [9796.214, 15853.101] | 11251.747 [8733.847, 14393.840] | 0.061 [0.033, 0.093] | False |
| B_pillar − A | 0.5 | 116 | 5503.491 [4438.904, 6963.613] | 5675.328 [4605.902, 7089.822] | -0.043 [-0.183, 0.024] | False |
| B_pillar − A | 0.8 | 105 | 12232.438 [9604.879, 14556.252] | 11589.743 [9140.128, 13787.738] | 0.021 [-0.008, 0.041] | False |
| B_pillar − A | 1.0 | 98 | 13912.714 [10130.719, 17982.862] | 13148.122 [9582.067, 17191.071] | 0.036 [0.020, 0.052] | False |
| D_matched − A | 0.5 | 119 | 5522.849 [4529.732, 6947.671] | 5346.454 [4387.728, 6651.621] | 0.013 [0.002, 0.026] | False |
| D_matched − A | 0.8 | 110 | 10820.200 [8472.353, 13320.409] | 10602.345 [8273.701, 12997.444] | 0.011 [-0.009, 0.032] | False |
| D_matched − A | 1.0 | 103 | 12635.592 [9760.169, 15626.403] | 12006.816 [9246.752, 14943.828] | 0.035 [0.020, 0.052] | False |

Recall at fixed token budgets:

| config | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| A | 0.138 [0.086, 0.195] | 0.240 [0.174, 0.311] | 0.392 [0.316, 0.470] | 0.535 [0.456, 0.620] | 0.687 [0.620, 0.751] |
| B_leaf | 0.138 [0.086, 0.195] | 0.247 [0.183, 0.312] | 0.383 [0.307, 0.460] | 0.537 [0.469, 0.615] | 0.670 [0.605, 0.741] |
| B_pillar | 0.135 [0.084, 0.193] | 0.234 [0.167, 0.305] | 0.377 [0.298, 0.457] | 0.501 [0.422, 0.584] | 0.642 [0.567, 0.718] |
| D_matched | 0.138 [0.086, 0.195] | 0.244 [0.178, 0.311] | 0.391 [0.315, 0.469] | 0.531 [0.451, 0.617] | 0.686 [0.619, 0.753] |

#### Setting G

In setting G, tokens on unjudged chunks count as cost and may include unlabeled relevant text, so every config's token cost is an upper bound.

| config | reachable r=0.5 | reachable r=0.8 | reachable r=1.0 | tokens@0.8 (reachable only) |
|---|---|---|---|---|
| A | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 537764.500 [399702.923, 682571.397] |
| B_leaf | 0.750 [0.500, 0.938] | 0.375 [0.125, 0.625] | 0.188 [0.000, 0.375] | 374005.333 [237757.333, 524411.667] |
| B_pillar | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.625 [0.375, 0.875] | 523155.438 [393434.897, 656843.223] |
| D_matched | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 [0.250, 0.750] | 531265.812 [394713.814, 671260.672] |

Paired token comparisons (both configs reach r):

| contrast | r | n_paired | tokens A | tokens B | relative saving | ≥15% and CI low>0 |
|---|---:|---:|---|---|---|---|
| B_leaf − A | 0.5 | 12 | 183527.500 [132993.967, 238603.754] | 183090.500 [118420.242, 259146.596] | 0.050 [-0.143, 0.193] | False |
| B_leaf − A | 0.8 | 6 | 452443.833 [273307.500, 655083.000] | 374005.333 [237757.333, 524411.667] | 0.138 [0.073, 0.207] | False |
| B_leaf − A | 1.0 | 3 | 720181.333 [486816.000, 1104394.000] | 530343.333 [436280.000, 692205.000] | 0.222 [0.104, 0.373] | True |
| B_pillar − A | 0.5 | 16 | 226984.000 [154060.650, 313898.684] | 223880.500 [153402.180, 307490.420] | 0.009 [0.004, 0.015] | False |
| B_pillar − A | 0.8 | 16 | 537764.500 [399702.923, 682571.397] | 523155.438 [393434.897, 656843.223] | 0.020 [-0.008, 0.043] | False |
| B_pillar − A | 1.0 | 10 | 953006.300 [615783.645, 1283244.087] | 818694.900 [543633.400, 1071597.902] | 0.102 [0.060, 0.149] | False |
| D_matched − A | 0.5 | 16 | 226984.000 [154060.650, 313898.684] | 225222.750 [153598.673, 309992.678] | 0.004 [-0.000, 0.009] | False |
| D_matched − A | 0.8 | 16 | 537764.500 [399702.923, 682571.397] | 531265.812 [394713.814, 671260.672] | 0.011 [-0.002, 0.022] | False |
| D_matched − A | 1.0 | 8 | 788862.000 [450089.578, 1111778.878] | 736119.000 [432854.909, 1030898.500] | 0.046 [0.020, 0.076] | False |

Recall at fixed token budgets:

| config | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| A | 0.008 [0.000, 0.023] | 0.015 [0.003, 0.032] | 0.035 [0.011, 0.068] | 0.053 [0.026, 0.086] | 0.079 [0.044, 0.118] |
| B_leaf | 0.008 [0.000, 0.023] | 0.015 [0.003, 0.032] | 0.035 [0.011, 0.068] | 0.055 [0.028, 0.089] | 0.076 [0.042, 0.115] |
| B_pillar | 0.008 [0.000, 0.023] | 0.015 [0.003, 0.032] | 0.035 [0.011, 0.068] | 0.053 [0.026, 0.086] | 0.079 [0.044, 0.118] |
| D_matched | 0.008 [0.000, 0.023] | 0.015 [0.003, 0.032] | 0.035 [0.011, 0.068] | 0.053 [0.026, 0.086] | 0.079 [0.044, 0.118] |

Figure: `results/figures/followup_recall_vs_tokens.png`.

### 4.3 Experiment 3 — where pruned gold ranks under A

| setting | best B | n gold | n pruned | pruned median A-rank (pct) | kept median A-rank (pct) | pruned in A top-10 | share of pruned in A top-10 | MW p |
|---|---|---:|---:|---|---|---:|---|---|
| R | B_leaf | 517 | 57 | 95.000 (0.517) | 15.500 (0.083) | 2 | 0.035 | 0.0000 |
| G | B_pillar | 517 | 14 | 4035.500 (0.477) | 828.000 (0.098) | 0 | 0.000 | 0.0000 |

Setting R pruned gold by relevance: rel=2 → 36, rel=3 → 21, other/missing → 0.
Setting G pruned gold by relevance: rel=2 → 4, rel=3 → 10, other/missing → 0.

Gold chunks that A ranked in its top-10 and B pruned are listed in `results/pruned_top10_gold.csv` (2 rows).

Figure: `results/figures/followup_prank_percentile.png`.

## 5. Insights

### Pre-registered rules

- **B beats A on retrieval (G)**: frozen-best B=B_pillar is NOT MET (B_leaf MET=False, B_pillar MET=False; frozen-best Δ capped R@50 = 0.000 [0.000, 0.000]).
- **B beats A on retrieval (R)**: frozen-best B=B_leaf is NOT MET (B_leaf MET=False, B_pillar MET=False; frozen-best Δ capped R@50 = -0.028 [-0.056, -0.001]).
- **B saves meaningful tokens (R, r=0.8)**: frozen-best B=B_leaf is NOT MET (relative saving 0.050 [0.029, 0.072], n_paired=103).
- **B saves meaningful tokens (G, r=0.8)**: frozen-best B=B_pillar is NOT MET (relative saving 0.020 [-0.008, 0.043], n_paired=16).
- **Pruning errors fall on hard evidence (R)**: MET (MW p=0.0000, share of pruned gold in A top-10=0.035).
- **Pruning errors fall on hard evidence (G)**: MET (MW p=0.0000, share of pruned gold in A top-10=0.000).

### Numbered insights

1. There is no evidence that B beating A on the primary retrieval metric in setting G: paired Δ capped Recall@50 = 0.000 [0.000, 0.000] (experiment_1.pooled_heldout.G.paired_diffs).
2. In the per-report setting the same comparison is no evidence that a retrieval win: Δ capped Recall@50 = -0.028 [-0.056, -0.001].
3. At r=0.8 in setting R there is consistent with a token saving for frozen-best B: relative saving 0.050 [0.029, 0.072] on n_paired=103 (experiment_2.settings.R.paired_vs_A).
4. At r=0.8 in setting G there is no evidence of a token saving for frozen-best B: relative saving 0.020 [-0.008, 0.043] on n_paired=16. Those G token totals are an upper bound because unjudged chunks count as cost.
5. D-matched versus B-pillar tests tree structure against a matched flat budget: in G, D-matched capped Recall@50 was 0.103 [0.061, 0.147] and B-pillar was 0.103 [0.061, 0.147] at selectivities 0.749 [0.708, 0.781] and 0.693 [0.646, 0.724].
6. Setting G pruned 14 gold chunks, 0 of which A had ranked in its top-10 (share 0.000); the corresponding R counts are 57 pruned and 2 in A's top-10. Inspect `results/pruned_top10_gold.csv`.

## 6. Verdict

NEGATIVE: no retrieval or token benefit at this scale

Separately, “pruning errors fall on hard evidence” is MET in setting R and MET in setting G.

## 7. What this does not show

- Setting G leaves most of the 8,464 chunks unjudged for any given question; recall is valid, but we do not report precision, and token costs are an upper bound.
- Only 16 core questions, all climate-related; extra questions are secondary and tiny (one report each).
- Chunk-to-topic labels are automatic cosine assignments, not human-verified (config C / E_human not in this report).
- Pilot synthetic timing at N=1e5/1e6 is not re-used here as evidence for retrieval quality.
- E_human: data/question_topics.csv was not present; run_e_human.py was not applied. Not used in rule evaluation.
- FAISS multi-thread crashes from the pilot timing run are irrelevant to these CPU exact-search experiments.

## 8. Reproduce

```bash
cd pilot
bash run_followup.sh
```

Requires the existing processed parquet and cached embeddings from the pilot (`data/processed/`, `data/processed/embeddings/`). Does not re-download ClimRetrieve or re-embed unless those files are missing. Optional later: fill `data/question_topics.csv` and run `PYTHONPATH=src python src/run_e_human.py` (writes `results/e_human_results.json` only).

Generated at 2026-11-05T05:03:41.327276+00:00. Seed 42.
