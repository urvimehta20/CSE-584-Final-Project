# ClimRetrieve hierarchy-guided retrieval pilot

This file is generated from `results/pilot_results.json`. Every number is copied from that JSON.

## Dataset stats

| item | value |
|---|---|
| reports | 32 |
| chunks | 8464 |
| chunks/report min/median/max | 23 / 184.5 / 2243 |
| unique questions (all) | 26 |
| unique core questions | 16 |
| unique extra questions | 10 |
| queries (all) | 172 |
| queries (core) | 162 |
| queries (extra) | 10 |
| gold chunks/query (core, rel≥2) min/median/max | 0 / 2.0 / 21 |
| core queries excluded (zero gold, rel≥2) | 39 |
| extra queries excluded (zero gold, rel≥2) | 4 |
| core-dev questions | 8 |
| core-test questions | 8 |
| frozen setting (tuned on core-dev) | level=leaf, λ=3, τ=0.5935922265052795 |

## Extra-question coverage

| question | #reports | #queries | #gold chunks (rel≥2) | #zero-gold queries | dropped |
|---|---:|---:|---:|---:|---|
| Does the company demonstrate alignment of its business strategy with relevant regional- or national-level climate change adaptation plans? | 1 | 1 | 0 | 1 | True |
| Does the company report a set of metrics to measure its progress on adapting to the impacts of climate change?  | 1 | 1 | 1 | 0 | False |
| Does the company report a set of metrics to measure its progress on reducing its impact on biodiversity? | 1 | 1 | 3 | 0 | False |
| Does the company report a strategy and activities to significantly reduce the negative ecological impact of its activities? | 1 | 1 | 11 | 0 | False |
| Does the company report a strategy and activities to significantly reduce water consumption and pollution? | 1 | 1 | 27 | 0 | False |
| Does the company report policies and guidelines in place to ensure employees uphold the principle of ‘Do no significant harm’ across business activities? | 1 | 1 | 0 | 1 | True |
| Does the company report the assets and revenue streams at risk of losing value due to climate change? | 1 | 1 | 0 | 1 | True |
| Does the company report the methodology used to identify the key risks and opportunities of climate change to its business activities? | 1 | 1 | 4 | 0 | False |
| Does the company state how it will engage with upstream and downstream partners to advance adaptation to climate change across the value chain? (or …to make its value chain more resilient to the impacts of climate change?) | 1 | 1 | 16 | 0 | False |
| What are Scope 1 emissions? | 1 | 1 | 0 | 1 | True |

Dropped extra questions (zero gold chunks across all reports):
- Does the company demonstrate alignment of its business strategy with relevant regional- or national-level climate change adaptation plans?
- Does the company report policies and guidelines in place to ensure employees uphold the principle of ‘Do no significant harm’ across business activities?
- Does the company report the assets and revenue streams at risk of losing value due to climate change?
- What are Scope 1 emissions?

## Main results (PRIMARY: core-test)

Frozen λ/τ/level from core-dev. Means with 95% bootstrap CIs over queries. Do not pool with extra.

| config | Recall@5 | Recall@10 | Recall@20 | miss rate | selectivity | latency median / p95 (ms) |
|---|---|---|---|---|---|---|
| A | 0.342 [0.263, 0.434] | 0.509 [0.422, 0.602] | 0.627 [0.547, 0.708] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.018 / 0.075 |
| B | 0.343 [0.263, 0.433] | 0.503 [0.414, 0.599] | 0.636 [0.555, 0.727] | 0.173 [0.114, 0.234] | 0.455 [0.404, 0.505] | 0.024 / 0.059 |
| D | 0.343 [0.263, 0.433] | 0.503 [0.414, 0.599] | 0.636 [0.555, 0.727] | 0.173 [0.114, 0.234] | 0.455 [0.404, 0.505] | 0.023 / 0.050 |
| E | 0.350 [0.267, 0.438] | 0.509 [0.422, 0.602] | 0.656 [0.582, 0.738] | 0.000 [0.000, 0.000] | 0.862 [0.815, 0.905] | 0.026 / 0.094 |

Eligible queries in this split: 68.

## Secondary results (extra questions)

Same frozen settings. Extra questions only; not pooled with core-test.

| config | Recall@5 | Recall@10 | Recall@20 | miss rate | selectivity | latency median / p95 (ms) |
|---|---|---|---|---|---|---|
| A | 0.066 [0.000, 0.157] | 0.085 [0.000, 0.210] | 0.146 [0.042, 0.276] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.026 / 0.046 |
| B | 0.066 [0.000, 0.157] | 0.100 [0.015, 0.210] | 0.146 [0.042, 0.276] | 0.283 [0.033, 0.604] | 0.532 [0.350, 0.701] | 0.029 / 0.049 |
| D | 0.066 [0.000, 0.157] | 0.100 [0.015, 0.210] | 0.146 [0.042, 0.276] | 0.283 [0.033, 0.604] | 0.532 [0.350, 0.701] | 0.027 / 0.044 |
| E | 0.066 [0.000, 0.157] | 0.085 [0.000, 0.210] | 0.146 [0.042, 0.276] | 0.000 [0.000, 0.000] | 0.956 [0.895, 0.990] | 0.033 / 0.042 |

Eligible queries in this split: 6.

## Tuning-data results (core-dev)

These numbers were used to pick λ/τ/level. They are not a test estimate.

| config | Recall@5 | Recall@10 | Recall@20 | miss rate | selectivity | latency median / p95 (ms) |
|---|---|---|---|---|---|---|
| A | 0.198 [0.135, 0.266] | 0.369 [0.279, 0.470] | 0.535 [0.439, 0.642] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.019 / 0.158 |
| B | 0.211 [0.140, 0.284] | 0.377 [0.288, 0.479] | 0.508 [0.411, 0.609] | 0.237 [0.153, 0.338] | 0.457 [0.405, 0.507] | 0.021 / 0.109 |
| D | 0.211 [0.140, 0.284] | 0.377 [0.288, 0.479] | 0.508 [0.411, 0.609] | 0.237 [0.153, 0.338] | 0.457 [0.405, 0.507] | 0.024 / 0.132 |
| E | 0.217 [0.150, 0.286] | 0.369 [0.279, 0.470] | 0.556 [0.459, 0.659] | 0.000 [0.000, 0.000] | 0.888 [0.831, 0.935] | 0.031 / 0.136 |

Eligible queries in this split: 55.

## Miss attribution (gold missed by B)

| split | gold missed by B | routing error (in E) | chunk-labeling error (not in E) | queries with any miss |
|---|---:|---:|---:|---:|
| core_test | 82 | 82 | 0 | 28 |
| extra | 7 | 7 | 0 | 4 |
| core_dev | 46 | 46 | 0 | 24 |

For gold chunks absent from B's candidate set: in E = routing error (query did not select the gold chunk's topic node); not in E = chunk-labeling error (gold chunk was not in the oracle topic nodes).

## Timing pilot

Queries timed per setting: 1000 after 20 warm-up. Real corpus N=8464.

| threads | N | synthetic | method | efSearch | filtered | selectivity | median ms | p95 ms | status |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| single_thread | 8464 | False | exact_numpy_matmul | None | False | 1.0 | 0.0662 | 0.0784 | ok |
| single_thread | 8464 | False | exact_numpy_matmul | None | True | 0.05 | 0.0048 | 0.0051 | ok |
| single_thread | 8464 | False | exact_numpy_matmul | None | True | 0.1 | 0.0060 | 0.0064 | ok |
| single_thread | 8464 | False | exact_numpy_matmul | None | True | 0.25 | 0.0103 | 0.0110 | ok |
| single_thread | 8464 | False | exact_numpy_matmul | None | True | 0.5 | 0.0175 | 0.0182 | ok |
| single_thread | 8464 | False | faiss_hnsw | 32 | False | 1.0 | 0.0246 | 0.0319 | ok |
| single_thread | 8464 | False | faiss_hnsw | 128 | False | 1.0 | 0.0803 | 0.0976 | ok |
| single_thread | 100000 | True | exact_numpy_matmul | None | False | 1.0 | 2.1461 | 2.3281 | ok |
| single_thread | 100000 | True | exact_numpy_matmul | None | True | 0.05 | 0.0200 | 0.0207 | ok |
| single_thread | 100000 | True | exact_numpy_matmul | None | True | 0.1 | 0.1218 | 0.1369 | ok |
| single_thread | 100000 | True | exact_numpy_matmul | None | True | 0.25 | 0.5420 | 0.6270 | ok |
| single_thread | 100000 | True | exact_numpy_matmul | None | True | 0.5 | 1.0758 | 1.2427 | ok |
| single_thread | 100000 | True | faiss_hnsw | 32 | False | 1.0 | 0.0254 | 0.0442 | ok |
| single_thread | 100000 | True | faiss_hnsw | 128 | False | 1.0 | 0.1171 | 0.2010 | ok |
| single_thread | 1000000 | True | exact_numpy_matmul | None | False | 1.0 | 21.8203 | 23.0008 | ok |
| single_thread | 1000000 | True | exact_numpy_matmul | None | True | 0.05 | 1.0769 | 1.2295 | ok |
| single_thread | 1000000 | True | exact_numpy_matmul | None | True | 0.1 | 2.1472 | 2.3561 | ok |
| single_thread | 1000000 | True | exact_numpy_matmul | None | True | 0.25 | 5.4028 | 5.6389 | ok |
| single_thread | 1000000 | True | exact_numpy_matmul | None | True | 0.5 | 10.9153 | 11.2152 | ok |
| single_thread | 1000000 | True | faiss_hnsw | 32 | False | 1.0 | 0.0224 | 0.0449 | ok |
| single_thread | 1000000 | True | faiss_hnsw | 128 | False | 1.0 | 0.0809 | 0.1610 | ok |
| single_thread | None | False | topic_scoring_overhead | None | None | None | 0.0018 | 0.0019 | ok |
| all_threads | 8464 | False | exact_numpy_matmul | None | False | 1.0 | 0.0598 | 0.0693 | ok |
| all_threads | 8464 | False | exact_numpy_matmul | None | True | 0.05 | 0.0047 | 0.0049 | ok |
| all_threads | 8464 | False | exact_numpy_matmul | None | True | 0.1 | 0.0060 | 0.0064 | ok |
| all_threads | 8464 | False | exact_numpy_matmul | None | True | 0.25 | 0.0101 | 0.0106 | ok |
| all_threads | 8464 | False | exact_numpy_matmul | None | True | 0.5 | 0.0175 | 0.1878 | ok |
| all_threads | 8464 | False | faiss_hnsw | 32 | False | 1.0 | NA | NA | failed |
| all_threads | 8464 | False | faiss_hnsw | 128 | False | 1.0 | NA | NA | failed |
| all_threads | 100000 | True | exact_numpy_matmul | None | False | 1.0 | 2.1504 | 2.3384 | ok |
| all_threads | 100000 | True | exact_numpy_matmul | None | True | 0.05 | 0.0203 | 0.0212 | ok |
| all_threads | 100000 | True | exact_numpy_matmul | None | True | 0.1 | 0.1113 | 0.1315 | ok |
| all_threads | 100000 | True | exact_numpy_matmul | None | True | 0.25 | 0.5395 | 0.5982 | ok |
| all_threads | 100000 | True | exact_numpy_matmul | None | True | 0.5 | 1.0734 | 1.2294 | ok |
| all_threads | 100000 | True | faiss_hnsw | 32 | False | 1.0 | NA | NA | failed |
| all_threads | 100000 | True | faiss_hnsw | 128 | False | 1.0 | NA | NA | failed |
| all_threads | 1000000 | True | exact_numpy_matmul | None | False | 1.0 | 21.8035 | 23.3249 | ok |
| all_threads | 1000000 | True | exact_numpy_matmul | None | True | 0.05 | 1.0785 | 1.2495 | ok |
| all_threads | 1000000 | True | exact_numpy_matmul | None | True | 0.1 | 2.1461 | 2.3249 | ok |
| all_threads | 1000000 | True | exact_numpy_matmul | None | True | 0.25 | 5.3954 | 5.7029 | ok |
| all_threads | 1000000 | True | exact_numpy_matmul | None | True | 0.5 | 10.9440 | 12.0363 | ok |
| all_threads | 1000000 | True | faiss_hnsw | 32 | False | 1.0 | NA | NA | failed |
| all_threads | 1000000 | True | faiss_hnsw | 128 | False | 1.0 | NA | NA | failed |
| all_threads | None | False | topic_scoring_overhead | None | None | None | 0.0019 | 0.0020 | ok |

Per-query saving (unfiltered exact median − filtered exact median):

| threads | N | synthetic | s | unfiltered ms | filtered ms | saving ms | exceeds 1 ms |
|---|---:|---|---:|---:|---:|---:|---|
| single_thread | 8464 | False | 0.05 | 0.0662 | 0.0048 | 0.0613 | False |
| single_thread | 8464 | False | 0.1 | 0.0662 | 0.0060 | 0.0601 | False |
| single_thread | 8464 | False | 0.25 | 0.0662 | 0.0103 | 0.0558 | False |
| single_thread | 8464 | False | 0.5 | 0.0662 | 0.0175 | 0.0486 | False |
| single_thread | 100000 | True | 0.05 | 2.1461 | 0.0200 | 2.1261 | True |
| single_thread | 100000 | True | 0.1 | 2.1461 | 0.1218 | 2.0243 | True |
| single_thread | 100000 | True | 0.25 | 2.1461 | 0.5420 | 1.6040 | True |
| single_thread | 100000 | True | 0.5 | 2.1461 | 1.0758 | 1.0703 | True |
| single_thread | 1000000 | True | 0.05 | 21.8203 | 1.0769 | 20.7434 | True |
| single_thread | 1000000 | True | 0.1 | 21.8203 | 2.1472 | 19.6731 | True |
| single_thread | 1000000 | True | 0.25 | 21.8203 | 5.4028 | 16.4175 | True |
| single_thread | 1000000 | True | 0.5 | 21.8203 | 10.9153 | 10.9050 | True |
| all_threads | 8464 | False | 0.05 | 0.0598 | 0.0047 | 0.0551 | False |
| all_threads | 8464 | False | 0.1 | 0.0598 | 0.0060 | 0.0538 | False |
| all_threads | 8464 | False | 0.25 | 0.0598 | 0.0101 | 0.0497 | False |
| all_threads | 8464 | False | 0.5 | 0.0598 | 0.0175 | 0.0423 | False |
| all_threads | 100000 | True | 0.05 | 2.1504 | 0.0203 | 2.1301 | True |
| all_threads | 100000 | True | 0.1 | 2.1504 | 0.1113 | 2.0390 | True |
| all_threads | 100000 | True | 0.25 | 2.1504 | 0.5395 | 1.6109 | True |
| all_threads | 100000 | True | 0.5 | 2.1504 | 1.0734 | 1.0769 | True |
| all_threads | 1000000 | True | 0.05 | 21.8035 | 1.0785 | 20.7249 | True |
| all_threads | 1000000 | True | 0.1 | 21.8035 | 2.1461 | 19.6574 | True |
| all_threads | 1000000 | True | 0.25 | 21.8035 | 5.3954 | 16.4080 | True |
| all_threads | 1000000 | True | 0.5 | 21.8035 | 10.9440 | 10.8595 | True |

Smallest N where any filtered-vs-unfiltered exact saving exceeded 1 ms: 100000. Smallest N among completed settings where median filtered exact search was >1 ms faster than unfiltered exact.

Timing failures / skipped settings:
- Skipped FAISS HNSW for all_threads N=8464: a previous all-threads HNSW run aborted this process with a native segfault (exit 139).
- Skipped FAISS HNSW for all_threads N=100000: a previous all-threads HNSW run aborted this process with a native segfault (exit 139).
- Skipped FAISS HNSW for all_threads N=1000000: a previous all-threads HNSW run aborted this process with a native segfault (exit 139).
- Original all-threads pass crashed after real_N exact/filtered; remaining all-threads settings were resumed in a new process.

## Routing diagnostic

| question | set | top pillar | top leaf |
|---|---|---|---|
| Do the environmental/sustainability targets set by the company align with external climate change adaptation goals/targets? | core | G (Governance) | Climate adaptation |
| Do the environmental/sustainability targets set by the company reference external climate change adaptation goals/targets? | core | E (Environmental) | Climate adaptation |
| Does the company demonstrate alignment of its business strategy with relevant regional- or national-level climate change adaptation plans? | extra | E (Environmental) | Climate adaptation |
| Does the company encourage downstream partners to carry out climate-related risk assessments? | core | E (Environmental) | Risk management |
| Does the company have a specific process in place to identify risks arising from climate change? | core | E (Environmental) | Risk management |
| Does the company have a strategy on waste management? | core | E (Environmental) | Waste and circularity |
| Does the company have any engagements with industry peers in relation to climate change? | core | E (Environmental) | Risk management |
| Does the company identify any impacts of its business activities on the environment? | core | E (Environmental) | Community |
| Does the company provide definitions for climate change adaptation? | core | E (Environmental) | Climate adaptation |
| Does the company refer to any third party scenarios when identifying climate-related risks or opportunities (e.g. IPCC trajectories, NGFS scenarios, etc.)? | core | G (Governance) | Risk management |
| Does the company report a plan to engage with downstream partners on water consumption or water pollution? | core | E (Environmental) | Water |
| Does the company report a set of metrics to measure its progress on adapting to the impacts of climate change?  | extra | E (Environmental) | Climate adaptation |
| Does the company report a set of metrics to measure its progress on reducing its impact on biodiversity? | extra | E (Environmental) | Biodiversity |
| Does the company report a strategy and activities to significantly reduce the negative ecological impact of its activities? | extra | E (Environmental) | Waste and circularity |
| Does the company report a strategy and activities to significantly reduce water consumption and pollution? | extra | E (Environmental) | Water |
| Does the company report how adjustments to its business operations will allow it to adapt to climate change? | core | E (Environmental) | Climate adaptation |
| Does the company report policies and guidelines in place to ensure employees uphold the principle of ‘Do no significant harm’ across business activities? | extra | G (Governance) | Ethics |
| Does the company report short-term actions taken or planned to reduce its waste generation? | core | E (Environmental) | Waste and circularity |
| Does the company report the assets and revenue streams at risk of losing value due to climate change? | extra | E (Environmental) | Transition risk |
| Does the company report the climate change scenarios used to test the resilience of its business strategy? | core | E (Environmental) | Risk management |
| Does the company report the methodology used to identify the dependencies and impact of its business activities on the environment? | core | E (Environmental) | Risk management |
| Does the company report the methodology used to identify the key risks and opportunities of climate change to its business activities? | extra | E (Environmental) | Risk management |
| Does the company seek to adjust its business model to better provide climate change adaptation products and services? | core | E (Environmental) | Climate adaptation |
| Does the company state how it will engage with upstream and downstream partners to advance adaptation to climate change across the value chain? (or …to make its value chain more resilient to the impacts of climate change?) | extra | S (Social) | Climate adaptation |
| Has the company identified any synergies between its climate change adaptation goals and other business goals? | core | E (Environmental) | Climate adaptation |
| What are Scope 1 emissions? | extra | G (Governance) | GHG emissions |

Core questions route to 2 distinct pillar(s) ['E', 'G']; 14 of 16 share modal pillar E.
Extra questions route to 3 distinct pillar(s) ['E', 'G', 'S']; 7 of 10 share modal pillar E.

At frozen τ=0.5935922265052795, mean leaves per chunk = 2.729087901701323 (min 1, max 14).

Chunks per leaf:
- e_physical_climate_risk: 768
- e_transition_risk: 570
- e_climate_adaptation: 1160
- e_ghg_emissions: 3525
- e_energy: 1706
- e_water: 1742
- e_biodiversity: 766
- e_waste: 1957
- s_workforce: 2072
- s_health_safety: 2532
- s_community: 3211
- g_board_oversight: 1031
- g_risk_management: 1594
- g_ethics: 465

## Slide-ready findings

Quoted from `retrieval.frozen_metrics.core_test` unless noted.

- On core-test, B did not beat A on Recall@10: B 0.503 vs A 0.509 at B selectivity 0.455.
- B kept 0.827 of gold chunks in its candidate set on core-test (miss rate 0.173) at selectivity 0.455.
- Oracle E core-test Recall@10 was 0.509 with miss rate 0.000; D (flat leaves) Recall@10 was 0.503.
- Of 82 gold chunks missed by B on core-test, 82 were in E (routing error) and 0 were missing from E (chunk-labeling error).
- Filtered vs unfiltered exact search first exceeded a 1 ms median saving at N=100000 (see timing.where_saving_exceeds_1ms).
- Core questions used 2 distinct routed pillar(s); 14 of 16 routed to E.

## Caveats

- Synthetic scale-up embeddings are real vectors plus Gaussian noise, then re-normalized; those N values are labeled synthetic in the JSON.
- Primary test set is 8 core questions only (queries are those questions asked on reports with gold at relevance ≥ 2).
- All ClimRetrieve questions are climate-related, so topic routing may collapse onto a small number of pillars/leaves.
- Chunk topic labels are automatic cosine assignments and have not been human-verified (config C is out of scope).
- λ and τ were chosen on core-dev; core-dev tables are tuning data, not a held-out estimate.
- Core-test and extra numbers are never pooled.
- FAISS multi-thread HNSW crashed natively; that comparison is missing from all-threads results.

## Run metadata

- embedding model: BAAI/bge-small-en-v1.5
- seed: 42
- CPU: Apple M4 (10 cores), RAM 16.0 GB
- Python 3.12.13 on macOS-15.6.1-arm64-arm-64bit
- library versions: {'python': '3.12.13', 'pandas': '2.2.3', 'pyarrow': '18.1.0', 'numpy': '2.2.6', 'sklearn': '1.6.1', 'matplotlib': '3.10.1', 'sentence_transformers': '3.4.1', 'torch': '2.6.0', 'faiss': '1.9.0', 'threadpoolctl': '3.6.0'}

## Notes from the run

- Dropped 15 report-level rows with null/empty paragraph (raw rows=47941).
- Report-level relevance has 9 NaN values. After max-aggregating per (report, question, paragraph), pairs whose only values are NaN are omitted from qrels.
- Query set_type is tagged from ClimRetrieve_base 'Core 16 Question' (max flag per Question text). It matches Core_Questions_with_Explanations.xlsx.
- Exact-text dedup within a report collapsed 1 paragraph text(s) that had more than one original row index.
- Expert-base (Document, Question) pairs absent from report-level (1): [('2022_BXP_ESG_Report.pdf', 'Does the company provide definitions for climate change adaptation?')].
- FAISS HNSW with all threads (omp>1) aborted with a native segfault (exit 139) on this machine; HNSW latencies are single-thread only. Exact numpy timings exist for both thread modes.
