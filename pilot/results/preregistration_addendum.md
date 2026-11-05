# Pre-registration addendum

Written on 2026-11-04 **after seeing setting-G gold-set sizes only, and before any follow-up retrieval, token, or pruning-rank results**. This file does not replace `preregistration.md` and does not change the three decision-rule sentences there. It specifies which metric, which r, and which averages are used when those rules are evaluated.

The gold-set sizes that were visible at the time of writing are in `results/setting_g_question_stats.json` (core gold counts from 3 to 89; three core questions have fewer than 10 gold chunks). No Recall@k, latency, token, or A-rank number had been computed for the follow-up when this addendum was written.

`preregistration.md` is not edited.

## 1. Primary metric for "B beats A on retrieval"

**Macro-averaged capped Recall@50.**

Per question (setting G) or per query (setting R):

`capped_recall@50 = |gold ∩ top-50| / min(50, |gold|)`

Then average with each **question** weighted equally in setting G, and with each **query** weighted equally in setting R (see §3).

The paired difference is (B − A) on this metric. The rule is MET only if that paired cluster-bootstrap 95% CI lies entirely above 0.

Raw Recall@20, Recall@50, Recall@100 and capped Recall@20 / capped Recall@100 are **secondary**. They are reported in tables and are **not** used to evaluate any pre-registered rule.

## 2. Primary recall level for "B saves meaningful tokens"

**r = 0.8.**

Also report r = 0.5 and r = 1.0 (reachable fractions and paired token totals), but **do not** use them to evaluate the rule.

The rule still requires, at r = 0.8: B needs ≥15% fewer prompt tokens than A **and** the paired CI of the relative saving `(tokens_A − tokens_B) / tokens_A` excludes 0. Comparison is only on units where **both** configs reach r.

## 3. What is averaged

- **Setting G:** every per-question average is a **macro average over questions** (each question counts once, regardless of how many reports it was annotated on).
- **Setting R:** the point estimate is an average over **queries**. The confidence interval is the cluster bootstrap over **questions** already registered in `preregistration.md` (resample questions, then include all queries of those questions).

Extra questions remain secondary and are never pooled with core.

## 4. Small gold-set questions are kept

Do **not** drop questions with fewer than 10 gold chunks. Report the full core set **and** a separate slice for these three questions (gold counts from setting-G stats):

- “Does the company provide definitions for climate change adaptation?” — 4 gold chunks
- “Does the company report a plan to engage with downstream partners on water consumption or water pollution?” — 3 gold chunks
- “Does the company refer to any third party scenarios when identifying climate-related risks or opportunities (e.g. IPCC trajectories, NGFS scenarios, etc.)?” — 8 gold chunks

Capped Recall@50 is well-defined for them (`min(50, |gold|)` = |gold|).

## 5. Unjudged chunks in setting G are costly

In setting G, tokens spent on unjudged chunks **count as cost**. Some of those chunks may actually be relevant (they were never labeled for that question), so the token cost for every config is an **upper bound** on the tokens needed to reach a given recall.

## E_human

Human-labeled oracle routing is **not** part of this follow-up run. If `data/question_topics.csv` is filled later, `src/run_e_human.py` writes `results/e_human_results.json` only. Those numbers are not folded into `REPORT.md` rule evaluation.
