# TOPG — Dataset Selection Protocol and Pre-specified Hypotheses

*Written 2026-10-11, before any TOPG or baseline result on the new datasets was computed.
This file is not to be edited after the screening results exist. Any later change must be appended as a dated amendment with its reason.*

## 1. Purpose

The goal is to support a **conditional claim** of the form: "TOPG can be expected to perform well on datasets with characteristics X, and not on datasets with characteristics Y."
To make that claim testable and free of cherry-picking:

- datasets are taken from a public benchmark suite chosen by others, using fixed structural rules;
- the dataset characteristics (moderators) and the hypotheses are fixed in advance;
- every dataset that passes the rules is analysed and reported, whether TOPG wins or loses on it.

## 2. Source pool

**OpenML-CC18** (72 curated classification tasks; Bischl et al., 2021). This suite is the standard benchmark for tabular classification, and its composition was decided independently of this study.

The ten datasets of the original TOPG benchmark are reported separately as the *original benchmark*. Their high-dimensional, small-sample microarray and proteomics datasets (P ≫ N) are kept as a separate *P ≫ N regime* analysis, because CC18 contains no such data.

## 3. Eligibility rules (applied mechanically, in this order)

| Rule | Criterion | Rationale |
|---|---|---|
| R1 | Number of features P ≥ 20 (excluding the target) | With fewer features, selection is nearly trivial. |
| R2 | Number of instances N ≤ 50,000 | Computational feasibility. This also removes the raw-pixel image sets (MNIST, Fashion-MNIST, CIFAR-10, Devnagari), which are not feature-selection problems. |
| R3 | Headroom: 0.60 ≤ ROC-AUC of a full-feature Random Forest < 0.99 | At the ceiling (≥ 0.99) or near chance (< 0.60), all selectors perform alike and the dataset cannot discriminate between them. |

The R3 reference model is fixed: a Random Forest with 100 trees and seed 42, evaluated by stratified 5-fold CV (seed 42) on the preprocessed data. This model is independent of every compared selector.

Preprocessing is identical to the benchmark: one-hot encoding of categorical features, median imputation, and stratified subsampling to 1,500 instances when N > 1,500 (seed 42).

## 4. Dataset characteristics (moderators), computed before any selector is run

All moderators are computed on the preprocessed data (≤ 1,500 instances).

| ID | Moderator | Definition |
|---|---|---|
| M-red | Redundancy | Mean over features of the maximum absolute Spearman correlation with any other feature. |
| M-pca | Effective dimensionality | Number of principal components explaining 95% of the variance (standardised data), divided by P. |
| M-nl | Non-linearity / interaction | ROC-AUC(full-feature RF) − ROC-AUC(full-feature L2 logistic regression), both by 5-fold CV. |
| M-head | Headroom | 1 − ROC-AUC(full-feature RF). |
| M-rel | Relevance density | Fraction of features with ANOVA F significant at FDR q < 0.05. |
| M-dim | Dimensionality | log10(P) and N/P. |
| M-imb | Imbalance | Majority / minority class ratio; number of classes. |
| M-cat | Categorical share | Fraction of original features that are categorical. |

## 5. Pre-specified hypotheses

Hypotheses H1–H3 follow from the mechanism analysis (manuscript Section 5.7). Truncation helps most when many candidates offer similar gains, that is, when the features are redundant, and the RF-OOB wrapper is useful when the relevant structure is non-linear.

- **H1 (accuracy).** TOPG's accuracy advantage over the baselines increases with non-linearity (M-nl).
- **H2 (stability).** TOPG's stability advantage over the net-gain criterion increases with redundancy (M-red, and inversely with M-pca).
- **H3 (scope).** TOPG's accuracy advantage decreases as headroom shrinks (M-head → 0) and is absent in the P ≫ N regime.

Other moderators (M-rel, M-dim, M-imb, M-cat) are analysed as **exploratory** and reported as such.

## 6. Outcomes and analysis plan

Per dataset (averaged over K targets, four classifiers and CV repeats):

- **ΔAUC**: ROC-AUC of TOPG minus the median ROC-AUC of all baselines. As a secondary outcome, we also report TOPG's Friedman rank among all methods.
- **ΔΦ**: Nogueira stability of TOPG minus that of the net-gain criterion (and, secondarily, the OOB Brier criterion).

For each hypothesis we will report:

1. the Spearman rank correlation between the moderator and the outcome across datasets, with a two-sided permutation p-value;
2. the outcome in the lower and upper halves of the moderator (median split), with a Mann–Whitney test.

The primary tests are H1 (M-nl vs ΔAUC), H2 (M-red vs ΔΦ) and H3 (M-head vs ΔAUC), with Holm correction across these three. The overall benchmark comparison (Friedman, Nemenyi, Holm-corrected Wilcoxon, bootstrap CIs; manuscript Section 4.3) is applied to the eligible CC18 datasets.

## 7. Amendments

**A1 (2026-10-11, during screening; implementation fix, no rule change).**
The first screening run failed to load OpenML `sick` because its column `TBG` is entirely missing, which broke the imputation step of the shared loader. The loader now drops all-missing columns before imputation (`topg_common.py`, step 1b). None of the other datasets contains such a column. The screening was re-run from scratch, and the results are in `topg_results/topg_dataset_screening/topg_screening_cc18.csv`.
Rule R1 uses the OpenML metadata feature count. For `climate-model-simulation-crashes` (metadata P = 20), two identifier-like columns are removed by the loader, leaving P = 18 in use. The dataset stays eligible under R1 as written.

**A2 (2026-10-11, before any selector was run on the CC18 datasets; design decision).**
The panel sizes are fixed for all datasets at **K ∈ {5, 10, 15, 20}**, keeping only the values with K ≤ P/2. The proportional rule of the original benchmark (10–40% of P) is not used, for two reasons. It would set K = 200 for madelon, which makes the wrapper selectors computationally infeasible, and it makes panel sizes incomparable across datasets. Each selector produces one ranked list of length max(K) per fold, and the first K features are evaluated.

**A3 (2026-10-12; registered BEFORE any moderator or selector result was computed on the confirmatory datasets).**

*Status of H1–H3.* In the interim CC18 analysis (3 repeats, 26 datasets, without varSelRF/VSURF), none of H1–H3 was supported after Holm correction (H1 ρ = +0.38, p = 0.059; H2 ρ = −0.29, opposite to the predicted direction; H3 ρ = −0.18). These hypotheses will be reported as not confirmed.

*New hypothesis, derived from the exploratory CC18 analysis* (relevance density M-rel vs ΔAUC ρ = −0.44, p = 0.025; vs ΔΦ ρ = −0.42, p = 0.032; both unadjusted and post hoc):

- **H4a (primary).** ΔAUC (TOPG minus the median of the baselines) decreases with relevance density M-rel. TOPG's accuracy advantage is larger when only a small fraction of the features is univariately relevant.
- **H4b (secondary).** ΔΦ (TOPG minus net gain) decreases with M-rel.

*Confirmatory dataset pool.* OpenML suite 271 ("AutoML Benchmark All Classification", 71 datasets), excluding every dataset that is in OpenML-CC18 by dataset id or by name. This leaves 40 candidates. The same rules are applied, with one change fixed now: **R2 is relaxed to N ≤ 1,000,000.** All data are subsampled to 1,500 instances anyway, so N only affects loading cost. With N ≤ 50,000, only 18 candidates would pass R1 + R2, which is too few for a confirmatory correlation test. R1 (P ≥ 20) and R3 (0.60 ≤ AUC of a full-feature RF < 0.99, same reference model) are unchanged. Moderators are computed exactly as in Section 4 by the same script.

*Design.*

- Same protocol as the CC18 benchmark: repeats with seeds 1, 2 and 3, K ∈ {5, 10, 15, 20} with K ≤ P/2, fold-wise imputation, and four classifiers.
- Methods: TOPG, the net-gain criterion and the twelve Python baselines of the interim analysis (ANOVA, mRMR, JMI, ReliefF, Elastic Net, HSIC-Lasso, Stability Selection, Boruta, SVM-RFE, RF impurity, RF permutation, RF-RFE).
- ΔAUC uses the median over these twelve baselines, exactly as in the analysis that generated H4. varSelRF and VSURF are not run on the confirmatory set, for computational reasons.

*Tests.*

- H4a: one-sided Spearman permutation test (10,000 permutations; alternative ρ < 0) of M-rel against ΔAUC, at α = 0.05.
- H4b: the same test against ΔΦ, at α = 0.05, reported as secondary.
- We also report whether TOPG's overall advantage over the baselines (Friedman rank, Holm-adjusted Wilcoxon tests, bootstrap CIs) replicates on the confirmatory set.

*Interpretation rule.* H4a is called confirmed only if the confirmatory test rejects. Otherwise the relevance-density pattern is reported as exploratory and not confirmed.

**A4 (2026-10-13, during the AMLB run; implementation fixes, no change to rules, methods or outcomes).**

1. The AMLB run aborted while loading `KDDCup09_appetency` with an out-of-memory error. The shared loader built a dense float64 one-hot matrix on all 50,000 rows (more than 70,000 dummy columns) before subsampling.
   - The loader now creates sparse float32 dummies, subsamples first, and densifies afterwards.
   - The selected rows depend only on y and the seed, so the output is unchanged.
   - Verified bit-identical (same columns, values, dtypes and labels) against the previous loader on credit-g, sick, cylinder-bands, PhishingWebsites, churn, jannis and kick, with and without numeric imputation.
   - KDDCup09_appetency now loads with 71,640 features, the same as in the screening.
2. *Transient infrastructure errors.* One selector call (RF permutation, fabert, seed 1, fold 3) failed with a joblib `PicklingError` ("could not pickle the task to send it to the workers"). This is a parallel-transport failure under memory pressure, not a property of the method.
   - Policy: a (seed, dataset, method) cell that failed with an infrastructure error (MemoryError, PicklingError or a worker crash) is re-run once in full.
   - Every error stays logged in `topg_cc18_errors.csv`.
   - An error raised by the selector's own computation is not retried; that cell stays empty, as specified in the runner.

**A5 (2026-10-14, during the AMLB run; efficiency fix, results unchanged).**
The RF-permutation baseline passed pandas DataFrames to `sklearn.inspection.permutation_importance`. joblib memory-maps NumPy arrays once but re-pickles a DataFrame for every per-feature job, so for P ≈ 70,000 (KDDCup09_appetency) a single fold had not finished after 18 h.

- The selector now passes NumPy arrays (same rows, values and seeds).
- Verified identical top-20 selections against the previous implementation on qsar-biodeg, madelon and cylinder-bands, for seeds 1 and 2.
- Results already computed (all of CC18; AMLB seed 1 for the smaller datasets) are therefore kept.
- The AMLB run was stopped during KDDCup09_appetency seed 1, fold 1 (nothing of that cell had been saved) and restarted; it resumes cell by cell.

**A6 (2026-10-15, during the AMLB run; crash fix in the JMI baseline, no change in non-crashing cases).**
JMI raised `list.remove(x): x not in list` on porto-seguro (seed 2, fold 3). When a feature already selected is constant within the training fold, its correlations are NaN, every remaining score is NaN, and no candidate was chosen.

- The fix: in exactly that situation, choose the remaining feature with the highest MI with y.
- Runs that did not crash follow the unchanged code path and are therefore unaffected.
- Treatment: this is a bug in our implementation of the baseline, not a property of the JMI method, so the affected (seed, dataset) JMI cells are re-run with the fix after the main AMLB run. The error stays logged in `topg_cc18_errors.csv`.
- The CC18 benchmark had no JMI failures.

**A7 (2026-10-20, after the AMLB main run finished; re-runs under the A4 policy, no change to rules, methods or outcomes).**
The main run left four cells empty, all with infrastructure errors (logged in `topg_results/topg_amlb/topg_cc18_errors.csv`):

- KDDCup09_appetency, seed 2: RF permutation (MemoryError) and ReliefF (MemoryError);
- jannis, seed 3: RF permutation (BrokenProcessPool);
- jasmine, seed 3: RF permutation (PicklingError).

Each was re-run once in full with unchanged code and settings (log `topg_results/topg_logs/topg_log_amlb_rerun.log`, 2026-10-19 19:12 to 2026-10-20 08:07). All four succeeded. The confirmatory set is complete: 24 datasets × 3 repeats × 14 methods.

The confirmatory analysis (`topg_analyze_amlb.py`, outputs in `topg_results/topg_amlb_analysis/`) was run once on the complete data.

**A8 (2026-10-20; registered before any result of this study was computed; additional robustness study, no change to the benchmark).**
*Hyperparameter sensitivity of the TOPG surrogate (self-review item B2).* The surrogate configuration was fixed a priori (T = 40 trees, maximum depth 5, candidate pool m = min(P, max(200, 5K)) for P > 150) and was not tuned. This study changes one setting at a time:

- **Part A, CC18:** all 26 eligible datasets, repeats 1–3. T ∈ {20, 80, 160}; maximum depth ∈ {3, 8}.
- **Part B, pool size:** the AMLB datasets with P > 150, where the pool is active (17 datasets), repeat 1 only. Pool factor ∈ {0.5, 2}, giving m = min(P, factor · max(200, 5K)).

Everything else is identical to the main benchmark: folds, K grid, imputation, four classifiers, seeds and Python parallelism. The default TOPG results are taken from the main benchmark files.

The code change is a `pool_size_factor` argument with default 1.0, which reproduces the original pool exactly.

*Outcomes.* For each variant, the per-dataset difference to default TOPG is reported for ROC-AUC, accuracy, Macro-F1 and Φ: mean, bootstrap 95% CI, wins and losses, and Wilcoxon p, Holm-adjusted over the variants of each part. For Part A, we also report:

- the variant's Friedman rank among the 15 CC18 methods when it replaces default TOPG;
- the variant's stability compared with the net-gain criterion.

*Reporting rule.* The study is descriptive. All variants are reported, whatever their results. No variant replaces the default in the main results.

*A8 completion note (2026-10-21).*

- **Correction to the text above.** Part B covers **16** AMLB datasets with P > 150, not 17. The rule "P > 150" was applied as written; the count of 17 was a counting error.
- **Execution.** Part A ran 2026-10-20 18:33 to 2026-10-21 00:28 and Part B to 02:12, with no failed cells and no re-runs.
- **Analysis.** `topg_analyze_sensitivity.py` was run once on the complete data; the outputs are in `topg_results/topg_sensitivity_analysis/`.
- **Interpretation caveat, recorded before the manuscript text was written.** The comparison "variant versus net gain" uses the net-gain criterion at the default T = 40. It is therefore not a same-T comparison.
