# Ghost Patterns: Null-Calibrated Validation of Data-Mining Results

**Project report** · Version 2.0.0 · October 2026

---

## Abstract

Unsupervised data-mining methods cannot return an empty answer. Apriori
produces association rules from any dataset at a low enough support
threshold. k-means partitions any point cloud into exactly *k* clusters.
Isolation Forest flags whatever fraction of points its `contamination`
parameter asks for. A statement of the form "the method found patterns" is
therefore not, by itself, evidence that the data contains structure.

This project, *Ghost Patterns*, is a Python package and toolset that turns
mining output into a statistically defensible finding. It reruns the
**identical mining pipeline** on many *null datasets*: copies of the data
whose structure has been deliberately destroyed while their marginal
properties are kept. It then keeps only the findings that exceed what the
null produced. The package combines eight null models, a sampling-error-aware
redundancy filter, per-rule false-discovery-rate and family-wise tests, a
guard against tests that are unable to reject, pre-mining leakage detection,
column attribution by permutation, and calibrated versions of cluster and
anomaly validation. It also benchmarks its own error rates on synthetic
data with known ground truth.

On pure noise, 20 naive rules reduce to 0 validated rules. On data with two
planted effects, 451 naive rules reduce to 8 validated rules, all of which
involve only the planted columns. The planted-data and benchmark figures come
from the archived experiment runs (Sections 6.1 and 6.6), not the case-study
reruns. Two further results are as important as the positive ones. First,
leakage introduced by sorting and feature engineering passes every
statistical test (280 "significant" lottery rules), so significance is
necessary but not sufficient. Second, choosing the wrong null model
manufactured 292 false discoveries on basket data that contained no real
associations. Both failure modes are detected by components of the package.

---

## 1. Introduction

### 1.1 The problem

Exploratory mining is normally reported as a list: the top rules, the
discovered clusters, the flagged anomalies. Three properties of the standard
algorithms make such lists unreliable.

1. **Algorithms always return something.** The output count is set by
   thresholds and parameters, not by the data.
2. **Multiplicity.** With *d* categorical columns and rules of up to three
   items, the number of candidate hypotheses grows combinatorially. Some
   rules with high lift are guaranteed by chance alone.
3. **Redundancy.** If {A}→{C} is real, then {A, X}→{C} also scores well for
   almost any irrelevant item X. Mining output repeats one finding many
   times, and a multiple-comparison correction applied to that inflated
   list is distorted.

We call a pattern that the pipeline would also have produced from
structureless data a **ghost**. The *ghost rate* is the number of patterns
the same pipeline returns, on average, from such data.

### 1.2 Objectives

1. Provide a single workflow that separates real patterns from ghosts for
   association rules, clustering and anomaly detection.
2. Make the null hypothesis explicit and selectable, because different data
   types need different nulls.
3. Detect self-inflicted structure (leakage) before mining, since no null
   model can detect it.
4. Measure the validator's own false-positive rate, power and false
   discovery rate rather than assume them.
5. Package the method so that a non-specialist can apply it: a command-line
   tool, a self-contained HTML report and an interactive web application.

### 1.3 Contributions

- A modular library (`ghostpatterns`) with eight null models, each
  documented by what it preserves and what it destroys.
- A **sampling-error redundancy threshold** that removes padded rules whose
  extra conditions add only estimation noise.
- A **replicate guard** that refuses to report a negative result from a test
  that could not have rejected.
- **Target-focused mining**, which narrows the hypothesis family to rules
  that predict one chosen column, with the null filtered identically.
- An empirical study (Section 6) covering planted signal, leakage, null-model
  choice, cluster and anomaly validation, and a self-benchmark.
- Three front ends sharing one analysis engine: CLI, HTML report, web app.

---

## 2. Background and related work

**Association rules.** Apriori (Agrawal & Srikant, 1994) enumerates
itemsets above a minimum support and derives rules with confidence and lift.
The statistical significance of discovered patterns has been studied by
Webb (2007), who proposed holdout and permutation-based control of false
discoveries in rule mining.

**Randomization testing of mining results.** Gionis, Mannila, Mielikäinen
and Tsaparas (2007) proposed assessing mining results by swap
randomization: generating random binary matrices with the same row and
column margins and comparing the mining output. The Curveball algorithm
(Strona et al., 2014) samples such matrices efficiently and is the basis of
`SwapRandomization` in this package.

**Multiple comparisons.** The Benjamini–Hochberg procedure (1995) controls
the false discovery rate. Family-wise error is controlled here with the
max-statistic method: a rule is compared against the *best* rule of each
null run.

**Empirical p-values.** Phipson and Smyth (2010) show that permutation
p-values must never be zero; the (b + 1)/(m + 1) estimator used throughout
this package follows that recommendation.

**Cluster validation.** The gap statistic (Tibshirani, Walther & Hastie,
2001) compares within-cluster dispersion to a reference distribution.
Stability-based validation (e.g. Ben-Hur, Elisseeff & Guyon, 2002; Lange et
al., 2004) measures whether partitions reproduce under resampling.

**Anomaly detection and dimensionality.** Isolation Forest (Liu, Ting &
Zhou, 2008) and Local Outlier Factor (Breunig et al., 2000) are standard
detectors. Beyer et al. (1999) and Aggarwal, Hinneburg & Keim (2001)
showed that distances concentrate as dimensionality grows, so that nearest
and farthest neighbours become nearly equidistant.

**Surrogate data.** Phase-randomized surrogates (Theiler et al., 1992)
preserve the power spectrum of a time series while destroying nonlinear
structure; they are the basis of `PhaseRandomized`.

**Leakage.** Kaufman, Rosset, Perlich and Stitelman (2012) describe leakage
in data mining as information in the data that would not legitimately be
available, producing results that are true by construction.

This project combines these strands into one workflow and adds the
redundancy threshold, the replicate guard and target-focused calibration.

---

## 3. Method

The analysis runs in a fixed order. The order matters: deduplicating after
calibration would apply the multiple-comparison correction to many copies of
the same finding, and skipping the leakage check would validate the
analyst's own preprocessing.

```
load ─▶ drop constant columns ─▶ leakage check ─▶ mine ─▶ target filter
     ─▶ redundancy filter ─▶ null distribution ─▶ calibrate ─▶ attribute
     ─▶ (numeric data) clustering vs null ─▶ anomalies vs null ─▶ report
```

### 3.1 Null models

A null model must destroy exactly the structure being claimed and preserve
everything else. Preserving too little makes real findings look falsely
impressive; preserving too much makes them look falsely weak.

| Null model | Preserves | Destroys | Intended data |
|---|---|---|---|
| `IndependentShuffle` | Each column's exact marginal (values and counts) | All dependence between columns | General tabular data (default) |
| `MarginalResample` | Each column's distribution (bootstrap) | Exact value multiset | Columns with rare values |
| `GaussianMatched` | Per-column mean and standard deviation | All correlation | Cluster and anomaly validation |
| `UniformHull` | Per-column range | Density | Classic gap statistic |
| `BlockShuffle` | Short-range autocorrelation within blocks | Long-range structure | Time series |
| `PhaseRandomized` | Power spectrum (linear autocorrelation) | Nonlinear structure | Financial and signal series |
| `SwapRandomization` (Curveball) | Row sums **and** column sums exactly | Specific item affinities | Transaction / basket data |
| `ConditionalPermutation` | All pairwise dependence with the conditioning set | Higher-order structure | Testing whether {A, B}→C exceeds A→C and B→C |

### 3.2 Rule mining

Each categorical column is one-hot encoded into items `column=value`.
Columns with more than `max_cardinality` distinct values (default 15) are
excluded, because high-cardinality columns produce many low-support items
and are a major ghost source. Frequent itemsets are found with Apriori using
boolean column intersections. For each frequent itemset *I* and each split
into antecedent *A* and consequent *C*:

- support s(I) = fraction of rows containing *I*
- confidence = s(I) / s(A)
- lift = confidence / s(C)
- leverage = s(I) − s(A)·s(C)

Rules below `min_confidence` (default 0.60) or `min_lift` (default 1.20)
are discarded.

### 3.3 Target-focused mining

When the analyst names a target column *T*, only rules whose consequent is
a single item of *T*, and whose antecedent does not mention *T*, are kept.
This reduces the hypothesis family from all column combinations to the
question actually being asked ("what predicts *T*?"), which makes every
downstream correction less punishing. The same filter is applied to the
rules mined from each null replicate, so the null distribution answers the
same question as the real rules.

### 3.4 Sampling-error redundancy filter

Adding a condition to an antecedent subsamples the data, and lift estimated
on a smaller sample is noisier. Over many irrelevant paddings, the
*maximum* child lift therefore sits above the parent's lift by chance. A
fixed tolerance lets these children through. Instead, a child rule *r* with
the same consequent as a parent *s* whose antecedent is a strict subset is
kept only if

$$\text{lift}(r) > \text{lift}(s) + z \cdot \frac{\text{lift}(s)}{\sqrt{\text{count}(r)}}, \quad z = 1.96,$$

using the approximation that the relative standard error of a proportion
estimated from *m* rows is about 1/√m. The filter runs **before**
calibration, so the correction applies to distinct hypotheses only.

### 3.5 Calibration against the null

For *R* null replicates, the identical pipeline (mining, target filter) is
run on each replicate. Two distributions are collected:

- **all null lifts** L: every rule lift from every replicate;
- **max null lifts** M: the highest lift in each replicate (0 if a replicate
  produced no rules).

For each real rule with lift ℓ:

$$p = \frac{\#\{x \in L : x \ge \ell\} + 1}{|L| + 1}, \qquad p_{\text{FWER}} = \frac{\#\{x \in M : x \ge \ell\} + 1}{R + 1}.$$

The per-rule p-values are corrected with Benjamini–Hochberg to give
q-values. Verdicts:

| Verdict | Condition | Interpretation |
|---|---|---|
| **REAL (family-wise)** | p_FWER ≤ α | Beats the best rule noise produced in at least (1 − α) of runs |
| **likely real (FDR)** | q ≤ α only | Passes false-discovery-rate control; report with a caveat |
| **GHOST** | neither | Indistinguishable from what noise produces |

### 3.6 The replicate guard

With *R* replicates, the smallest attainable empirical p-value is
1/(R + 1). If 1/(R + 1) > α, the test can never reject, and it reports that
inability as "nothing found". This was found as a real bug during
development: `silhouette_vs_null` with 10 replicates reported
`beats_noise=False` for two perfectly separated Gaussian blobs. The minimum
is R = ⌈1/α⌉ − 1, which is 19 for α = 0.05 and 99 for α = 0.01. Every
module that computes an empirical p-value calls
`significance.check_replicates`, which warns (or raises, in strict mode).
`pipeline.run` refuses an inert configuration outright.

### 3.7 Leakage detection

Leakage passes every statistical test, because the dependence it creates is
genuinely present. It is detected separately, before mining, on raw data.

| Detector | Signature | Measure |
|---|---|---|
| Functional dependency | One column determines another | Uncertainty coefficient U(B\|A) = 1 − H(B\|A)/H(B) ≥ 0.999 |
| Near-determinism | Almost determines | U ≥ 0.95 |
| Duplicate encoding | Mutual determination | min(U(B\|A), U(A\|B)) ≥ 0.95 |
| Row ordering | Values sorted across a block of columns | ≥ 95% of rows monotone over a contiguous window of ≥ 3 columns |
| Arithmetic derivation | Column equals sum or difference of two others | Mean absolute error relative to std below 10⁻⁶ |
| Strong association (advisory) | High shared information | Normalised MI = I(A;B)/min(H(A),H(B)) ≥ 0.15 |

The row-ordering detector scans every contiguous window, because a single
appended column (such as a total) would otherwise break global monotonicity
and hide the sorted block. The advisory tier exists because transformation
*launders* leakage: binning weakens the signatures of the hard detectors
without weakening the spurious rules (Section 6.2).

### 3.8 Column attribution

Let *S(D)* be the number of strong rules (lift > 1.5, at most 2 items,
filtered to the target if one is set). For each column *j*, column *j* alone
is permuted 50 times and *S* recomputed. The *drop* is S(D) minus the mean
permuted value, with an empirical p-value. Columns with no significant drop
are decoration: the structure does not depend on them. For datasets with
more than 12 candidate columns, the 12 columns most used by the surviving
rules are tested, to keep cost bounded.

### 3.9 Clustering validation

Data are standardised. Three checks are made, against `GaussianMatched`
nulls unless noted.

1. **Null-calibrated silhouette.** For k = 2…8, the real silhouette is
   compared with the silhouettes of k-means on 25 null replicates. The
   reported quantity is the *excess* (real minus null mean) with an
   empirical p-value.
2. **Gap statistic.** Dispersion against a `UniformHull` reference (as in
   the original paper), selecting the smallest k with gap(k) ≥ gap(k+1) −
   s(k+1).
3. **Stability.** Mean pairwise adjusted Rand index over 15 random 80%
   subsamples, compared with the same procedure on noise. Noise is not
   perfectly unstable (k-means on a blob has preferred cut directions), so
   a raw ARI without a baseline overstates stability. A partition is
   reported stable when p ≤ 0.05 and mean ARI ≥ 0.6.

### 3.10 Anomaly validation

Detectors (Isolation Forest, LOF, Mahalanobis distance) are scored on the
real data and on 25 `GaussianMatched` null replicates. The threshold is the
null's 99th percentile score, so 1% of points exceed it under the null by
construction. The observed exceedance rate is compared with each replicate's
rate to give a p-value. This replaces the meaningless "count the flags"
approach, where the count is set by `contamination`.

Before trusting any distance-based detector, **relative contrast**
(farthest − nearest)/nearest, averaged over probe points, is reported:
above 1.0 distances are informative, 0.3–1.0 degrading, below 0.3 collapsed.

### 3.11 Plain-English explanations

Every rule is rendered as a sentence for non-specialist readers, for
example: *"When odor=f, Mushroom_quality=p holds 100% of the time (2.07x its
usual rate)."*

---

## 4. Implementation

### 4.1 Architecture

The package is about 3,500 non-blank lines of Python in 16 modules, with a
446-line test suite and about 440 lines of experiment scripts.

| Layer | Module | Responsibility |
|---|---|---|
| Statistics core | `significance.py` | Empirical p-values, BH correction, replicate guard |
| | `nullmodels.py` | Eight null models and a factory |
| | `permutation.py` | Permutation tests, single-column attribution |
| Mining and validation | `rules.py` | Encoding, Apriori, rules, target filter, redundancy filter, calibration |
| | `leakage.py` | Hard and advisory leakage detectors |
| | `clusters.py` | Calibrated silhouette, gap statistic, stability |
| | `anomaly.py` | Calibrated anomaly testing, relative contrast |
| Orchestration | `analyse.py` | End-to-end engine: `analyse()` computes, `save()` writes |
| | `pipeline.py` | One-call rule pipeline with reproducibility manifest |
| | `benchmark.py` | False-positive rate, power curve, FDR on synthetic data |
| | `datasets.py` | Built-in datasets with known ground truth |
| Presentation | `viz.py` | Matplotlib figures |
| | `report.py` | Markdown report and self-contained interactive HTML report |
| | `cli.py`, `__main__.py` | `ghostpatterns` command |
| | `app.py` | Streamlit web application |

### 4.2 Design decisions

- **One engine, three front ends.** `analyse()` returns a pure `Analysis`
  data object and touches no files; `save()` writes outputs. The CLI, the
  legacy `apply_to_your_data.py` script and the web app all call these two
  functions, so every interface produces identical numbers.
- **Explicit options.** All tunables live in one `Options` dataclass (and
  `Config` for `pipeline.run`), so a run is fully described by one object.
- **Reproducibility.** Every random operation takes a seed. `pipeline.run`
  writes a manifest with parameters, library versions, seeds and a SHA-256
  fingerprint of the input.
- **Self-contained report.** `report.html` embeds figures as base64 images
  and uses only inline JavaScript, so it opens offline and can be shared as
  one file. A test asserts that it makes no network requests.
- **Fail loudly.** Inert configurations warn or raise; unknown target
  columns raise; a failing anomaly detector is reported without ending the
  run.

### 4.3 User interfaces

**Command line.** Installed with `pip install .`:

```
ghostpatterns analyse data.csv --target <column> [--n-null 100] [--open]
ghostpatterns leakage data.csv          # exit code 2 if hard leakage found
ghostpatterns benchmark [--quick]
ghostpatterns figures
ghostpatterns app
```

Outputs go to `results/<dataset>/`: `report.html`, `report.md`, `log.txt`,
`rules_all.csv`, `rules_survivors.csv`, `leakage.csv`,
`column_attribution.csv`, cluster and anomaly CSVs, and `figures/*.png`.

**HTML report.** Summary verdict, headline number cards, a searchable,
sortable rules table with verdict filter and plain-English explanations,
column attribution bars, leakage table, cluster and anomaly sections, and
embedded figures. It follows the system light/dark preference.

**Web application.** Upload a CSV or choose a demo dataset (planted rules or
pure noise), pick the column to explain, adjust thresholds, and run with
live step-by-step progress. Results appear as a verdict banner, metric
cards, a rule funnel (naive → candidates → distinct → validated), headline
finding cards, interactive Altair charts with tooltips, and tabs for
columns, leakage, clusters and anomalies, the log, and downloads (HTML
report or ZIP of all outputs). Data stays on the local machine.

### 4.4 Testing and quality assurance

The suite contains **56 tests**, all passing at the time of writing. They
are grouped as follows.

| Group | What is checked |
|---|---|
| Null-model invariants | Each null preserves exactly what it claims (marginals, moments, spectrum, both margins for Curveball, pairwise dependence for conditional permutation) and destroys the rest |
| Calibration | BH monotonicity, p-values never zero, replicate minimums, warnings for too few replicates |
| Redundancy | Padded child removed; genuinely better child kept; margin scales with sample size |
| Leakage | Functional dependency, sorted block (also with an appended column), arithmetic derivation, clean data passes |
| Pipeline | Planted rule recovered; pure noise rejected; inert config refused; manifest reproducible |
| Clusters, p-value floor | Noise does not beat null; real blobs do; regression test for the inert-test bug |
| Anomaly | Contrast falls with dimension; noise shows no excess; planted outliers detected |
| Target rules | Filter correctness; null filtered identically; unknown target rejected |
| Benchmark, figures | Synthetic generator; strong effect recovered; every plot writes a file |
| Front ends | End-to-end CLI run writes all outputs; HTML self-contained; leakage exit codes; library API; web app runs end to end headlessly |

A GitHub Actions workflow runs the suite on Python 3.10 and 3.12.

---

## 5. Experimental setup

| Experiment | Data | Ground truth | Script |
|---|---|---|---|
| A, A2 | 1,500 rows; 2 planted rules + 18 noise columns | {weather=rain}→{delay=yes}; {promo=yes, weekend=yes}→{basket=large} | `demo.py` |
| B, C | Lottery draws, (B) sorted with derived features, (C) raw draw order | No real structure | `demo.py` |
| C2 | Gaussian noise | No clusters | `demo.py` |
| D | Gaussian noise, d = 2…200 | No anomalies | `demo.py` |
| E | Lottery, raw | Leakage from sorting and features | `demo_advanced.py` |
| F | 2,500 baskets, 18 items, heterogeneous sizes | No item affinity | `demo_advanced.py` |
| G | Synthetic benchmark | Known planted effect or none | `demo_advanced.py` |
| Case studies | Pure noise; planted; UCI Mushroom (8,124 × 23) | Nothing; 2 effects; edibility | `case_studies.py` |

Experiments A–G used 60 null replicates for rules unless stated. The case
studies used 40. Default thresholds: support 0.05, confidence 0.60, lift
1.20, maximum 3 items.

---

## 6. Results

### 6.1 Planted signal is recovered and noise is rejected (A)

| Stage | Rules |
|---|---:|
| Naive Apriori | 451 |
| After redundancy filter | 166 (285 were padded copies) |
| Ghost rate (mean rules per null run) | 113.2 (max 162) |
| REAL (family-wise) | **8** |
| likely real (FDR) | 0 |
| GHOST | 158 |

Null lift: median 1.223, 95th percentile 1.294, maximum 1.468. Both planted
rules were recovered: {weather=rain}→{delay=yes} (lift 2.70) and
{promo=yes, weekend=yes}→{basket=large} (lift 3.07). All eight survivors
involve only the planted columns. The other six restate the planted
effects (reverse direction, or a planted rule with an extra consequent).

**Column attribution (A2).** Shuffling `weather` or `delay` removed both
strong rules (drop 2.0, p = 0.024). Every noise column, and also `promo`,
`weekend` and `basket`, showed no drop. The last three are a limitation of
the attribution statistic, not of the rule test: it counts two-item rules
with lift > 1.5, and the basket effect needs three items. See Section 7.3.

### 6.2 Significant is not the same as meaningful (B, C, E)

Mining sorted lottery draws with engineered features (`has_low`,
`sum_bucket`, `spread`) gave 900 naive rules, 321 distinct, and **280
survivors** (275 family-wise, 5 FDR), against a ghost rate of only 3.6. The
top rule had lift 7.06. Every survivor is real dependence, and every one is
an artefact: sorting forces ball1 ≤ ball2 ≤ …, and the features are
arithmetic functions of the balls.

Mining the same lottery in raw draw order without derived columns returned
**0** rules.

The leakage detector (E), run on the raw data before any mining, found five
hard issues in under a second:

| Kind | Columns | Detail |
|---|---|---|
| Functional dependency | ball1 / has_low | U = 1.000 |
| Row ordering | ball1 … ball6 / total | 100% of rows sorted ascending |
| Arithmetic derivation (×3) | ball1, ball6, spread | spread = ball6 − ball1 |

The honest preparation reported 0 hard issues. **Statistical significance
confirms that a dependence exists; only the leakage check (or reading the
rules) can say whether it was self-inflicted.**

### 6.3 Null model choice changes the conclusion (F)

Basket data with heterogeneous sizes (mean 7, median 6, max 18 items) and
no genuine affinity gave 442 naive rules.

| Null model | Ghost rate | Strongest ghost lift | Survivors |
|---|---:|---:|---:|
| `IndependentShuffle` | 2.7 | 1.236 | **292 / 442** (all false) |
| `SwapRandomization` | 445.8 | 8.207 | **0 / 442** |

`IndependentShuffle` destroys row margins, so the null loses the large
baskets that generate co-occurrence mechanically, and ordinary rules look
extraordinary. Curveball keeps basket sizes and item frequencies exact.
**The null model is a modelling decision and must be reported.**

### 6.4 Clustering pure noise (C2)

| k | Silhouette | Null mean | Excess | p | Beats noise |
|---:|---:|---:|---:|---:|:---:|
| 2 | 0.0954 | 0.0935 | 0.0019 | 0.19 | no |
| 3 | 0.0902 | 0.0881 | 0.0021 | 0.14 | no |
| 4 | 0.0940 | 0.0887 | 0.0053 | 0.048 | yes |
| 5 | 0.0895 | 0.0901 | −0.0006 | 0.62 | no |
| 6 | 0.0932 | 0.0920 | 0.0012 | 0.38 | no |

Excess silhouette stays near zero. The single rejection at k = 4
(p = 0.048) is consistent with one false positive among five tests at
α = 0.05 with no correction across *k*; it is discussed in Section 7.3.
The stability check in this archived run used only 8 null replicates and
was flagged as inert by the replicate guard (Section 7.4).

### 6.5 Anomaly counts are set by a parameter (D)

On pure Gaussian noise, Isolation Forest with `contamination=0.05` flagged
exactly 25 of 500 points (5%) at every dimensionality from 2 to 200, while
relative contrast fell from 79.9 (d = 2) to 0.30 (d = 200). All 175 flags
are false by construction. The calibrated test reported no excess at d = 5
(observed 1.6% vs 1.0%, p = 0.125) or d = 50 (0.8% vs 1.0%, p = 0.81).

### 6.6 Benchmarking the validator (G, quick mode)

| Metric | Result | Notes |
|---|---|---|
| Family-wise false-positive rate | **0.0%** (0 of 10 noise datasets) | Target ≤ 5%; 10 trials is a small sample |
| Ghosts per noise dataset | 0.7 naive → 0.00 validated | |
| Power at planted effect 0.05 / 0.10 / 0.20 / 0.30 / 0.50 | 0% / 0% / 67% / 83% / 100% | 6 trials per effect |
| Minimum detectable effect (80% power) | **0.30** | |
| False discovery rate at effect 0.35 | 11.1% (8 true, 1 false) | **Inconclusive**: 9 discoveries, about 40 needed |
| Recall at effect 0.35 | 100% | |

### 6.7 Case studies

| Dataset | Ground truth | Naive rules | Ghost rate | Distinct | Validated |
|---|---|---:|---:|---:|---:|
| Pure noise (1,500 × 16) | nothing | 20 | 17.0 | 20 | **0** |
| Planted (1,500 × 15) | 2 effects | 170 | 2.5 | 14 | **8** |
| UCI Mushroom, target = edibility | strong | 11,340 (all rules) | 0.0 | 312 | **312** |

On pure noise the naive miner returned 20 rules and the null produced 17 on
average, so all 20 were rejected. On the planted data all 8 survivors
involve only planted columns.

**Mushroom (8,124 rows, 22 usable columns after dropping the constant
`veil_type`).** Restricting to rules that predict edibility reduced 11,340
mined rules to 975 candidates and 312 distinct hypotheses; all 312 passed
the family-wise test because noise produced essentially no rules at these
thresholds. Single-feature rules with 100% confidence include `odor=f`,
`gill_color=b` and `ring_type=l` (poisonous). Column attribution identified
`odor`, `gill_color`, `spore_print_color` and `stalk_color_above_ring` as
load-bearing; `stalkshape` and `population` contributed nothing. The
leakage check flagged `stalk_color_above_ring` and `stalk_color_below_ring`
as nearly determining `gill_attachment` (U = 0.974), and listed `odor` as
strongly associated with the target (NMI = 0.907). In this dataset, that is
the phenomenon, not preprocessing.

Note that the naive count for Mushroom counts every rule over all columns,
while the validated count counts only rules predicting edibility. The
comparison shows the size of the raw output, not a like-for-like
suppression rate.

---

## 7. Discussion

### 7.1 What the results show

1. **The method suppresses ghosts.** On structureless data, naive mining
   output is reduced to zero survivors, while planted effects of moderate
   size are recovered.
2. **The ghost rate is informative in both directions.** A high ghost rate
   (113 rules per noise run in A) shows how much of a naive list is
   expected by chance. A ghost rate near zero (Mushroom) shows the
   thresholds are strict enough that anything mined is meaningful.
3. **Significance is necessary, not sufficient.** Experiment B is the
   central lesson: leakage produces genuine dependence that every
   statistical test accepts. Mechanical leakage detection on raw data
   closes most of this gap.
4. **The null hypothesis is a modelling choice.** Experiment F shows that
   the wrong null can create hundreds of false discoveries. Reporting which
   null was used, and why, is part of the result.

### 7.2 Practical recommendations

- Report the ghost rate next to every mining result.
- Run the leakage check on raw data, before binning or feature engineering.
- Use a target column whenever the question has one.
- Use at least 19 replicates for α = 0.05 and 100 or more for final results.
- Choose the null by data type: `SwapRandomization` for baskets,
  `BlockShuffle` or `PhaseRandomized` for time series.
- Treat a negative result as a result, and quote the minimum detectable
  effect with it.

### 7.3 Limitations and threats to validity

- **Global lift comparison.** The per-rule p-value compares a rule's lift
  with the pooled lifts of all null rules, not with the same itemset's null
  distribution. Rules on rare items have noisier lifts and may be treated
  slightly differently from common ones. When the null produces no rules,
  per-rule p and q values are uninformative (they equal 0.5 by the +1
  correction) and only the family-wise test should be read; the tool says
  so in its log.
- **Conditioning on thresholds.** Null rules exist only above the mining
  thresholds, so the null distribution is truncated. The family-wise test
  remains valid because a replicate with no rules contributes a maximum of
  zero.
- **No correction across k** in clustering, nor across the three anomaly
  detectors. The k = 4 rejection on noise (Section 6.4) illustrates this.
- **Attribution statistic.** Counting two-item rules with lift > 1.5 misses
  effects that need three items (Section 6.1) and saturates on very
  structured data. Columns that are interchangeable with others can show no
  drop although they carry signal, and z-scores are infinite when the
  permuted statistic has zero variance.
- **Small benchmark samples.** The quick benchmark used 10 noise datasets
  and 6 trials per effect. The FDR estimate is explicitly inconclusive.
- **Archived experiment outputs.** Results in Sections 6.1–6.6 come from the
  stored outputs of `demo.py` and `demo_advanced.py`. Two of those runs used
  fewer replicates than the guard allows (stability with 8, anomaly with 15)
  and were flagged; the corresponding p-values should not be cited as
  significant. The case studies (6.7) were rerun for this report.
- **Existence, not meaning.** A null model shows that structure exists, not
  that it matters in the domain.
- **Compute cost.** Each replicate reruns the full mining. Mushroom with
  100 replicates took about five minutes on a laptop.

### 7.4 Development findings

Two defects were found and fixed during the project, and both are now
covered by regression tests.

1. **Inert tests.** Cluster and anomaly code originally lacked the replicate
   guard and reported "no structure" for obviously clustered data when
   given too few replicates. The shared `significance` module now guards
   every empirical p-value.
2. **Inert stability check in the analysis script.** The analysis template
   ran stability with 10 null replicates, below the minimum of 19, so it
   could never declare a partition stable. It now uses 20.

---

## 8. Conclusion and future work

Ghost Patterns shows that mining results can be made statistically
defensible with a small number of principled steps: an explicit null, the
identical pipeline run on that null, redundancy filtering before
correction, family-wise and FDR tests, a guard against inert tests, and a
leakage check before any of it. The experiments show both what the method
achieves (zero false survivors on noise, recovery of planted effects) and
what it cannot do alone (leakage passes every test; a wrong null invents
findings).

Possible extensions:

1. Per-itemset null distributions, to remove the global-lift approximation.
2. Multiple-comparison correction across *k* and across anomaly detectors.
3. Vectorised and parallel null replicates for large datasets.
4. A three-item attribution statistic, or attribution by information gain.
5. Automatic null-model recommendation from data type (binary matrix,
   time index, numeric).
6. Larger benchmark runs to estimate FDR with adequate precision.
7. Public deployment of the web app as a hosted demonstration.

---

## 9. Reproducing the results

```bash
pip install -e ".[app,dev]"
pytest                                   # 56 tests
python demo.py                           # Experiments A–D (~15 min)
python demo_advanced.py                  # Experiments E–G (~10 min)
python case_studies.py                   # Section 6.7 and figures/case_studies.png
ghostpatterns analyse mushrooms.csv --target Mushroom_quality
ghostpatterns app                        # interactive web application
```

All random operations are seeded, so reruns with the same parameters
reproduce the same numbers.

---

## References

- Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). On the surprising
  behavior of distance metrics in high dimensional space. *ICDT*.
- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association
  rules. *VLDB*.
- Ben-Hur, A., Elisseeff, A., & Guyon, I. (2002). A stability based method
  for discovering structure in clustered data. *Pacific Symposium on
  Biocomputing*.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
  rate: a practical and powerful approach to multiple testing. *Journal of
  the Royal Statistical Society, Series B*, 57(1).
- Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is
  "nearest neighbor" meaningful? *ICDT*.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF:
  identifying density-based local outliers. *SIGMOD*.
- Gionis, A., Mannila, H., Mielikäinen, T., & Tsaparas, P. (2007). Assessing
  data mining results via swap randomization. *ACM Transactions on
  Knowledge Discovery from Data*, 1(3).
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in
  data mining: formulation, detection, and avoidance. *ACM Transactions on
  Knowledge Discovery from Data*, 6(4).
- Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004).
  Stability-based validation of clustering solutions. *Neural Computation*,
  16(6).
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM*.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Schlimmer, J. (1987). Mushroom data set. *UCI Machine Learning
  Repository*.
- Strona, G., Nappo, D., Boccacci, F., Fattorini, S., & San-Miguel-Ayanz, J.
  (2014). A fast and unbiased procedure to randomize ecological binary
  matrices with fixed row and column totals. *Nature Communications*, 5.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
  (1992). Testing for nonlinearity in time series: the method of surrogate
  data. *Physica D*, 58.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of
  clusters in a data set via the gap statistic. *Journal of the Royal
  Statistical Society, Series B*, 63(2).
- Webb, G. I. (2007). Discovering significant patterns. *Machine Learning*,
  68(1).

---

## Appendix A. Default parameters

| Parameter | Default | Effect |
|---|---|---|
| `min_support` | 0.05 | Lower finds more rules and more ghosts |
| `min_confidence` | 0.60 | |
| `min_lift` | 1.20 | |
| `max_len` | 3 | Maximum items per rule |
| `max_cardinality` | 15 | Columns with more values are excluded |
| `n_null` | 100 (CLI), 60 (web app) | Null replicates for rules |
| `alpha` | 0.05 | Significance level |
| Redundancy `z` | 1.96 | |
| Silhouette / gap / anomaly nulls | 25 replicates | |
| Stability | 15 runs, 80% subsample, 20 null replicates | |
| Attribution | 50 permutations, at most 12 columns | |

## Appendix B. Output files

| File | Contents |
|---|---|
| `report.html` | Self-contained interactive report |
| `report.md` | Markdown report |
| `log.txt` | Full console output |
| `rules_all.csv` | rule, support, confidence, lift, p_value, q_value, fwer_p, verdict, leverage, count |
| `rules_survivors.csv` | Rules with verdict other than GHOST |
| `leakage.csv` | kind, columns, severity, detail |
| `column_attribution.csv` | column, baseline, shuffled_mean, drop, z_score, p_value, load_bearing |
| `clusters_silhouette.csv`, `clusters_gap.csv` | Clustering results |
| `anomalies.csv` | method, n_flagged, observed_rate, null_rate, p_value, real |
| `figures/` | lift_null, ghost_rate, top_rules, column_attribution, silhouette, anomaly_* |
