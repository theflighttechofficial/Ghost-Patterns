# 👻 Ghost Patterns v2

> **Telling real statistical patterns from artifacts and ghosts in data mining.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-37%2F37%20passing-brightgreen.svg)](file:///c:/Users/DELL/Downloads/ghost-patterns-v2/ghost-patterns-v2/tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📌 Executive Summary

Standard data mining algorithms cannot refrain from returning results:
- **k-means** always returns $k$ clusters, even on pure uniform noise.
- **Apriori** always produces association rules at a sufficiently low support threshold.
- **Isolation Forest** always flags its designated contamination quota, even on random isotropic Gaussians.

Consequently, **"I found patterns in this data" is not a scientific finding.** 

`Ghost Patterns v2` turns pattern discovery into a rigorously validated finding. It executes your exact mining pipeline on **structurally matched noise (null models)**, calibrates raw candidate discoveries against empirical null distributions, suppresses redundancy and target leakage, and reports its own false discovery and error rates.

---

## ✨ Key Features

- 🧬 **8 Null Models**: From `IndependentShuffle` and `GaussianMatched` to **Curveball `SwapRandomization`** (baskets) and `ConditionalPermutation` (higher-order rule validation).
- 🚨 **Pre-Mining Leakage Detection**: Detects functional dependencies, row-ordering artifacts, duplicate encodings, and arithmetic derivations *before* mining.
- 🎯 **Redundancy Filtering with Sampling Error Threshold**: Adaptive margin scaling ($z \cdot \text{lift} / \sqrt{N}$) to suppress noise-padded rules.
- 🛡️ **Replicate Guard**: Prevents inert statistical tests when replicate count $R$ is insufficient ($p_{\min} = 1/(R+1)$).
- 📊 **Self-Validating Benchmarks**: Quantifies family-wise error rates (FWER), false discovery rates (FDR), power curves, and minimum detectable effects (MDE).
- 📈 **Automated Visualizations**: Generates report-ready diagnostic figures and lift curves.

---

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/ghost-patterns-v2.git
cd ghost-patterns-v2

# Install dependencies
pip install -r ghost-patterns-v2/requirements.txt
```

### 30-Second Example

```python
import pandas as pd
from ghostpatterns import run, Config, check_leakage

# 1. Load data
df = pd.read_csv("mydata.csv")

# 2. Check for self-inflicted structure BEFORE mining
check_leakage(df)

# 3. Run null-calibrated mining pipeline
result = run(df, Config(n_replicates=100))

# 4. Inspect survivors and export manifest
print(result.summary())
print(result.survivors)
result.save("my_run_output")
```

---

## 🧪 Benchmark Results & Empirical Validation

### 1. Signal Recovery vs. Noise Rejection
*Planted signal: 2 real rules hidden among 18 pure-noise columns across 1,500 rows.*

| Mining Stage | Rule Count | Resulting Discovery Quality |
|---|:---:|---|
| **Naive Apriori** | 451 | Overwhelmed by noise artifacts |
| **After Redundancy Filtering** | 166 | Padded variants removed |
| **Ghost Baseline (Mean Null Run)** | 113 | Expected noise discoveries |
| **Survived Ghost Calibration** | **8** | **100% Signal (0 noise columns in survivors)** |

---

### 2. Statistical Significance vs. Practical Meaning
*Demonstrated on lottery draw datasets (sorted balls & engineered arithmetic features).*

- **Naive Mining (Sorted balls & derived features)**: **280 rules survive** standard statistical tests. All are real dependencies, but all are trivial artifacts of sorting or arithmetic.
- **Honest Mining (Raw draw order, no derived features)**: **0 rules survive.**
- **Takeaway**: Statistical significance confirms a dependence exists; `check_leakage` confirms whether that dependence is meaningful or self-inflicted.

---

### 3. Impact of Null Model Selection ⭐
*Synthetic market basket dataset with heterogeneous basket sizes and zero real item affinity.*

| Null Model | Ghost Rate (per run) | Max Ghost Lift | Total Spurious Discoveries |
|---|:---:|:---:|:---:|
| **`IndependentShuffle`** (destroys basket size distribution) | 2.7 | 1.24 | **292 false discoveries** ❌ |
| **`SwapRandomization`** (preserves row & column margins) | 445.8 | 8.21 | **0 false discoveries** ✅ |

> **Critical Insight**: Null model selection is a primary modeling decision, not an automatic default. Using `IndependentShuffle` on basket data destroys row margins, making co-occurrences in large baskets look extraordinary when they are purely mechanical.

---

### 4. Validator Diagnostics & Precision

| Metric | Empirical Value | Target / Specification |
|---|:---:|:---:|
| **False Positive Rate (Family-wise)** | **0.0%** | $\le 5.0\%$ |
| **Minimum Detectable Effect (80% Power)** | **0.30** | Benchmark calibrated |
| **Power at Effect 0.50** | **100%** | Complete signal capture |
| **Recall at Effect 0.35** | **100%** | Zero false negatives |
| **Ghost Suppression Rate** | **0.7 $\rightarrow$ 0.00** | Per structureless dataset |

---

## 🧱 Package Architecture

```
ghostpatterns/
├── pipeline.py      # One-call orchestration & reproducibility manifests
├── leakage.py       # Pre-mining leakage & structural artifact detection
├── nullmodels.py    # 8 structural null model generators
├── rules.py         # Apriori, sampling-error redundancy, FDR/FWER calibration
├── clusters.py      # Null-calibrated silhouette score & gap statistic
├── anomaly.py       # Isolation Forest calibration & dimensionality diagnostics
├── permutation.py   # Single-column attribution permutation framework
├── significance.py  # P-value calculation & replicate guard checks
├── benchmark.py     # Framework evaluation (FPR, FWER, Power, MDE)
└── viz.py           # Report figures & lift curve plots
```

---

## 🔬 Null Model Matrix

| Null Model | Preserves | Destroys | Ideal Use Case |
|---|---|---|---|
| **`IndependentShuffle`** | Exact column marginals | All cross-column dependence | Standard tabular features |
| **`MarginalResample`** | Marginal distributions | Exact value frequencies | Data with extreme rare values |
| **`GaussianMatched`** | Feature means & covariance | High-order interactions | Clustering validation |
| **`UniformHull`** | Bounding hyper-box | Density variations | Classic Gap Statistic |
| **`BlockShuffle`** | Short-range autocorrelation | Long-range temporal trends | Time series data |
| **`PhaseRandomized`** | Full power spectrum | Nonlinear structures | Financial returns & signal data |
| **`SwapRandomization`** | **Row AND column margins** | Specific item affinity | **Market basket / transaction data** |
| **`ConditionalPermutation`** | **All pairwise dependence** | Higher-order interactions | Validating if $\{A,B\}\rightarrow C$ beats $A\rightarrow C$ + $B\rightarrow C$ |

---

## 📐 Key Methodological Contributions

### 1. Sampling-Error Redundancy Thresholding
When rule $\{A\} \rightarrow \{C\}$ has high lift, noise-padded variants like $\{A, \text{noise}\} \rightarrow \{C\}$ frequently score higher purely due to sample-variance inflation in smaller sub-samples. We apply an adaptive noise threshold:

$$\text{Margin} = z \cdot \frac{\text{lift}_{\text{parent}}}{\sqrt{N_{\text{child}}}}$$

This reduced candidate survivor counts in synthetic benchmarks from 62 (mostly noise-padded) to 8 (100% genuine signal).

### 2. The Replicate Guard
With $R$ null replicates, the minimum attainable empirical p-value is:

$$p_{\min} = \frac{1}{R + 1}$$

If $p_{\min} > \alpha$, the test cannot reject the null hypothesis under any circumstances. `significance.check_replicates` automatically intercepts inadequate replicate counts before execution.

---

## 🚀 Running Experiments & Tests

```bash
# Core validation benchmarks (~15 min)
python ghost-patterns-v2/demo.py

# Advanced experiments (leakage, null choice, benchmarks ~10 min)
python ghost-patterns-v2/demo_advanced.py

# Apply directly to custom CSV dataset
python ghost-patterns-v2/apply_to_your_data.py --file my_dataset.csv

# Run unit test suite (37 tests, ~6s)
pytest ghost-patterns-v2/tests/ -q
```

---

## ⚠️ Known Limitations

- **Existence vs. Meaning**: Null models evaluate whether statistical structure exists, not whether it is domain-meaningful (e.g., lottery result).
- **Null Selection Dependency**: Result validity relies on choosing a null model aligned with data generative constraints.
- **Computational Budget**: High-replicate swap randomizations and permutations demand dedicated execution time on massive matrices.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
