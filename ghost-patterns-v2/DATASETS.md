# Datasets for Ghost Patterns

## The pairing rule

**Always use at least two datasets: a negative control and a positive control.**

This is not optional padding. If you only run the famous-ghosts dataset and
report "nothing survived validation," you cannot distinguish two very different
situations:

- the method worked and there was genuinely nothing there, or
- the method is broken and would reject real findings too.

The positive control settles it. Run a dataset with well-documented
associations; if the method recovers them, your negative result on the other
dataset means something. `benchmark.full_benchmark()` does this on synthetic
data, but a real positive control is more convincing in a report.

A strong three-dataset submission:

| Role | Purpose | Suggested |
|---|---|---|
| Negative control | Famous ghosts; show they're illusory | Lottery draws |
| Positive control | Known real structure; prove sensitivity | UCI Mushroom |
| The real question | Genuinely unknown; your contribution | Air quality or railway delays |

---

## Tier 1 — Negative controls (famous ghosts)

### Lottery draw history ⭐ best starting point
Where to get it: most national lotteries publish complete draw archives.
Kerala State Lottery and other Indian state lotteries publish results; the UK
National Lottery, Powerball, and EuroMillions all have downloadable full
histories. Kaggle mirrors several.

- **Size:** 1,000–3,000 draws typically
- **Null model:** `IndependentShuffle`
- **Why it's ideal:** ground truth is known — there is no structure. Any
  survivor is a demonstrable false positive, which makes it a direct
  measurement of your method's error rate on real data.
- **The trap, which is the point:** do not sort the balls and do not engineer
  `sum`, `spread`, or `has_low` features. That's exactly the leakage
  Experiment B demonstrates. Run `check_leakage()` on the raw file first, then
  mine draw-order columns only.

### Daily stock returns ⭐ best for time-series techniques
Where: `yfinance` (Python, free, no key) for any ticker; NSE and BSE publish
Indian market history.

- **Size:** 10 years of daily data ≈ 2,500 rows per ticker
- **Null models:** `IndependentShuffle`, then `BlockShuffle`, then
  `PhaseRandomized`
- **Why it's the richest option:** it has a *genuine* nonlinear effect
  (volatility clustering — large moves follow large moves) hiding among
  hundreds of spurious "signals." The three-null ladder separates them:
  a finding surviving `IndependentShuffle` but dying under `PhaseRandomized`
  was only linear autocorrelation; surviving all three means something real.
- **Note:** always convert prices to returns first. Mining raw prices finds
  trend, which is not a pattern.

### Cricket ball-by-ball data
Where: Cricsheet publishes free ball-by-ball data for IPL, internationals, and
domestic matches in CSV and JSON.

- **Size:** IPL alone is hundreds of thousands of deliveries
- **Null model:** `IndependentShuffle`; `BlockShuffle` for within-over effects
- **Why it's good:** the "hot hand" and "form" debates are exactly this
  argument, and Indian cricket data makes the writeup engaging. Known result to
  test against: most perceived streaks in sport are consistent with
  independence.

### Earthquake catalogs
Where: the USGS earthquake catalog offers free bulk download with a query API.

- **Null model:** `BlockShuffle` or `PhaseRandomized`
- **Why:** claims of lunar or seasonal periodicity in earthquakes are a classic
  pattern-in-noise case. Aftershock clustering, by contrast, is real — another
  good separation test.

---

## Tier 2 — Positive controls (known real structure)

### UCI Mushroom ⭐ best positive control
Where: UCI Machine Learning Repository.

- **Size:** 8,124 rows, 22 categorical attributes
- **Null model:** `IndependentShuffle`
- **Why it's perfect:** all categorical (no binning needed, so no leakage risk),
  and it contains associations so strong they're nearly deterministic — odor
  alone almost perfectly predicts edibility. If your validator doesn't recover
  those, it's broken. It should also correctly flag the near-determinism in
  `check_leakage`, which is a nice demonstration that the detector fires on
  genuine strong structure too.

### UCI Adult / Census Income
Where: UCI Machine Learning Repository.

- **Size:** 48,842 rows, mixed categorical and numeric
- **Null model:** `IndependentShuffle`
- **Why:** widely studied, so the real associations (education↔income,
  hours↔income) are documented and you can check your survivors against
  published findings. Mixed types let you exercise the rules, clustering, and
  anomaly paths on one dataset.

### Online Retail II ⭐ the one that justifies SwapRandomization
Where: UCI Machine Learning Repository.

- **Size:** roughly a million transaction lines, 2009–2011
- **Null model:** **`SwapRandomization`** — this is the dataset that makes the
  case for it
- **Why it's methodologically valuable:** real basket data has wildly
  heterogeneous basket sizes. `IndependentShuffle` destroys those row margins,
  so big baskets stop generating co-occurrences and ordinary rules look
  spectacular against it. Curveball preserves basket sizes *and* item
  frequencies. **Running both nulls on this dataset and showing how many
  "significant" rules evaporate under the correct one is a genuine result** —
  and a section most course projects won't have.

---

## Tier 3 — Open questions (where the interesting work is)

### Indian air quality (CPCB / OpenAQ)
Where: CPCB publishes station data; OpenAQ aggregates globally with a free API.

- **Null models:** `BlockShuffle` for temporal; `IndependentShuffle` for
  discretized station comparisons
- **Why:** strong real structure (seasonal, diurnal, meteorological) mixed with
  heavily over-claimed correlations. Good place to show that some widely-cited
  relationships don't survive a proper null.
- **Leakage warning:** AQI is a *function* of the pollutant concentrations.
  Mining AQI alongside PM2.5 finds arithmetic. `check_leakage` will catch this —
  it's a great worked example for your report.

### Road accident records
Where: data.gov.in publishes Indian road accident statistics; the UK STATS19
dataset is a well-documented alternative with full per-incident records.

- **Null model:** `IndependentShuffle`
- **Why:** almost entirely categorical, high public interest, and full of
  plausible-but-unverified claims about conditions and severity.
- **Leakage warning:** severity codes are often partly derived from casualty
  counts. Check before mining.

### Electricity load / weather
Where: national grid operators publish demand data; Open-Meteo offers free
historical weather with no API key.

- **Null model:** `BlockShuffle` — daily and weekly cycles are enormous, and a
  plain shuffle would make every trivial finding look significant
- **Why:** an excellent demonstration of why null model choice matters. The
  same finding can be "highly significant" under the wrong null and vanish
  under the right one.

---

## Datasets to avoid

- **Titanic, Iris, Boston Housing** — tiny, over-analyzed, and too small for
  the multiple-comparison problem to appear. The project has nothing to show.
- **Anything pre-cleaned "for machine learning"** — engineered features are
  leakage, and you'll spend the project rediscovering the feature engineer's
  arithmetic.
- **Datasets with under ~500 rows** — the null distribution is too wide;
  everything comes back inconclusive.
- **Very wide data (1000+ columns) without reduction** — Apriori explodes
  combinatorially. Filter to 15–30 columns first.

## Practical sizing

| Property | Recommendation | Why |
|---|---|---|
| Rows | 1,000–50,000 | Below 1,000 the null is too wide; above 50,000 null replication gets slow |
| Columns for rules | 10–30 categorical | More explodes Apriori and inflates ghosts |
| Cardinality per column | ≤ 15 values | High cardinality is a major ghost source; `max_cardinality` drops the rest |
| Null replicates | 100+ final, 25 while developing | Below 19 the family-wise test is inert (see `significance.minimum_replicates`) |

## Before you mine anything

```python
import pandas as pd
from ghostpatterns import check_leakage, run, Config

df = pd.read_csv("yourdata.csv")

# 1. ALWAYS on the RAW file, before binning or feature engineering.
report = check_leakage(df)

# 2. Only then bin/encode, and mine.
result = run(df, Config(n_replicates=100))
print(result.summary())
print(result.survivors)
result.save("yourdata_run")
```

Transformation launders leakage: binning weakens the signatures the hard
detectors look for without weakening the spurious rules leakage produces. The
order above is not a suggestion.
