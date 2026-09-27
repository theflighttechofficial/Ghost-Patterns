"""
Null models: generate datasets that are structurally matched to the real data
but contain NO inter-column dependence.

This is the heart of the method. Every "pattern" found in real data must be
compared against what the same pipeline finds in these. A finding that also
appears here is a ghost.

Design principle: a null model should destroy exactly the structure you are
claiming to have found, and preserve everything else. Preserving too little
makes your real findings look falsely impressive; preserving too much makes
them look falsely weak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NullModel:
    """Base class. Subclasses implement .sample(rng) -> DataFrame."""

    name = "null"

    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        self.n, self.d = data.shape

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        raise NotImplementedError

    def replicates(self, n_replicates: int, seed: int = 0):
        """Yield n_replicates null datasets."""
        rng = np.random.default_rng(seed)
        for _ in range(n_replicates):
            yield self.sample(rng)

    def __repr__(self):
        return f"<{self.__class__.__name__} n={self.n} d={self.d}>"


class IndependentShuffle(NullModel):
    """
    Shuffle each column independently.

    Preserves: every column's exact marginal distribution, n, d, cardinality,
               missing-value counts, and any within-column quirks.
    Destroys:  all dependence between columns.

    This is the default null for association-rule and clustering work. If a
    rule survives against this, the co-occurrence it describes is real.
    """

    name = "independent_shuffle"

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        out = {}
        for col in self.data.columns:
            values = self.data[col].to_numpy(copy=True)
            out[col] = rng.permutation(values)
        return pd.DataFrame(out, columns=self.data.columns)


class MarginalResample(NullModel):
    """
    Resample each column with replacement from its own empirical marginal.

    Like IndependentShuffle but also destroys the exact multiset of values,
    which matters when a column has rare values whose exact count could itself
    drive a spurious rule. Slightly more conservative.
    """

    name = "marginal_resample"

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        out = {}
        for col in self.data.columns:
            values = self.data[col].to_numpy()
            idx = rng.integers(0, self.n, size=self.n)
            out[col] = values[idx]
        return pd.DataFrame(out, columns=self.data.columns)


class GaussianMatched(NullModel):
    """
    For numeric data: independent Gaussians matched to each column's mean and
    standard deviation. Zero correlation by construction.

    Use when you want to ask "is this cluster structure real, or is it what
    k-means always does to a featureless blob of the same scale?"
    """

    name = "gaussian_matched"

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.numeric = data.select_dtypes(include=[np.number])
        if self.numeric.shape[1] == 0:
            raise ValueError("GaussianMatched requires numeric columns")
        self.means = self.numeric.mean().to_numpy(copy=True)
        self.stds = self.numeric.std(ddof=1).to_numpy(copy=True)
        self.stds[self.stds == 0] = 1e-12

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        arr = rng.normal(
            loc=self.means, scale=self.stds, size=(self.n, self.numeric.shape[1])
        )
        return pd.DataFrame(arr, columns=self.numeric.columns)


class UniformHull(NullModel):
    """
    For numeric data: uniform over the bounding box of each column.

    This is the classic null used by the gap statistic. It is a WEAKER null
    than GaussianMatched for cluster validation, because a uniform box has no
    central concentration, so almost any real data beats it. Included so you
    can show the difference between a weak and a strong null -- a good result
    for the report.
    """

    name = "uniform_hull"

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.numeric = data.select_dtypes(include=[np.number])
        if self.numeric.shape[1] == 0:
            raise ValueError("UniformHull requires numeric columns")
        self.lo = self.numeric.min().to_numpy(copy=True)
        self.hi = self.numeric.max().to_numpy(copy=True)

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        arr = rng.uniform(
            low=self.lo, high=self.hi, size=(self.n, self.numeric.shape[1])
        )
        return pd.DataFrame(arr, columns=self.numeric.columns)


class BlockShuffle(NullModel):
    """
    For time-ordered data: shuffle contiguous blocks rather than single rows.

    Preserves: short-range temporal dependence inside each block.
    Destroys:  long-range structure and cross-series alignment.

    Use this when the data is a time series and you want to test whether a
    finding is more than local autocorrelation. Plain row shuffling would
    destroy the autocorrelation too, making trivial findings look significant.
    """

    name = "block_shuffle"

    def __init__(self, data: pd.DataFrame, block_size: int = 20):
        super().__init__(data)
        self.block_size = block_size

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        bs = self.block_size
        n_blocks = int(np.ceil(self.n / bs))
        starts = rng.integers(0, max(self.n - bs, 1), size=n_blocks)
        pieces = [self.data.iloc[s : s + bs] for s in starts]
        out = pd.concat(pieces, ignore_index=True).iloc[: self.n]
        return out.reset_index(drop=True)


class PhaseRandomized(NullModel):
    """
    For numeric time series: randomize Fourier phases, keep the amplitude
    spectrum.

    Preserves: the exact power spectrum, therefore the entire linear
               autocorrelation structure, and (approximately) the marginal.
    Destroys:  all NONLINEAR structure -- including volatility clustering.

    This is the strong null for financial returns. If your "pattern" in stock
    data survives IndependentShuffle but dies here, it was just
    autocorrelation. If it survives this too, you have found something
    nonlinear and genuinely interesting.
    """

    name = "phase_randomized"

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.numeric = data.select_dtypes(include=[np.number])
        if self.numeric.shape[1] == 0:
            raise ValueError("PhaseRandomized requires numeric columns")

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        out = {}
        for col in self.numeric.columns:
            x = self.numeric[col].to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=float(np.nanmean(x)))
            fft = np.fft.rfft(x)
            amps = np.abs(fft)
            phases = rng.uniform(0, 2 * np.pi, size=len(fft))
            phases[0] = 0.0  # keep the mean real
            if len(x) % 2 == 0:
                phases[-1] = 0.0  # Nyquist term must stay real
            surrogate = np.fft.irfft(amps * np.exp(1j * phases), n=len(x))
            out[col] = surrogate
        return pd.DataFrame(out, columns=self.numeric.columns)


class SwapRandomization(NullModel):
    """
    Curveball swap randomization for BINARY / transaction matrices.

    This is the correct null for market-basket data, and the one most projects
    get wrong.

    IndependentShuffle destroys column dependence but also destroys the ROW
    margins -- every transaction ends up with roughly the average number of
    items. Real basket data is wildly heterogeneous: some customers buy 2
    items, some buy 40. Big baskets generate co-occurrences mechanically, so a
    null without them makes ordinary rules look spectacular.

    Curveball preserves BOTH margins exactly:
      - row sums    (each transaction keeps its exact basket size)
      - column sums (each item keeps its exact overall frequency)
    and randomizes everything else. A rule that survives this null describes
    co-occurrence beyond what basket sizes and item popularity alone force.

    Reference: Strona et al. 2014, "A fast and unbiased procedure to randomize
    ecological binary matrices with fixed row and column totals."

    Input must be a 0/1 or boolean DataFrame.
    """

    name = "swap_randomization"

    def __init__(self, data: pd.DataFrame, n_swaps: int | None = None):
        super().__init__(data)
        arr = data.to_numpy()
        if not np.isin(arr, [0, 1, True, False]).all():
            raise ValueError(
                "SwapRandomization needs a binary matrix. One-hot encode first "
                "(see rules.one_hot) or use IndependentShuffle."
            )
        self.binary = arr.astype(bool)
        # Rule of thumb: ~5n trades mixes the chain well past the burn-in.
        self.n_swaps = n_swaps if n_swaps is not None else 5 * self.n

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        m = self.binary.copy()
        # Precompute item sets per row for speed.
        rows = [set(np.flatnonzero(m[i])) for i in range(self.n)]

        for _ in range(self.n_swaps):
            i, j = rng.integers(0, self.n, size=2)
            if i == j:
                continue
            a, b = rows[i], rows[j]
            only_a = a - b
            only_b = b - a
            if not only_a or not only_b:
                continue  # nothing tradeable; margins already force this pair

            # Trade an equal number from each side: margins stay exact.
            k = min(len(only_a), len(only_b))
            n_trade = int(rng.integers(1, k + 1))
            give = rng.choice(sorted(only_a), size=n_trade, replace=False)
            take = rng.choice(sorted(only_b), size=n_trade, replace=False)

            rows[i] = (a - set(give)) | set(take)
            rows[j] = (b - set(take)) | set(give)

        out = np.zeros_like(m)
        for i, items in enumerate(rows):
            if items:
                out[i, sorted(items)] = True
        return pd.DataFrame(out, columns=self.data.columns)


class ConditionalPermutation(NullModel):
    """
    Preserve all PAIRWISE dependence; destroy only higher-order structure.

    The sharpest null in the package, and the one that answers the question
    reviewers actually ask:

        "Is your three-way rule {A, B} -> C anything more than the two-way
         relationships A->C and B->C that you already knew about?"

    IndependentShuffle cannot answer that -- it kills the pairwise structure
    too, so any three-way rule built on real pairwise effects passes trivially.

    Method: shuffle the target column WITHIN strata defined by `condition_on`.
    Inside each stratum the target's marginal is preserved, so the
    target<->condition relationship survives intact, while the target's
    relationship to everything else is destroyed.

    Use it to demote "discoveries" that are just two known effects stacked.
    """

    name = "conditional_permutation"

    def __init__(self, data: pd.DataFrame, target: str, condition_on):
        super().__init__(data)
        if isinstance(condition_on, str):
            condition_on = [condition_on]
        missing = [c for c in [target, *condition_on] if c not in data.columns]
        if missing:
            raise KeyError(f"columns not in data: {missing}")
        self.target = target
        self.condition_on = list(condition_on)

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        out = self.data.copy()
        values = out[self.target].to_numpy(copy=True)
        strata = out.groupby(self.condition_on, observed=True, dropna=False).indices

        for idx in strata.values():
            if len(idx) > 1:
                values[idx] = rng.permutation(values[idx])

        out[self.target] = values
        return out


NULL_MODELS = {
    m.name: m
    for m in [
        IndependentShuffle,
        MarginalResample,
        GaussianMatched,
        UniformHull,
        BlockShuffle,
        PhaseRandomized,
        SwapRandomization,
    ]
}
# ConditionalPermutation is excluded from the factory: it needs a target and
# strata, so it has no sensible default construction.


def get_null_model(name: str, data: pd.DataFrame, **kwargs) -> NullModel:
    """Factory. get_null_model('independent_shuffle', df)"""
    if name not in NULL_MODELS:
        raise KeyError(f"Unknown null model {name!r}. Options: {sorted(NULL_MODELS)}")
    return NULL_MODELS[name](data, **kwargs)
