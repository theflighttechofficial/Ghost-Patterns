"""
Permutation tests.

The null models in nullmodels.py destroy ALL structure at once. Permutation
tests are the scalpel: shuffle one column, leave everything else intact, and
see whether the finding survives. If it does, that column was never what the
pattern was about.

This is how you go from "there is structure here" to "the structure is
between these two specific columns."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PermutationResult:
    statistic: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    n_permutations: int
    column: str | None = None

    @property
    def survives(self) -> bool:
        return self.p_value <= 0.05

    def __str__(self):
        tag = "SURVIVES" if self.survives else "dies"
        col = f" [shuffling {self.column}]" if self.column else ""
        return (
            f"{tag}{col}: stat={self.statistic:.4f} "
            f"null={self.null_mean:.4f}+/-{self.null_std:.4f} "
            f"z={self.z_score:.2f} p={self.p_value:.4f}"
        )


def permutation_test(data: pd.DataFrame, statistic_fn, column: str | None = None,
                     n_permutations: int = 500, seed: int = 0,
                     alternative: str = "greater") -> PermutationResult:
    """
    Generic permutation test.

    statistic_fn : callable(DataFrame) -> float. Any number you care about:
                   a rule's lift, a silhouette score, a correlation, the count
                   of anomalies -- anything.
    column       : which column to shuffle. None shuffles every column
                   independently (equivalent to the IndependentShuffle null).
    alternative  : 'greater', 'less', or 'two-sided'.

    The +1 in the p-value is not a rounding choice. With a finite number of
    permutations you can never observe p = 0, and reporting it would be a lie.
    """
    rng = np.random.default_rng(seed)
    observed = float(statistic_fn(data))

    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = data.copy()
        if column is None:
            for c in shuffled.columns:
                shuffled[c] = rng.permutation(shuffled[c].to_numpy())
        else:
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        null_stats[i] = float(statistic_fn(shuffled))

    if alternative == "greater":
        hits = np.sum(null_stats >= observed)
    elif alternative == "less":
        hits = np.sum(null_stats <= observed)
    elif alternative == "two-sided":
        centre = null_stats.mean()
        hits = np.sum(np.abs(null_stats - centre) >= abs(observed - centre))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")

    p = (hits + 1) / (n_permutations + 1)
    std = float(null_stats.std(ddof=1))
    mean = float(null_stats.mean())

    # A degenerate null (every permutation gave the identical value) makes the
    # z-score meaningless -- dividing by ~0 produces numbers like 2e12 that
    # look like overwhelming evidence and are pure artefact. Report inf/nan
    # honestly and let the p-value carry the inference.
    if std < 1e-12:
        z = float("inf") if observed > mean else (
            float("-inf") if observed < mean else float("nan")
        )
    else:
        z = (observed - mean) / std

    return PermutationResult(
        statistic=observed,
        null_mean=mean,
        null_std=std,
        p_value=float(p),
        z_score=float(z),
        n_permutations=n_permutations,
        column=column,
    )


def attribute_to_columns(data: pd.DataFrame, statistic_fn,
                         columns=None, n_permutations: int = 300,
                         seed: int = 0) -> pd.DataFrame:
    """
    Shuffle each column in turn and measure how much the statistic drops.

    The columns causing the biggest drop are the ones the pattern actually
    lives in. Columns causing no drop are decoration -- you can delete them
    from the analysis and lose nothing.

    This turns "my clustering found something" into "my clustering found
    something, and it is entirely driven by columns 3 and 7."
    """
    columns = list(columns or data.columns)
    baseline = float(statistic_fn(data))

    rows = []
    for j, col in enumerate(columns):
        res = permutation_test(
            data, statistic_fn, column=col,
            n_permutations=n_permutations, seed=seed + j,
        )
        rows.append(
            {
                "column": col,
                "baseline": round(baseline, 5),
                "shuffled_mean": round(res.null_mean, 5),
                "drop": round(baseline - res.null_mean, 5),
                "z_score": round(res.z_score, 3),
                "p_value": round(res.p_value, 5),
                "load_bearing": res.p_value <= 0.05,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("drop", ascending=False)
        .reset_index(drop=True)
    )
