"""
Shared significance machinery.

This module exists because of a bug found in testing. The family-wise p-value
floor -- 1/(R+1) for R null replicates -- was guarded in the association-rule
code but not in the clustering or anomaly code. The result was silent and
nasty: `silhouette_vs_null` with 10 replicates reported `beats_noise=False`
for two perfectly separated Gaussian blobs, because the smallest attainable
p-value was 1/11 = 0.091, above alpha.

A test that cannot reject is not conservative. It is inert, and it reports
that inertness as a negative finding. Any module that computes an empirical
p-value must use the guard here.
"""

from __future__ import annotations

import warnings

import numpy as np


def minimum_replicates(alpha: float = 0.05) -> int:
    """
    Smallest replicate count that can produce a significant empirical p-value.

    With R replicates the smallest attainable p-value is 1/(R+1), because of
    the +1 correction (a finite null can never justify p = 0). If
    1/(R+1) > alpha, nothing can ever reach significance.

    alpha = 0.05 needs at least 19 replicates; alpha = 0.01 needs 99.
    Use 100+ for final results regardless.
    """
    return int(np.ceil(1 / alpha)) - 1


def check_replicates(n_replicates: int, alpha: float = 0.05,
                     strict: bool = False, context: str = "") -> bool:
    """Warn (or raise) when the null sample is too small to detect anything."""
    need = minimum_replicates(alpha)
    if n_replicates < need:
        where = f" in {context}" if context else ""
        msg = (
            f"n_replicates={n_replicates}{where} cannot yield p <= {alpha}: "
            f"the p-value floor is 1/{n_replicates + 1} = "
            f"{1 / (n_replicates + 1):.4f}. This test will report zero "
            f"significant results regardless of the data. Use at least "
            f"{need} replicates (100+ recommended)."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return False
    return True


def empirical_pvalue(observed: float, null_values, alternative="greater") -> float:
    """
    Empirical p-value with the +1 correction applied in both numerator and
    denominator. Never returns 0.
    """
    null = np.asarray(null_values, dtype=float)
    n = len(null)
    if n == 0:
        return 1.0
    if alternative == "greater":
        hits = np.sum(null >= observed)
    elif alternative == "less":
        hits = np.sum(null <= observed)
    elif alternative == "two-sided":
        c = null.mean()
        hits = np.sum(np.abs(null - c) >= abs(observed - c))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return float((hits + 1) / (n + 1))


def benjamini_hochberg(p_values, alpha: float = 0.05):
    """Standard BH step-up procedure. Returns (q_values, rejected_mask)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out, out <= alpha
