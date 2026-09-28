"""
Anomaly detection, and why most of it is measuring the wrong thing.

Two separate problems get conflated:

  1. Detectors take a `contamination` parameter. Set it to 0.05 and you will
     get 5% flagged -- from real data, from noise, from anything. The output
     rate tells you nothing.

  2. In high dimensions, distances concentrate: the nearest and farthest
     points become nearly equidistant, so every point looks equally strange.
     Distance-based scores stop meaning anything, usually well before d = 50.

The fix for (1) is to compare the real score DISTRIBUTION against the null's,
not to count flags. The fix for (2) is to measure relative contrast first and
find out whether your detector can work at all in your dimensionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .nullmodels import GaussianMatched
from .significance import check_replicates, empirical_pvalue


def _scale(df):
    return StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))


def _scores(X, method="isolation_forest", seed=0):
    """Return anomaly scores where HIGHER = more anomalous."""
    if method == "isolation_forest":
        m = IsolationForest(n_estimators=200, random_state=seed).fit(X)
        return -m.score_samples(X)
    if method == "lof":
        m = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1))
        m.fit_predict(X)
        return -m.negative_outlier_factor_
    if method == "mahalanobis":
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
        inv = np.linalg.pinv(cov)
        diff = X - mu
        return np.einsum("ij,jk,ik->i", diff, inv, diff)
    raise ValueError(f"unknown method {method!r}")


@dataclass
class AnomalyResult:
    method: str
    n_flagged: int
    excess_rate: float
    null_rate: float
    p_value: float
    threshold: float
    # Raw score arrays, kept for plotting. Excluded from repr: they are long.
    scores: np.ndarray | None = field(default=None, repr=False)
    null_scores: np.ndarray | None = field(default=None, repr=False)

    @property
    def real(self) -> bool:
        return self.p_value <= 0.05

    def __str__(self):
        tag = "REAL anomalies present" if self.real else "no excess over noise"
        return (
            f"{self.method}: {tag} -- {self.n_flagged} points above the null's "
            f"99th pct (expected {self.null_rate:.1%}, observed "
            f"{self.excess_rate:.1%}, p={self.p_value:.4f})"
        )


def anomalies_vs_null(df: pd.DataFrame, method="isolation_forest",
                      n_null=30, percentile=99, seed=0) -> AnomalyResult:
    """
    The right question: does the real data contain MORE extreme points than a
    structureless dataset of the same shape and scale?

    Procedure: build the null's score distribution, take its 99th percentile
    as the threshold, and count how many real points exceed it. Under the null
    that fraction is 1% by construction, so anything well above 1% is real
    excess. The p-value comes from how often a null replicate produces an
    excess that large.
    """
    check_replicates(n_null, context="anomalies_vs_null")
    numeric = df.select_dtypes(include=[np.number])
    X = _scale(numeric)
    real_scores = _scores(X, method, seed)

    nm = GaussianMatched(numeric)
    null_reps = [_scale(rep) for rep in nm.replicates(n_null, seed=seed)]
    null_score_sets = [_scores(R, method, seed) for R in null_reps]

    threshold = float(np.percentile(np.concatenate(null_score_sets), percentile))

    real_rate = float(np.mean(real_scores > threshold))
    null_rates = np.array([float(np.mean(s > threshold)) for s in null_score_sets])
    p = empirical_pvalue(real_rate, null_rates)

    return AnomalyResult(
        method=method,
        n_flagged=int(np.sum(real_scores > threshold)),
        excess_rate=real_rate,
        null_rate=float(null_rates.mean()),
        p_value=float(p),
        threshold=threshold,
        scores=real_scores,
        null_scores=np.concatenate(null_score_sets),
    )


def relative_contrast(X: np.ndarray, n_probe=200, seed=0) -> float:
    """
    (farthest - nearest) / nearest, averaged over probe points.

    A diagnostic for whether distance-based anomaly detection is meaningful at
    all in this dimensionality. Rough reading:

        > 1.0   distances are informative
        0.3-1.0 degrading
        < 0.3   distances have collapsed; LOF and kNN-distance are noise

    Run this BEFORE trusting any distance-based detector. If contrast is low,
    reduce dimensionality or switch to a method that does not rely on a global
    metric (Isolation Forest holds up better).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    probes = rng.choice(n, size=min(n_probe, n), replace=False)
    ratios = []
    for i in probes:
        d = np.linalg.norm(X - X[i], axis=1)
        d = d[d > 0]
        if len(d) == 0:
            continue
        ratios.append((d.max() - d.min()) / d.min())
    return float(np.mean(ratios))


def dimensionality_curse_curve(n=500, dims=(2, 5, 10, 20, 50, 100, 200),
                               seed=0) -> pd.DataFrame:
    """
    Generate the headline figure for the report.

    Pure Gaussian noise -- no anomalies exist by construction -- at increasing
    dimensionality. Shows relative contrast collapsing while detectors keep
    confidently flagging their contamination quota.

    Every flagged point in this table is a ghost. That is the entire argument
    of the project in one chart.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in dims:
        X = rng.normal(size=(n, d))
        iso = IsolationForest(n_estimators=100, contamination=0.05,
                              random_state=seed).fit(X)
        flagged = int(np.sum(iso.predict(X) == -1))
        rows.append(
            {
                "dimensions": d,
                "relative_contrast": round(relative_contrast(X, seed=seed), 4),
                "points_flagged": flagged,
                "flag_rate": round(flagged / n, 4),
                "true_anomalies": 0,
                "all_ghosts": True,
            }
        )
    return pd.DataFrame(rows)
