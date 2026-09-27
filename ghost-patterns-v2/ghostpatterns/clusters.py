"""
Cluster validation.

k-means returns k clusters. Always. On your data, on shuffled data, on pure
Gaussian noise. A silhouette score of 0.55 sounds good until you learn that
featureless noise of the same shape scores 0.53.

Three independent checks here. A cluster structure should pass all three:

  1. Null-calibrated silhouette -- is separation better than noise gets?
  2. Gap statistic              -- is within-cluster dispersion lower than
                                   a reference distribution's?
  3. Stability                  -- do the same points cluster together when
                                   you re-run on a random 80% subsample?

Stability is the one most often skipped and the one that catches the most
ghosts. Real structure is reproducible; noise partitions are arbitrary, so
they move every run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .nullmodels import GaussianMatched, UniformHull
from .significance import check_replicates, empirical_pvalue


def _fit(X, k, seed=0):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def _scale(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))


# ------------------------------------------------ 1. calibrated silhouette


def silhouette_vs_null(df: pd.DataFrame, k_range=range(2, 9), n_null=30,
                       null="gaussian_matched", seed=0) -> pd.DataFrame:
    """
    Silhouette on real data vs the same on null replicates.

    The column that matters is 'excess': real silhouette minus the mean null
    silhouette. If excess is near zero, your clusters are an artefact of the
    algorithm, not a property of the data.
    """
    check_replicates(n_null, context="silhouette_vs_null")
    X = _scale(df)
    numeric = df.select_dtypes(include=[np.number])
    null_cls = {"gaussian_matched": GaussianMatched, "uniform_hull": UniformHull}[null]
    nm = null_cls(numeric)

    null_reps = [_scale(rep) for rep in nm.replicates(n_null, seed=seed)]

    rows = []
    for k in k_range:
        real = silhouette_score(X, _fit(X, k, seed))
        nulls = np.array([silhouette_score(R, _fit(R, k, seed)) for R in null_reps])
        excess = real - nulls.mean()
        p = empirical_pvalue(real, nulls)
        rows.append(
            {
                "k": k,
                "silhouette": round(float(real), 4),
                "null_mean": round(float(nulls.mean()), 4),
                "null_std": round(float(nulls.std(ddof=1)), 4),
                "excess": round(float(excess), 4),
                "p_value": round(float(p), 4),
                "beats_noise": p <= 0.05,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------ 2. gap statistic


def _dispersion(X, labels) -> float:
    """log of pooled within-cluster sum of squares."""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 1:
            total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return np.log(total + 1e-12)


def gap_statistic(df: pd.DataFrame, k_range=range(1, 11), n_null=25,
                  null="uniform_hull", seed=0) -> pd.DataFrame:
    """
    Tibshirani's gap statistic, with a selectable reference distribution.

    The original paper uses a uniform box (the default here, for
    comparability). Try null='gaussian_matched' as well -- it is a harder
    reference, and showing that your chosen k survives BOTH is a much stronger
    claim than the standard analysis makes.

    Selection rule: the smallest k where gap(k) >= gap(k+1) - s(k+1).
    """
    X = _scale(df)
    numeric = df.select_dtypes(include=[np.number])
    null_cls = {"gaussian_matched": GaussianMatched, "uniform_hull": UniformHull}[null]
    nm = null_cls(numeric)
    null_reps = [_scale(rep) for rep in nm.replicates(n_null, seed=seed)]

    rows = []
    for k in k_range:
        real_disp = _dispersion(X, _fit(X, k, seed) if k > 1 else np.zeros(len(X), int))
        null_disps = np.array(
            [
                _dispersion(R, _fit(R, k, seed) if k > 1 else np.zeros(len(R), int))
                for R in null_reps
            ]
        )
        gap = null_disps.mean() - real_disp
        sk = null_disps.std(ddof=1) * np.sqrt(1 + 1 / len(null_disps))
        rows.append({"k": k, "gap": round(float(gap), 4), "s_k": round(float(sk), 4)})

    out = pd.DataFrame(rows)
    chosen = None
    for i in range(len(out) - 1):
        if out.gap[i] >= out.gap[i + 1] - out.s_k[i + 1]:
            chosen = int(out.k[i])
            break
    out.attrs["chosen_k"] = chosen
    return out


# ------------------------------------------------ 3. stability


@dataclass
class StabilityResult:
    k: int
    mean_ari: float
    std_ari: float
    null_mean_ari: float
    p_value: float

    @property
    def stable(self) -> bool:
        return self.p_value <= 0.05 and self.mean_ari >= 0.6

    def __str__(self):
        tag = "STABLE" if self.stable else "unstable"
        return (
            f"k={self.k} {tag}: ARI={self.mean_ari:.3f}+/-{self.std_ari:.3f} "
            f"(noise gets {self.null_mean_ari:.3f}, p={self.p_value:.4f})"
        )


def stability(df: pd.DataFrame, k: int, n_runs=20, subsample=0.8,
              n_null=25, seed=0) -> StabilityResult:
    """
    Cluster repeated random subsamples; measure agreement on shared points
    with the Adjusted Rand Index.

    Compared against the same procedure on Gaussian noise, because noise is
    not perfectly unstable -- k-means on a blob has preferred cut directions
    and can score ARI around 0.3-0.5. Reporting a raw ARI without this
    baseline overstates your result.
    """
    check_replicates(n_null, context="stability")
    rng = np.random.default_rng(seed)

    def mean_pairwise_ari(X):
        n = len(X)
        size = int(subsample * n)
        runs = []
        for _ in range(n_runs):
            idx = rng.choice(n, size=size, replace=False)
            runs.append((set(idx), dict(zip(idx, _fit(X[idx], k, seed)))))
        aris = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                shared = sorted(runs[i][0] & runs[j][0])
                if len(shared) < k * 2:
                    continue
                aris.append(
                    adjusted_rand_score(
                        [runs[i][1][s] for s in shared],
                        [runs[j][1][s] for s in shared],
                    )
                )
        return np.array(aris)

    X = _scale(df)
    real = mean_pairwise_ari(X)

    nm = GaussianMatched(df.select_dtypes(include=[np.number]))
    null_means = np.array(
        [mean_pairwise_ari(_scale(rep)).mean() for rep in nm.replicates(n_null, seed=seed)]
    )

    p = empirical_pvalue(float(real.mean()), null_means)
    return StabilityResult(
        k=k,
        mean_ari=float(real.mean()),
        std_ari=float(real.std(ddof=1)),
        null_mean_ari=float(null_means.mean()),
        p_value=float(p),
    )


def full_report(df: pd.DataFrame, k_range=range(2, 9), seed=0):
    """Run all three checks and summarise which k (if any) is defensible."""
    sil = silhouette_vs_null(df, k_range=k_range, seed=seed)
    gap = gap_statistic(df, k_range=range(1, max(k_range) + 2), seed=seed)

    candidates = sil[sil.beats_noise].k.tolist()
    stab = {k: stability(df, k, seed=seed) for k in candidates} if candidates else {}

    verdict = [k for k, s in stab.items() if s.stable]
    return {
        "silhouette": sil,
        "gap": gap,
        "gap_chosen_k": gap.attrs.get("chosen_k"),
        "stability": stab,
        "defensible_k": verdict,
    }
