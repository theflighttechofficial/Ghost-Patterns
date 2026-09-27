"""
Test suite. Run with: pytest tests/ -v

The tests that matter most are the INVARIANT tests: a null model that quietly
fails to preserve what it claims to preserve invalidates every result built on
it, and does so silently.
"""

import numpy as np
import pandas as pd
import pytest

from ghostpatterns import clusters, leakage, rules
from ghostpatterns.anomaly import relative_contrast
from ghostpatterns.nullmodels import (
    ConditionalPermutation,
    GaussianMatched,
    IndependentShuffle,
    MarginalResample,
    PhaseRandomized,
    SwapRandomization,
)
from ghostpatterns.permutation import permutation_test
from ghostpatterns.pipeline import Config, run


@pytest.fixture
def categorical():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "a": rng.choice(["x", "y", "z"], n),
        "b": rng.choice(["p", "q"], n),
        "c": rng.choice(["1", "2", "3", "4"], n),
    })
    df["dep"] = np.where(df.a == "x", "yes", rng.choice(["yes", "no"], n))
    return df


@pytest.fixture
def numeric():
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(300, 5)),
                        columns=[f"f{i}" for i in range(5)])


# ------------------------------------------------- null model invariants


class TestNullInvariants:
    """Each null must preserve exactly what it claims and destroy the rest."""

    def test_independent_shuffle_preserves_marginals(self, categorical):
        nm = IndependentShuffle(categorical)
        sample = nm.sample(np.random.default_rng(0))
        for col in categorical.columns:
            pd.testing.assert_series_equal(
                categorical[col].value_counts().sort_index(),
                sample[col].value_counts().sort_index(),
                check_names=False,
            )

    def test_independent_shuffle_destroys_dependence(self, categorical):
        """The planted a->dep link must not survive."""
        real = leakage.uncertainty_coefficient(categorical.dep, categorical.a)
        nm = IndependentShuffle(categorical)
        nulls = [
            leakage.uncertainty_coefficient(s.dep, s.a)
            for s in nm.replicates(10, seed=0)
        ]
        assert real > 0.1
        assert np.mean(nulls) < real / 2

    def test_marginal_resample_preserves_value_set(self, categorical):
        nm = MarginalResample(categorical)
        sample = nm.sample(np.random.default_rng(0))
        for col in categorical.columns:
            assert set(sample[col]) <= set(categorical[col])

    def test_gaussian_matched_preserves_moments(self, numeric):
        nm = GaussianMatched(numeric)
        samples = list(nm.replicates(30, seed=0))
        means = np.mean([s.mean().to_numpy() for s in samples], axis=0)
        assert np.allclose(means, numeric.mean().to_numpy(), atol=0.1)

    def test_gaussian_matched_has_no_correlation(self, numeric):
        nm = GaussianMatched(numeric)
        s = nm.sample(np.random.default_rng(0))
        corr = s.corr().to_numpy()
        off = corr[~np.eye(len(corr), dtype=bool)]
        assert np.abs(off).max() < 0.25

    def test_phase_randomized_preserves_spectrum(self):
        """The defining property: the power spectrum must be unchanged."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(size=256))
        df = pd.DataFrame({"x": x})
        s = PhaseRandomized(df).sample(rng)
        p_real = np.abs(np.fft.rfft(x))
        p_null = np.abs(np.fft.rfft(s.x.to_numpy()))
        assert np.allclose(p_real, p_null, rtol=1e-6)

    def test_swap_randomization_preserves_both_margins(self):
        """Curveball's whole purpose: row AND column sums stay exact."""
        rng = np.random.default_rng(0)
        m = pd.DataFrame((rng.random((80, 12)) < 0.3).astype(int))
        nm = SwapRandomization(m, n_swaps=500)
        s = nm.sample(rng).astype(int)
        assert (s.sum(axis=1).to_numpy() == m.sum(axis=1).to_numpy()).all()
        assert (s.sum(axis=0).to_numpy() == m.sum(axis=0).to_numpy()).all()

    def test_swap_randomization_actually_randomizes(self):
        rng = np.random.default_rng(0)
        m = pd.DataFrame((rng.random((80, 12)) < 0.3).astype(int))
        s = SwapRandomization(m, n_swaps=2000).sample(rng).astype(int)
        assert (s.to_numpy() != m.to_numpy()).sum() > 0

    def test_swap_randomization_rejects_non_binary(self, numeric):
        with pytest.raises(ValueError, match="binary"):
            SwapRandomization(numeric)

    def test_conditional_permutation_preserves_pairwise(self, categorical):
        """a<->dep must survive; dep's link to everything else must not."""
        nm = ConditionalPermutation(categorical, target="dep", condition_on="a")
        s = nm.sample(np.random.default_rng(0))
        real_u = leakage.uncertainty_coefficient(categorical.dep, categorical.a)
        null_u = leakage.uncertainty_coefficient(s.dep, s.a)
        assert abs(real_u - null_u) < 1e-9


# ------------------------------------------------- statistics


class TestCalibration:
    def test_benjamini_hochberg_monotone(self):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
        q, rej = rules.benjamini_hochberg(p, alpha=0.05)
        assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
        assert rej[0] and not rej[-1]

    def test_benjamini_hochberg_empty(self):
        q, rej = rules.benjamini_hochberg(np.array([]))
        assert len(q) == 0 and len(rej) == 0

    def test_pvalue_never_zero(self):
        """The +1 correction: a finite null can never justify p=0."""
        r = rules.Rule(frozenset([0]), frozenset([1]), 0.1, 0.9, 99.0, 0.05, 50)
        rules.calibrate([r], np.array([1.0] * 100), np.array([1.0] * 100))
        assert r.p_value > 0 and r.fwer_p > 0

    def test_minimum_replicates(self):
        assert rules.minimum_replicates(0.05) == 19
        assert rules.minimum_replicates(0.01) == 99

    def test_too_few_replicates_warns(self):
        r = rules.Rule(frozenset([0]), frozenset([1]), 0.1, 0.9, 3.0, 0.05, 50)
        with pytest.warns(RuntimeWarning, match="p-value floor"):
            rules.calibrate([r], np.array([1.0] * 5), np.array([1.0] * 5))

    def test_permutation_pvalue_bounds(self, numeric):
        res = permutation_test(numeric, lambda d: float(d.f0.mean()),
                               n_permutations=50, seed=0)
        assert 0 < res.p_value <= 1

    def test_permutation_degenerate_null_no_fake_zscore(self):
        """A constant statistic must not produce a huge z-score."""
        df = pd.DataFrame({"a": [1, 2, 3, 4] * 10})
        res = permutation_test(df, lambda d: 5.0, n_permutations=20, seed=0)
        assert not np.isfinite(res.z_score) or res.z_score == 0


class TestRedundancy:
    def test_padded_rule_removed(self):
        """A child matching its parent's lift is the same finding."""
        parent = rules.Rule(frozenset([0]), frozenset([9]), .2, .8, 2.5, .1, 200)
        child = rules.Rule(frozenset([0, 1]), frozenset([9]), .1, .8, 2.5, .05, 100)
        kept = rules.remove_redundant([parent, child])
        assert parent in kept and child not in kept

    def test_genuinely_better_child_kept(self):
        """A child far beyond sampling error earns its place."""
        parent = rules.Rule(frozenset([0]), frozenset([9]), .2, .5, 1.5, .1, 400)
        child = rules.Rule(frozenset([0, 1]), frozenset([9]), .1, .95, 5.0, .05, 300)
        kept = rules.remove_redundant([parent, child])
        assert child in kept

    def test_tolerance_scales_with_sample_size(self):
        """Small-sample children need a bigger margin than large-sample ones."""
        parent = rules.Rule(frozenset([0]), frozenset([9]), .3, .6, 2.0, .1, 500)
        small = rules.Rule(frozenset([0, 1]), frozenset([9]), .02, .7, 2.4, .01, 25)
        large = rules.Rule(frozenset([0, 2]), frozenset([9]), .25, .7, 2.4, .08, 500)
        kept = rules.remove_redundant([parent, small, large])
        assert small not in kept       # 2.4 is within noise at n=25
        assert large in kept           # 2.4 is real at n=500


# ------------------------------------------------- leakage


class TestLeakage:
    def test_detects_functional_dependency(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.integers(0, 50, 400)})
        df["derived"] = np.where(df.x <= 10, "low", "high")
        rep = leakage.check_leakage(df, verbose=False)
        assert not rep.clean
        assert any(f.kind in ("functional_dependency", "near_determinism")
                   for f in rep.hard)

    def test_detects_row_sorting(self):
        rng = np.random.default_rng(0)
        rows = [sorted(rng.choice(np.arange(1, 50), 6, replace=False))
                for _ in range(300)]
        df = pd.DataFrame(rows, columns=[f"b{i}" for i in range(6)])
        rep = leakage.check_leakage(df, verbose=False)
        assert any(f.kind == "row_ordering" for f in rep.hard)

    def test_sorted_block_found_despite_extra_column(self):
        """An appended total must not mask the sorted block."""
        rng = np.random.default_rng(0)
        rows = [sorted(rng.choice(np.arange(1, 50), 6, replace=False))
                for _ in range(300)]
        df = pd.DataFrame(rows, columns=[f"b{i}" for i in range(6)])
        df["spread"] = df.b5 - df.b0
        rep = leakage.check_leakage(df, verbose=False)
        assert any(f.kind == "row_ordering" for f in rep.hard)

    def test_detects_arithmetic_derivation(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        df["total"] = df.a + df.b
        rep = leakage.check_leakage(df, verbose=False)
        assert any(f.kind == "arithmetic_derivation" for f in rep.hard)

    def test_clean_data_passes(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({c: rng.choice(list("abcd"), 400) for c in "wxyz"})
        assert leakage.check_leakage(df, verbose=False).clean

    def test_uncertainty_coefficient_bounds(self, categorical):
        u = leakage.uncertainty_coefficient(categorical.dep, categorical.a)
        assert 0 <= u <= 1
        self_u = leakage.uncertainty_coefficient(categorical.a, categorical.a)
        assert self_u == pytest.approx(1.0)


# ------------------------------------------------- end to end


class TestPipeline:
    def test_recovers_planted_rule(self):
        rng = np.random.default_rng(5)
        n = 1200
        df = pd.DataFrame({"src": rng.choice(["a", "b", "c"], n)})
        df["tgt"] = np.where(df.src == "a",
                             np.where(rng.random(n) < 0.85, "yes", "no"),
                             np.where(rng.random(n) < 0.15, "yes", "no"))
        for i in range(5):
            df[f"n{i}"] = rng.choice(["p", "q", "r"], n)

        res = run(df, Config(n_replicates=25, min_support=0.05,
                             min_confidence=0.5, seed=0), verbose=False)
        assert len(res.survivors) > 0
        assert res.survivors.rule.str.contains("src").any()

    def test_rejects_pure_noise(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({f"c{i}": rng.choice(list("abc"), 800)
                           for i in range(8)})
        res = run(df, Config(n_replicates=25, seed=0), verbose=False)
        assert len(res.survivors) == 0

    def test_run_rejects_too_few_replicates(self):
        """A config that cannot reject must fail loudly, not silently."""
        df = pd.DataFrame({"a": list("abcd") * 50, "b": list("xy") * 100})
        with pytest.raises(ValueError, match="inert"):
            run(df, Config(n_replicates=5), verbose=False)

    def test_config_validate_reports_problems(self):
        assert Config(n_replicates=5).validate()
        assert not Config(n_replicates=100).validate()

    def test_manifest_is_reproducible(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({f"c{i}": rng.choice(list("ab"), 300)
                           for i in range(4)})
        a = run(df, Config(n_replicates=20, seed=42), verbose=False)
        b = run(df, Config(n_replicates=20, seed=42), verbose=False)
        assert (a.manifest["dataset"]["fingerprint"]
                == b.manifest["dataset"]["fingerprint"])
        assert a.ghost_rate == b.ghost_rate


class TestClusters:
    def test_noise_does_not_beat_null(self, numeric):
        sil = clusters.silhouette_vs_null(numeric, k_range=range(2, 5),
                                          n_null=20, seed=0)
        assert (sil.excess.abs() < 0.05).all()

    def test_real_clusters_beat_null(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-6, 0.4, (120, 3)),
                       rng.normal(6, 0.4, (120, 3))])
        df = pd.DataFrame(X, columns=list("xyz"))
        sil = clusters.silhouette_vs_null(df, k_range=[2], n_null=25, seed=0)
        assert sil.excess.iloc[0] > 0.3
        assert bool(sil.beats_noise.iloc[0])


class TestPValueFloor:
    """
    Regression tests for the bug that motivated significance.py: an empirical
    p-value cannot go below 1/(R+1), so a small R makes a test inert rather
    than conservative -- and it reports that inertness as a negative result.
    """

    def test_clusters_warn_on_too_few_nulls(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-6, .4, (60, 3)), rng.normal(6, .4, (60, 3))])
        df = pd.DataFrame(X, columns=list("xyz"))
        with pytest.warns(RuntimeWarning, match="p-value floor"):
            clusters.silhouette_vs_null(df, k_range=[2], n_null=8, seed=0)

    def test_anomaly_warns_on_too_few_nulls(self, numeric):
        from ghostpatterns.anomaly import anomalies_vs_null
        with pytest.warns(RuntimeWarning, match="p-value floor"):
            anomalies_vs_null(numeric, n_null=5, seed=0)

    def test_obvious_clusters_detected_with_adequate_nulls(self):
        """The exact case that silently failed before the fix."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-6, .4, (120, 3)), rng.normal(6, .4, (120, 3))])
        df = pd.DataFrame(X, columns=list("xyz"))
        sil = clusters.silhouette_vs_null(df, k_range=[2], n_null=25, seed=0)
        assert bool(sil.beats_noise.iloc[0])


class TestAnomaly:
    def test_relative_contrast_falls_with_dimension(self):
        rng = np.random.default_rng(0)
        low = relative_contrast(rng.normal(size=(300, 2)), seed=0)
        high = relative_contrast(rng.normal(size=(300, 100)), seed=0)
        assert low > high
