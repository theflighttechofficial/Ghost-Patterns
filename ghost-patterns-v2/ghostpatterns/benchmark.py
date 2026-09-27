"""
Benchmarking the validator itself.

Every ghost-hunting method claims to separate real patterns from noise. Almost
none report their own error rates. This module does, using synthetic data where
ground truth is known by construction:

  FALSE POSITIVE RATE  On datasets with NO planted structure, how often does
                       the method report at least one survivor? Should be at
                       or below alpha. Above it, the method is a ghost
                       generator itself.

  POWER                On datasets WITH a planted rule, how often is the rule
                       recovered? Low power means you will miss real findings
                       and report false negatives as "nothing here."

  FALSE DISCOVERY RATE Among all reported survivors, what fraction is not a
                       planted rule? Should sit at or below alpha after BH.

  MINIMUM DETECTABLE   How strong must an effect be, at this sample size,
  EFFECT               before the method can see it? Report this alongside any
                       negative result -- "we found nothing" is only
                       meaningful if you state what you could have found.

This section is what separates a course project from a methods contribution.
An uncalibrated method that reports 30% false positives is worse than no
method, and you cannot know which you have without measuring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import rules
from .nullmodels import IndependentShuffle


# ------------------------------------------------------------- generators


def synthetic(n=800, n_noise=14, effect=0.0, seed=0, n_values=4):
    """
    A dataset with a single planted rule of controllable strength.

    effect = 0.0  -> no structure at all (for false-positive measurement)
    effect = 0.5  -> P(target=yes | source=a) rises from base to base+0.5

    Returns (DataFrame, planted_pair or None).
    """
    rng = np.random.default_rng(seed)
    vals = [f"v{i}" for i in range(n_values)]

    df = pd.DataFrame()
    df["source"] = rng.choice(vals, n)

    base = 0.25
    p = np.where(df.source == vals[0], min(base + effect, 0.99), base)
    df["target"] = np.where(rng.random(n) < p, "yes", "no")

    for i in range(n_noise):
        df[f"noise{i:02d}"] = rng.choice(vals, n)

    planted = None if effect == 0 else ("source", "target")
    return df, planted


# ------------------------------------------------------------- one trial


@dataclass
class TrialResult:
    survivors: int
    recovered: bool
    false_survivors: int
    naive_rules: int
    ghost_rate: float


def _is_planted(rule_label: str) -> bool:
    """A survivor counts as the planted rule only if it links source<->target."""
    return "source=" in rule_label and "target=" in rule_label


def run_trial(df, planted, n_null=40, alpha=0.05, seed=0, strict=True,
              **mine_kwargs) -> TrialResult:
    """
    One full pipeline run. `strict=True` counts only family-wise survivors.
    """
    # Deliberately permissive: this is a STRESS test. Tight thresholds would
    # suppress ghosts by themselves and flatter the validator.
    params = dict(min_support=0.03, min_confidence=0.40, min_lift=1.10,
                  max_len=2, max_cardinality=10)
    params.update(mine_kwargs)

    found, names = rules.mine(df, **params)
    if not found:
        return TrialResult(0, False, 0, 0, 0.0)

    minimal = rules.remove_redundant(found)
    nm = IndependentShuffle(df)
    all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
        nm, n_replicates=n_null, seed=seed + 1000, **params
    )
    rules.calibrate(minimal, all_lifts, max_lifts, alpha=alpha)

    table = rules.rules_to_frame(minimal, names)
    keep = (table.verdict == "REAL (family-wise)") if strict else (
        table.verdict != "GHOST"
    )
    survivors = table[keep]

    if len(survivors) == 0:
        recovered, false_survivors = False, 0
    elif planted is None:
        recovered, false_survivors = False, len(survivors)
    else:
        mask = survivors.rule.map(_is_planted).astype(bool).to_numpy()
        recovered = bool(mask.any())
        false_survivors = int((~mask).sum())

    return TrialResult(
        survivors=len(survivors),
        recovered=recovered,
        false_survivors=false_survivors,
        naive_rules=len(found),
        ghost_rate=float(counts.mean()),
    )


# ------------------------------------------------------------- suites


def false_positive_rate(n_trials=30, n=800, n_noise=14, n_null=40, alpha=0.05,
                        seed=0, verbose=True) -> dict:
    """
    Run the pipeline on structureless data repeatedly. Any survivor is an
    error by construction.
    """
    if verbose:
        print(f"False-positive rate: {n_trials} structureless datasets "
              f"(n={n}, {n_noise + 2} columns)")

    any_survivor, total_survivors, naive_total, ghosts = 0, 0, 0, []
    for t in range(n_trials):
        df, _ = synthetic(n=n, n_noise=n_noise, effect=0.0, seed=seed + t)
        r = run_trial(df, None, n_null=n_null, alpha=alpha, seed=seed + t)
        any_survivor += int(r.survivors > 0)
        total_survivors += r.survivors
        naive_total += r.naive_rules
        ghosts.append(r.ghost_rate)
        if verbose and (t + 1) % 10 == 0:
            print(f"  {t + 1}/{n_trials} trials, "
                  f"{any_survivor} with >=1 false survivor")

    out = {
        "trials": n_trials,
        "alpha": alpha,
        "fwer_observed": any_survivor / n_trials,
        "mean_false_survivors": total_survivors / n_trials,
        "mean_naive_rules": naive_total / n_trials,
        "mean_ghost_rate": float(np.mean(ghosts)),
    }
    if verbose:
        print(f"\n  Naive Apriori reported {out['mean_naive_rules']:.1f} rules "
              f"per structureless dataset.")
        print(f"  After validation: {out['mean_false_survivors']:.2f} survivors "
              f"per dataset.")
        print(f"  Family-wise error rate: {out['fwer_observed']:.1%} "
              f"(target <= {alpha:.0%}) "
              f"{'PASS' if out['fwer_observed'] <= alpha * 2 else 'FAIL'}")
    return out


def power_curve(effects=(0.05, 0.10, 0.20, 0.30, 0.50), n_trials=15, n=800,
                n_noise=14, n_null=40, alpha=0.05, seed=100,
                verbose=True) -> pd.DataFrame:
    """
    Recovery rate as a function of effect size -- the method's sensitivity.

    The smallest effect reaching ~80% power is your minimum detectable effect.
    Quote it with every negative result.
    """
    if verbose:
        print(f"\nPower curve: {len(effects)} effect sizes x {n_trials} trials")

    rows = []
    for e in effects:
        rec, false_s = 0, 0
        for t in range(n_trials):
            df, planted = synthetic(n=n, n_noise=n_noise, effect=e,
                                    seed=seed + int(e * 1000) + t)
            r = run_trial(df, planted, n_null=n_null, alpha=alpha,
                          seed=seed + t)
            rec += int(r.recovered)
            false_s += r.false_survivors
        rows.append(
            {
                "effect_size": e,
                "power": rec / n_trials,
                "mean_false_survivors": false_s / n_trials,
                "trials": n_trials,
            }
        )
        if verbose:
            print(f"  effect={e:.2f}  power={rec / n_trials:.0%}  "
                  f"false survivors/run={false_s / n_trials:.2f}")

    out = pd.DataFrame(rows)
    adequate = out[out.power >= 0.8]
    out.attrs["min_detectable_effect"] = (
        float(adequate.effect_size.min()) if len(adequate) else None
    )
    if verbose:
        mde = out.attrs["min_detectable_effect"]
        print(f"\n  Minimum detectable effect (80% power): "
              f"{mde if mde is not None else 'not reached in tested range'}")
    return out


def false_discovery_rate(n_trials=20, effect=0.35, n=800, n_noise=14,
                         n_null=40, alpha=0.05, seed=200, verbose=True) -> dict:
    """
    Among everything reported on data that DOES contain a real rule, what
    fraction is spurious?
    """
    if verbose:
        print(f"\nFalse discovery rate: {n_trials} datasets with a planted "
              f"effect of {effect}")

    true_pos, false_pos = 0, 0
    for t in range(n_trials):
        df, planted = synthetic(n=n, n_noise=n_noise, effect=effect,
                                seed=seed + t)
        r = run_trial(df, planted, n_null=n_null, alpha=alpha, seed=seed + t)
        true_pos += int(r.recovered)
        false_pos += r.false_survivors

    total = true_pos + false_pos
    fdr = false_pos / total if total else 0.0

    # An FDR estimated from a handful of discoveries is mostly sampling noise:
    # with 9 discoveries the estimate moves in 11-point jumps. Refuse to grade
    # it below a usable denominator rather than reporting a spurious PASS/FAIL.
    MIN_DISCOVERIES = 40
    if total < MIN_DISCOVERIES:
        grade = f"INCONCLUSIVE (only {total} discoveries; need ~{MIN_DISCOVERIES})"
    elif fdr <= alpha * 2:
        grade = "PASS"
    else:
        grade = "FAIL"

    out = {
        "trials": n_trials,
        "effect": effect,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "total_discoveries": total,
        "fdr": fdr,
        "fdr_grade": grade,
        "recall": true_pos / n_trials,
    }
    if verbose:
        print(f"  {true_pos} true positives, {false_pos} false positives")
        print(f"  FDR = {fdr:.1%} (target <= {alpha:.0%}) {grade}")
        print(f"  Recall = {out['recall']:.0%}")
    return out


def full_benchmark(quick=False, verbose=True) -> dict:
    """
    Complete validation suite. quick=True for a fast smoke run.

    The output table is the centrepiece of your methods section.
    """
    # n_null must clear rules.minimum_replicates(alpha) or family-wise tests
    # are inert -- see the guard in rules.check_replicates.
    cfg = (
        dict(n_trials_fp=10, n_trials_pow=6, n_trials_fdr=8, n_null=24, n=600)
        if quick
        else dict(n_trials_fp=30, n_trials_pow=15, n_trials_fdr=20,
                  n_null=40, n=800)
    )

    if verbose:
        print("=" * 78)
        print("VALIDATOR BENCHMARK" + ("  (quick mode)" if quick else ""))
        print("=" * 78)

    fp = false_positive_rate(n_trials=cfg["n_trials_fp"], n=cfg["n"],
                             n_null=cfg["n_null"], verbose=verbose)
    pw = power_curve(n_trials=cfg["n_trials_pow"], n=cfg["n"],
                     n_null=cfg["n_null"], verbose=verbose)
    fdr = false_discovery_rate(n_trials=cfg["n_trials_fdr"], n=cfg["n"],
                               n_null=cfg["n_null"], verbose=verbose)

    if verbose:
        print("\n" + "=" * 78)
        print("BENCHMARK SUMMARY")
        print("=" * 78)
        print(f"  False positive rate (FWER) : {fp['fwer_observed']:.1%}")
        print(f"  False discovery rate       : {fdr['fdr']:.1%}")
        print(f"  Minimum detectable effect  : "
              f"{pw.attrs.get('min_detectable_effect')}")
        print(f"  Ghosts suppressed          : "
              f"{fp['mean_naive_rules']:.1f} -> "
              f"{fp['mean_false_survivors']:.2f} per structureless dataset")

    return {"false_positive": fp, "power": pw, "false_discovery": fdr}
