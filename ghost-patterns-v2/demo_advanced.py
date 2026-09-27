"""
Ghost Patterns -- advanced demonstrations (v2 features).

  E. LEAKAGE DETECTION     Catch self-inflicted structure before mining.
  F. NULL MODEL CHOICE     The same data, two nulls, opposite conclusions.
  G. VALIDATOR BENCHMARK   The method's own false-positive rate and power.

Run:  python demo_advanced.py [--full]
"""

import sys

import numpy as np
import pandas as pd

from ghostpatterns import rules
from ghostpatterns.benchmark import full_benchmark
from ghostpatterns.leakage import check_leakage
from ghostpatterns.nullmodels import IndependentShuffle, SwapRandomization

pd.set_option("display.width", 200)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------- E


def experiment_leakage():
    banner("EXPERIMENT E -- LEAKAGE DETECTION")
    print("The lottery, as a student would naively prepare it: sorted balls")
    print("plus engineered features. Checked on the RAW data, before binning.\n")

    rng = np.random.default_rng(11)
    draws = [sorted(rng.choice(np.arange(1, 50), 6, replace=False))
             for _ in range(1200)]
    raw = pd.DataFrame(draws, columns=[f"ball{i+1}" for i in range(6)])
    raw["total"] = raw[[f"ball{i+1}" for i in range(6)]].sum(axis=1)
    raw["has_low"] = np.where(raw.ball1 <= 10, "yes", "no")
    raw["spread"] = raw.ball6 - raw.ball1

    rep = check_leakage(raw, verbose=False)
    print(rep.to_frame()[rep.to_frame().kind != "strong_association"]
          .to_string(index=False))

    print(f"\n{len(rep.hard)} hard issues. Every one generates rules that are")
    print("true by construction. This is what produced the 280 'significant'")
    print("lottery rules in demo.py -- caught here in under a second, before")
    print("any mining happened.\n")

    print("Now the same lottery prepared honestly (draw order, no derived cols):")
    clean = pd.DataFrame(
        [rng.choice(np.arange(1, 50), 6, replace=False) for _ in range(1200)],
        columns=[f"ball{i+1}" for i in range(6)])
    rep2 = check_leakage(clean, verbose=False)
    print(f"  hard issues: {len(rep2.hard)}   clean: {rep2.clean}")

    print("\nCAVEAT worth stating in your report: run this on RAW data.")
    print("Binning launders leakage -- it weakens the signatures these")
    print("detectors look for without weakening the spurious rules.")


# --------------------------------------------------------------- F


def make_baskets(n=2500, n_items=18, seed=4):
    """
    Synthetic transactions with HETEROGENEOUS basket sizes and NO real
    item-to-item affinity. Basket size and item popularity alone.

    Ground truth: there are no genuine associations. Any rule is a ghost.
    """
    rng = np.random.default_rng(seed)

    # Heavy-tailed basket sizes, like real retail: mostly small, a few huge.
    sizes = np.clip(rng.negative_binomial(2, 0.25, n) + 1, 1, n_items)
    popularity = rng.dirichlet(np.ones(n_items) * 0.7)

    rows = np.zeros((n, n_items), dtype=bool)
    for i, s in enumerate(sizes):
        picked = rng.choice(n_items, size=int(s), replace=False, p=popularity)
        rows[i, picked] = True

    return pd.DataFrame(rows, columns=[f"item{j:02d}" for j in range(n_items)])


def experiment_null_choice():
    banner("EXPERIMENT F -- NULL MODEL CHOICE CHANGES THE ANSWER")
    print("Synthetic basket data: heterogeneous basket sizes, popular and rare")
    print("items, and NO genuine item affinity. Every rule found is a ghost.\n")

    baskets = make_baskets()
    sizes = baskets.sum(axis=1)
    print(f"{len(baskets)} baskets, {baskets.shape[1]} items")
    print(f"Basket size: mean {sizes.mean():.1f}, median {sizes.median():.0f}, "
          f"max {sizes.max()}  <- heterogeneous, as in real retail\n")

    params = dict(min_support=0.03, min_confidence=0.35, min_lift=1.1,
                  max_len=2, max_cardinality=2)

    found, names = rules.mine(baskets, **params)
    minimal = rules.remove_redundant(found)
    print(f"Naive mining: {len(found)} rules ({len(minimal)} distinct)")

    results = {}
    for label, nm in [
        ("IndependentShuffle (destroys basket sizes)", IndependentShuffle(baskets)),
        ("SwapRandomization (preserves both margins)",
         SwapRandomization(baskets, n_swaps=3 * len(baskets))),
    ]:
        import copy
        candidates = copy.deepcopy(minimal)
        all_l, max_l, counts = rules.ghost_lift_distribution(
            nm, n_replicates=25, seed=1, **params)
        rules.calibrate(candidates, all_l, max_l)
        table = rules.rules_to_frame(candidates, names)
        surv = int((table.verdict != "GHOST").sum())
        results[label] = (surv, float(counts.mean()),
                          float(np.max(all_l)) if len(all_l) else 0.0)
        print(f"\n  {label}")
        print(f"    ghost rate      : {counts.mean():.1f} rules per null run")
        print(f"    strongest ghost : lift {results[label][2]:.3f}")
        print(f"    SURVIVORS       : {surv} / {len(minimal)}")

    wrong = results["IndependentShuffle (destroys basket sizes)"][0]
    right = results["SwapRandomization (preserves both margins)"][0]

    print("\nGROUND TRUTH: zero real associations exist in this data.")
    print(f"  Wrong null  -> {wrong} 'discoveries', all false")
    print(f"  Correct null -> {right} 'discoveries'")
    if wrong > right:
        print(f"\n  Choosing the wrong null manufactured {wrong - right} false")
        print("  findings. IndependentShuffle destroys the row margins, so the")
        print("  null loses the big baskets that generate co-occurrence")
        print("  mechanically -- making ordinary rules look extraordinary.")
        print("  Curveball keeps basket sizes and item frequencies exact.")
    else:
        print("\n  The two nulls agree here. That is itself worth reporting:")
        print("  margin effects matter when basket sizes are heterogeneous AND")
        print("  support thresholds are low enough to admit large baskets.")
    print("\n  Report which null you used and why. It is a modelling decision,")
    print("  not a default, and it can change the conclusion.")


# --------------------------------------------------------------- G


def experiment_benchmark(quick=True):
    banner("EXPERIMENT G -- BENCHMARKING THE VALIDATOR ITSELF")
    print("Every ghost-hunting method claims to work. This measures whether")
    print("this one does, on synthetic data with known ground truth.\n")
    full_benchmark(quick=quick)


def main():
    quick = "--full" not in sys.argv
    experiment_leakage()
    experiment_null_choice()
    experiment_benchmark(quick=quick)

    banner("SUMMARY")
    print("E. Leakage is caught mechanically, before mining -- closing the gap")
    print("   that demo.py's Experiment B exposed.")
    print("F. Null model choice is a modelling decision that changes results.")
    print("G. The validator's own error rates are measured, not assumed.")


if __name__ == "__main__":
    main()
