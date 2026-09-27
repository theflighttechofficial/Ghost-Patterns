"""
Ghost Patterns -- end-to-end demonstration.

Four experiments:

  A. PLANTED SIGNAL. Two known real rules buried in 18 noise columns. Does
     the pipeline recover them and reject the rest? This validates the
     validator; without it there is no evidence the method works.

  B. LOTTERY, MINED NAIVELY. Sorted draws plus engineered features. Produces
     hundreds of rules that survive every statistical test -- and are still
     worthless. Significance is not meaning.

  C. LOTTERY, MINED HONESTLY. Same lottery, draw order preserved, no derived
     features. Now the pipeline should find nothing, because there is nothing.

  D. DIMENSIONALITY CURSE. Anomaly detection on featureless noise as
     dimensionality rises.

Run:  python demo.py
"""

import numpy as np
import pandas as pd

from ghostpatterns import clusters, rules
from ghostpatterns.anomaly import anomalies_vs_null, dimensionality_curse_curve
from ghostpatterns.nullmodels import IndependentShuffle
from ghostpatterns.permutation import attribute_to_columns

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

MINE = dict(min_support=0.04, min_confidence=0.55, min_lift=1.2, max_len=3)
N_NULL = 60


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ---------------------------------------------------------------- datasets


def make_planted(n=1500, n_noise=18, seed=7):
    """
    Two real rules planted in noise.

      RULE 1: weather=rain -> delay=yes                 (~80%)
      RULE 2: {promo=yes, weekend=yes} -> basket=large  (~75%)

    Everything else is independent noise.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()

    df["weather"] = rng.choice(["sun", "rain", "cloud"], n, p=[0.5, 0.25, 0.25])
    base_delay = rng.random(n) < 0.15
    rain_delay = rng.random(n) < 0.80
    df["delay"] = np.where(df.weather == "rain", rain_delay, base_delay)
    df["delay"] = np.where(df.delay, "yes", "no")

    df["promo"] = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    df["weekend"] = rng.choice(["yes", "no"], n, p=[0.28, 0.72])
    combo = (df.promo == "yes") & (df.weekend == "yes")
    big = rng.random(n) < 0.75
    small = rng.random(n) < 0.20
    df["basket"] = np.where(
        combo, np.where(big, "large", "small"), np.where(small, "large", "small")
    )

    for i in range(n_noise):
        k = rng.integers(2, 5)
        df[f"noise{i:02d}"] = rng.choice([f"v{j}" for j in range(k)], n)

    return df


def make_lottery(n_draws=1200, seed=11, naive=True):
    """
    Six numbers from 1-49 drawn independently. No lottery structure exists.

    naive=True  : sort the balls and add engineered features -- exactly what a
                  student does without thinking. Sorting makes ball1<ball2<...
                  a hard constraint, and the derived features are functions of
                  the balls. Both create REAL dependence that has nothing to do
                  with the lottery.

    naive=False : keep draw order, no derived features. Genuinely structureless.
    """
    rng = np.random.default_rng(seed)
    draws = [rng.choice(np.arange(1, 50), size=6, replace=False) for _ in range(n_draws)]

    if naive:
        rows = [sorted(d) for d in draws]
    else:
        rows = draws

    df = pd.DataFrame(rows, columns=[f"ball{i+1}" for i in range(6)])
    ball_cols = list(df.columns)

    if naive:
        df["sum_bucket"] = pd.cut(
            df[ball_cols].sum(axis=1), 5,
            labels=["vlow", "low", "mid", "high", "vhigh"]
        ).astype(str)
        df["n_even"] = (df[ball_cols] % 2 == 0).sum(axis=1).astype(str)
        df["has_low"] = np.where(df.ball1 <= 10, "yes", "no")
        df["spread"] = pd.cut(
            df.ball6 - df.ball1, 4, labels=["tight", "narrow", "wide", "vwide"]
        ).astype(str)

    for c in ball_cols:
        df[c] = pd.cut(df[c], 5, labels=["a", "b", "c", "d", "e"]).astype(str)
    return df


# ---------------------------------------------------------------- pipeline


def run_rules(df, title, planted_terms=None, note=None):
    banner(title)
    if note:
        print(note + "\n")

    found, names = rules.mine(df, **MINE)
    print(f"Naive mining returned {len(found)} rules above the thresholds.")
    if not found:
        print("Nothing to validate.")
        return None

    minimal = rules.remove_redundant(found)
    print(f"After removing padded/redundant variants: {len(minimal)} distinct findings.")
    print(f"  -> {len(found) - len(minimal)} of the original rules were the same")
    print("     finding with an irrelevant condition bolted on.")

    nm = IndependentShuffle(df)
    all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
        nm, n_replicates=N_NULL, seed=1, **MINE
    )

    print(f"\nGHOST BASELINE ({N_NULL} null replicates, identical pipeline)")
    print(f"  rules per null run : mean {counts.mean():.1f}, max {counts.max()}")
    print(f"  ghost rate         : a structureless copy of this dataset yields "
          f"{counts.mean():.0f} rules")
    if len(all_lifts):
        print(f"  null lift          : median {np.median(all_lifts):.3f}, "
              f"95th pct {np.percentile(all_lifts, 95):.3f}, max {all_lifts.max():.3f}")
    print(f"  best-of-run lift   : mean {max_lifts.mean():.3f}, max {max_lifts.max():.3f}")

    rules.calibrate(minimal, all_lifts, max_lifts, alpha=0.05)
    table = rules.rules_to_frame(minimal, names)

    vc = table.verdict.value_counts().to_dict()
    print("\nVERDICTS (on the deduplicated set)")
    for v in ["REAL (family-wise)", "likely real (FDR)", "GHOST"]:
        print(f"  {v:<22} {vc.get(v, 0)}")

    survivors = table[table.verdict != "GHOST"]
    print(f"\nSURVIVORS ({len(survivors)} of {len(table)}):")
    if len(survivors):
        print(survivors.head(10).to_string(index=False))
    else:
        print("  none -- every pattern found was a ghost")

    if planted_terms:
        print("\nRECOVERY CHECK (did we find what we planted?)")
        for term in planted_terms:
            hits = table[table.rule.str.contains(term, regex=False)]
            if len(hits):
                b = hits.iloc[0]
                print(f"  {term:<40} FOUND  lift={b.lift:.2f}  {b.verdict}")
            else:
                print(f"  {term:<40} MISSED")
    return table


def main():
    # ------------------------------------------------ A
    planted = make_planted()
    run_rules(
        planted,
        "EXPERIMENT A -- PLANTED SIGNAL (2 real rules hidden in 18 noise columns)",
        planted_terms=["{weather=rain} -> {delay=yes}",
                       "-> {basket=large}"],
    )

    banner("EXPERIMENT A2 -- WHICH COLUMNS ARE LOAD-BEARING?")
    print("Statistic: number of rules with lift > 1.5. Shuffle one column at a")
    print("time. Columns whose shuffling destroys the rules are where the")
    print("structure actually lives; the rest are decoration.\n")

    def n_strong_rules(d):
        r, _ = rules.mine(d, min_support=0.05, min_confidence=0.6,
                          min_lift=1.5, max_len=2, max_cardinality=6)
        return len(r)

    attrib = attribute_to_columns(
        planted[["weather", "delay", "promo", "weekend", "basket",
                 "noise00", "noise01", "noise02"]],
        n_strong_rules, n_permutations=40, seed=3,
    )
    print(attrib.to_string(index=False))

    # ------------------------------------------------ B
    run_rules(
        make_lottery(naive=True),
        "EXPERIMENT B -- LOTTERY, MINED NAIVELY (sorted balls + derived features)",
        note=(
            "The balls are sorted, so ball1 <= ball2 <= ... is a hard constraint,\n"
            "and has_low/sum_bucket/spread are arithmetic functions of the balls.\n"
            "Watch what the validation says."
        ),
    )
    print("\nREADING THIS RESULT -- the most important lesson in the project:")
    print("  These rules PASS. The statistics are not broken; the dependence is")
    print("  genuinely there. But it comes from SORTING and from FEATURE LEAKAGE,")
    print("  not from the lottery. A null model tests whether structure exists.")
    print("  It cannot tell you the structure is an artefact of your own")
    print("  preprocessing. Only reading the rules can do that.")
    print("  Statistical significance is necessary, not sufficient.")

    # ------------------------------------------------ C
    run_rules(
        make_lottery(naive=False),
        "EXPERIMENT C -- LOTTERY, MINED HONESTLY (draw order, no derived features)",
        note=(
            "Same lottery, leakage removed. If the method is sound, the haul\n"
            "should collapse to nothing."
        ),
    )

    # ------------------------------------------------ clustering
    banner("EXPERIMENT C2 -- CLUSTERING PURE NOISE")
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(rng.normal(size=(600, 8)), columns=[f"f{i}" for i in range(8)])
    sil = clusters.silhouette_vs_null(noise, k_range=range(2, 7), n_null=20, seed=2)
    print("Silhouette on structureless data vs matched null:\n")
    print(sil.to_string(index=False))
    print("\nSeveral raw silhouettes look acceptable by the usual rule of thumb.")
    print("The 'excess' column is what matters, and it sits at ~0.")

    print(f"\nStability check: {clusters.stability(noise, k=3, n_runs=12, n_null=8, seed=2)}")

    # ------------------------------------------------ D
    banner("EXPERIMENT D -- ANOMALY DETECTION AND THE CURSE OF DIMENSIONALITY")
    print("Pure Gaussian noise. Zero anomalies exist. Every flag is a ghost.\n")
    print(dimensionality_curse_curve(n=500).to_string(index=False))

    print("\nSame data, validated against a matched null instead of a fixed quota:")
    for d in (5, 50):
        X = pd.DataFrame(np.random.default_rng(9).normal(size=(500, d)))
        print(f"  d={d:<4} {anomalies_vs_null(X, n_null=15, seed=4)}")

    banner("SUMMARY")
    print("A. Planted structure is recovered and noise is rejected.")
    print("B. Statistically real does not mean meaningful -- leakage passes every test.")
    print("C. With leakage removed, a structureless dataset yields nothing.")
    print("D. Anomaly counts are set by the contamination parameter, not the data.")
    print("\nReport the ghost rate next to every mining result you publish.")


if __name__ == "__main__":
    main()
