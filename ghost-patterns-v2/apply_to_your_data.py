"""
Template: run the full validated pipeline on your own dataset.

    python apply_to_your_data.py mydata.csv

Edit the CONFIG block, run it, and you have the results section of your report.
"""

import sys

import numpy as np
import pandas as pd

from ghostpatterns import clusters, rules
from ghostpatterns.anomaly import anomalies_vs_null, relative_contrast
from ghostpatterns.nullmodels import IndependentShuffle
from ghostpatterns.permutation import attribute_to_columns

# ----------------------------------------------------------------- CONFIG

MINE = dict(
    min_support=0.05,      # lower = more rules AND more ghosts
    min_confidence=0.60,
    min_lift=1.20,
    max_len=3,
    max_cardinality=15,    # drop columns with more distinct values than this
)

N_NULL = 100               # null replicates; 100+ for final results
ALPHA = 0.05
NULL_MODEL = IndependentShuffle   # BlockShuffle/PhaseRandomized for time series

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns")

    categorical = df.select_dtypes(exclude=[np.number])
    numeric = df.select_dtypes(include=[np.number])
    print(f"  {categorical.shape[1]} categorical, {numeric.shape[1]} numeric")

    # ---------------------------------------------------- LEAKAGE WARNING
    section("STEP 0 -- CHECK FOR LEAKAGE BEFORE YOU MINE ANYTHING")
    print("Experiment B in the demo shows leakage passes every statistical")
    print("test. No null model will catch it for you. Check by hand:")
    print("  - Is any column a function of another? (totals, ratios, flags)")
    print("  - Are rows sorted in a way that creates ordering constraints?")
    print("  - Do any two columns encode the same underlying quantity?")
    print("Drop those columns now, or your 'findings' will be tautologies.")

    # ---------------------------------------------------- RULES
    if categorical.shape[1] >= 2:
        section("STEP 1 -- ASSOCIATION RULES, CALIBRATED")

        found, names = rules.mine(categorical, **MINE)
        print(f"Naive mining: {len(found)} rules")
        if not found:
            print("No rules at these thresholds. Lower min_support or min_lift.")
        else:
            minimal = rules.remove_redundant(found)
            print(f"After redundancy filtering: {len(minimal)} distinct findings")

            nm = NULL_MODEL(categorical)
            all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
                nm, n_replicates=N_NULL, seed=1, **MINE
            )

            print(f"\nGHOST RATE: a structureless copy of this dataset yields "
                  f"{counts.mean():.0f} rules (max {counts.max()})")
            if len(all_lifts):
                print(f"Null lift: median {np.median(all_lifts):.3f}, "
                      f"95th pct {np.percentile(all_lifts, 95):.3f}, "
                      f"max {all_lifts.max():.3f}")

            rules.calibrate(minimal, all_lifts, max_lifts, alpha=ALPHA)
            table = rules.rules_to_frame(minimal, names)

            print("\nVERDICTS")
            for v, c in table.verdict.value_counts().items():
                print(f"  {v:<22} {c}")

            survivors = table[table.verdict != "GHOST"]
            print(f"\nSURVIVORS ({len(survivors)}):")
            print(survivors.head(25).to_string(index=False) if len(survivors)
                  else "  none -- every pattern found was a ghost")

            table.to_csv("results_rules.csv", index=False)
            print("\nFull table -> results_rules.csv")

            # which columns carry the structure
            if categorical.shape[1] <= 12:
                section("STEP 2 -- COLUMN ATTRIBUTION")

                def n_strong(d):
                    r, _ = rules.mine(d, min_support=MINE["min_support"],
                                      min_confidence=MINE["min_confidence"],
                                      min_lift=1.5, max_len=2,
                                      max_cardinality=MINE["max_cardinality"])
                    return len(r)

                print(attribute_to_columns(
                    categorical, n_strong, n_permutations=50, seed=3
                ).to_string(index=False))
                print("\nColumns with load_bearing=False contribute nothing.")

    # ---------------------------------------------------- CLUSTERS
    if numeric.shape[1] >= 2:
        section("STEP 3 -- CLUSTERING, CALIBRATED")

        sil = clusters.silhouette_vs_null(numeric, k_range=range(2, 9),
                                          n_null=25, seed=2)
        print(sil.to_string(index=False))
        print("\n'excess' is real silhouette minus null silhouette. Near zero")
        print("means your clusters are an artefact of k-means, not the data.")

        good = sil[sil.beats_noise].k.tolist()
        if good:
            print(f"\nk values beating noise: {good}. Stability check:")
            for k in good:
                print(f"  {clusters.stability(numeric, k=k, n_runs=15, n_null=10, seed=2)}")
        else:
            print("\nNo k beats the null. Report this as a negative result --")
            print("it is a finding, not a failure.")

        gap = clusters.gap_statistic(numeric, k_range=range(1, 10), n_null=25, seed=2)
        print(f"\nGap statistic chooses k = {gap.attrs.get('chosen_k')}")
        print(gap.to_string(index=False))

    # ---------------------------------------------------- ANOMALIES
    if numeric.shape[1] >= 2:
        section("STEP 4 -- ANOMALIES, CALIBRATED")

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(numeric)
        rc = relative_contrast(X)
        print(f"Relative contrast: {rc:.3f}", end="  ")
        print("(distances informative)" if rc > 1.0
              else "(degrading)" if rc > 0.3
              else "(COLLAPSED -- distance-based detection is meaningless here)")

        for method in ("isolation_forest", "lof", "mahalanobis"):
            try:
                print(f"  {anomalies_vs_null(numeric, method=method, n_null=25, seed=4)}")
            except Exception as e:
                print(f"  {method}: failed ({e})")

    section("DONE")
    print("Report the ghost rate alongside every finding. Negative results")
    print("are results -- 'we found nothing that survived validation' is a")
    print("stronger claim than an uncalibrated list of rules.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
