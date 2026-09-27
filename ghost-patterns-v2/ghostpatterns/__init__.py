"""
ghostpatterns -- telling real patterns from ghosts in data mining.

Mining methods cannot return nothing. k-means always returns k clusters,
Apriori always returns rules at a low enough support, and Isolation Forest
always flags its contamination quota. So "I found patterns" is not a result.

This package makes it one, by running your identical pipeline on structurally
matched noise and calibrating every finding against what the noise produced.

Quick start:

    import pandas as pd
    from ghostpatterns.pipeline import run, Config

    df = pd.read_csv("mydata.csv")
    result = run(df, Config(min_support=0.05, n_replicates=200))
    print(result.survivors)
    result.save("myrun")

Modules:
    pipeline    one-call orchestration + reproducibility manifest
    leakage     detect self-inflicted structure BEFORE mining
    nullmodels  8 null models, each destroying a specific kind of structure
    rules       Apriori + redundancy filtering + FDR/FWER calibration
    clusters    null-calibrated silhouette, gap statistic, stability
    anomaly     score-distribution testing + dimensionality diagnostics
    permutation single-column attribution
    benchmark   measure the validator's own FPR, power, and FDR
    viz         report figures
"""

from . import (  # noqa: F401
    anomaly,
    benchmark,
    clusters,
    leakage,
    nullmodels,
    permutation,
    pipeline,
    rules,
)
from .leakage import check_leakage, drop_leaking_columns  # noqa: F401
from .nullmodels import (  # noqa: F401
    BlockShuffle,
    ConditionalPermutation,
    GaussianMatched,
    IndependentShuffle,
    MarginalResample,
    PhaseRandomized,
    SwapRandomization,
    UniformHull,
    get_null_model,
)
from .permutation import attribute_to_columns, permutation_test  # noqa: F401
from .pipeline import Config, Result, run  # noqa: F401

__version__ = "2.0.0"

__all__ = [
    "run", "Config", "Result",
    "check_leakage", "drop_leaking_columns",
    "permutation_test", "attribute_to_columns",
    "IndependentShuffle", "MarginalResample", "GaussianMatched",
    "UniformHull", "BlockShuffle", "PhaseRandomized",
    "SwapRandomization", "ConditionalPermutation", "get_null_model",
    "pipeline", "leakage", "nullmodels", "rules", "clusters",
    "anomaly", "permutation", "benchmark",
]
