"""
Small built-in datasets with KNOWN ground truth, for demos and case studies.

pure_noise  -- no structure at all. Every rule mined from it is a ghost.
planted     -- two real rules buried among noise columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pure_noise(n: int = 1500, n_columns: int = 16, seed: int = 0) -> pd.DataFrame:
    """Independent categorical columns: the right answer is 'nothing'."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        f"col{i:02d}": rng.choice([f"v{j}" for j in range(rng.integers(2, 5))], n)
        for i in range(n_columns)
    })


def planted(n: int = 1500, n_noise: int = 10, seed: int = 7) -> pd.DataFrame:
    """
    Two planted rules among noise columns:

        weather=rain           -> delay=yes    (80% vs 15% otherwise)
        promo=yes, weekend=yes -> basket=large (75% vs 20% otherwise)
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"weather": rng.choice(["sun", "rain", "cloud"], n,
                                             p=[0.5, 0.25, 0.25])})
    df["delay"] = np.where(
        np.where(df.weather == "rain", rng.random(n) < 0.8,
                 rng.random(n) < 0.15), "yes", "no")
    df["promo"] = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    df["weekend"] = rng.choice(["yes", "no"], n, p=[0.28, 0.72])
    combo = (df.promo == "yes") & (df.weekend == "yes")
    df["basket"] = np.where(combo,
                            np.where(rng.random(n) < 0.75, "large", "small"),
                            np.where(rng.random(n) < 0.20, "large", "small"))
    for i in range(n_noise):
        df[f"noise{i:02d}"] = rng.choice([f"v{j}" for j in range(3)], n)
    return df
