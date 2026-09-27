"""
One-call orchestration, with a reproducibility manifest.

Everything in the package wired into the correct order, which matters:

    leakage check  ->  mine  ->  deduplicate  ->  calibrate  ->  report

Getting the order wrong invalidates the result. Deduplicating AFTER
calibration means the multiple-comparison correction was applied to thousands
of copies of the same finding. Skipping the leakage check means you may be
validating your own preprocessing.

Every run writes a manifest: parameter values, library versions, seeds, and a
hash of the input. Re-running with the same manifest reproduces the numbers
exactly, which is what makes results checkable.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import rules
from .leakage import check_leakage
from .nullmodels import IndependentShuffle, NullModel


@dataclass
class Config:
    """Every knob in one place, so a run is fully described by this object."""

    min_support: float = 0.05
    min_confidence: float = 0.60
    min_lift: float = 1.20
    max_len: int = 3
    max_cardinality: int = 15

    n_replicates: int = 100
    alpha: float = 0.05
    seed: int = 0

    redundancy_z: float = 1.96
    check_leakage: bool = True
    leakage_threshold: float = 0.95

    def mine_kwargs(self) -> dict:
        return dict(
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_lift=self.min_lift,
            max_len=self.max_len,
            max_cardinality=self.max_cardinality,
        )

    def validate(self):
        """Catch configurations that cannot produce a valid result."""
        problems = []
        need = rules.minimum_replicates(self.alpha)
        if self.n_replicates < need:
            problems.append(
                f"n_replicates={self.n_replicates} is below {need}, the minimum "
                f"that can reach p<={self.alpha}. Family-wise tests would be inert."
            )
        if not 0 < self.min_support < 1:
            problems.append("min_support must be in (0, 1)")
        if self.min_lift < 1:
            problems.append("min_lift below 1 admits negatively-associated rules")
        return problems


@dataclass
class Result:
    table: pd.DataFrame
    survivors: pd.DataFrame
    ghost_rate: float
    ghost_rate_max: int
    naive_count: int
    deduplicated_count: int
    null_lifts: np.ndarray = field(repr=False, default=None)
    max_null_lifts: np.ndarray = field(repr=False, default=None)
    leakage: object = None
    manifest: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "GHOST PATTERNS RESULT",
            "=" * 70,
            f"  Naive mining output    : {self.naive_count} rules",
            f"  After deduplication    : {self.deduplicated_count} distinct findings",
            f"  Ghost rate             : {self.ghost_rate:.1f} rules from a "
            f"structureless copy (max {self.ghost_rate_max})",
            f"  Survived calibration   : {len(self.survivors)}",
        ]
        if self.naive_count:
            lines.append(
                f"  Suppression            : "
                f"{1 - len(self.survivors) / self.naive_count:.1%} of the naive "
                f"haul was noise or duplication"
            )
        if self.leakage is not None and not self.leakage.clean:
            lines += [
                "",
                f"  WARNING: {len(self.leakage.hard)} hard leakage issue(s) "
                f"detected. Survivors below may be tautologies.",
            ]
        return "\n".join(lines)

    def save(self, prefix="results"):
        """Write the results table and the manifest next to each other."""
        self.table.to_csv(f"{prefix}_rules.csv", index=False)
        with open(f"{prefix}_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        print(f"Wrote {prefix}_rules.csv and {prefix}_manifest.json")


def _fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the input, so a manifest identifies its dataset."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(str(df.shape).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def run(df: pd.DataFrame, config: Config | None = None,
        null_model: NullModel | type | None = None, verbose=True) -> Result:
    """
    The whole pipeline, in the right order.

        result = run(df, Config(min_support=0.05, n_replicates=200))
        print(result.summary())
        print(result.survivors)
        result.save("myrun")
    """
    config = config or Config()
    problems = config.validate()
    if problems:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(problems))

    params = config.mine_kwargs()

    # --- 0. leakage, BEFORE anything else
    leak = None
    if config.check_leakage:
        if verbose:
            print("STEP 0 -- leakage check\n")
        leak = check_leakage(df, threshold=config.leakage_threshold,
                             verbose=verbose)
        if verbose:
            print()

    # --- 1. mine
    found, names = rules.mine(df, **params)
    if verbose:
        print(f"STEP 1 -- mined {len(found)} rules")

    if not found:
        empty = rules.rules_to_frame([], names)
        return Result(empty, empty, 0.0, 0, 0, 0, leakage=leak)

    # --- 2. deduplicate BEFORE correcting, or the correction is meaningless
    minimal = rules.remove_redundant(found, z=config.redundancy_z)
    if verbose:
        print(f"STEP 2 -- {len(minimal)} distinct findings after deduplication")

    # --- 3. ghost baseline
    nm = null_model or IndependentShuffle
    nm = nm(df) if isinstance(nm, type) else nm
    if verbose:
        print(f"STEP 3 -- {config.n_replicates} null replicates "
              f"({nm.__class__.__name__}) ...")
    all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
        nm, n_replicates=config.n_replicates, seed=config.seed, **params)

    # --- 4. calibrate
    rules.calibrate(minimal, all_lifts, max_lifts, alpha=config.alpha)
    table = rules.rules_to_frame(minimal, names)
    survivors = table[table.verdict != "GHOST"].reset_index(drop=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "fingerprint": _fingerprint(df),
            "column_names": list(map(str, df.columns)),
        },
        "config": asdict(config),
        "null_model": nm.__class__.__name__,
        "results": {
            "naive_rules": len(found),
            "deduplicated": len(minimal),
            "survivors": len(survivors),
            "ghost_rate_mean": float(counts.mean()),
            "ghost_rate_max": int(counts.max()),
            "null_lift_p95": float(np.percentile(all_lifts, 95))
            if len(all_lifts) else None,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    result = Result(
        table=table,
        survivors=survivors,
        ghost_rate=float(counts.mean()),
        ghost_rate_max=int(counts.max()),
        naive_count=len(found),
        deduplicated_count=len(minimal),
        null_lifts=all_lifts,
        max_null_lifts=max_lifts,
        leakage=leak,
        manifest=manifest,
    )

    if verbose:
        print("\n" + result.summary())
    return result
