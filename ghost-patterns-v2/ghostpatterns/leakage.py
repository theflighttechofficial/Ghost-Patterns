"""
Leakage detection: catching the ghosts that null models structurally cannot.

The lottery experiment is the motivating case. Sorted balls plus derived
features produced hundreds of rules that passed every statistical test, and
every one was worthless. The dependence was real -- it came from preprocessing,
not from the phenomenon.

No null model can catch this. A null model asks "is there dependence here?"
and the honest answer was yes. The question that needed asking was "did I
manufacture this dependence myself?"

This module asks it mechanically, before mining:

  1. Functional dependency  -- does column A determine column B?
  2. Duplicate encoding     -- do two columns carry the same information?
  3. Ordering constraints   -- are the columns sorted row-wise?
  4. Arithmetic derivation  -- is one column a sum/difference of others?

Run check_leakage(df) FIRST. Every flag is a column pair that will generate
tautological rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

import numpy as np
import pandas as pd


# ------------------------------------------------------------- information


def _entropy(series: pd.Series) -> float:
    """Shannon entropy in bits."""
    p = series.value_counts(normalize=True, dropna=False).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _conditional_entropy(target: pd.Series, given: pd.Series) -> float:
    """H(target | given) in bits."""
    df = pd.DataFrame({"t": target, "g": given})
    total = 0.0
    n = len(df)
    for _, group in df.groupby("g", observed=True, dropna=False):
        total += (len(group) / n) * _entropy(group["t"])
    return float(total)


def uncertainty_coefficient(target: pd.Series, given: pd.Series) -> float:
    """
    U(target | given) = (H(target) - H(target|given)) / H(target)

    The fraction of the target's information that `given` already contains.
    Asymmetric, which matters: a fine-grained column determines a coarse one
    but not the reverse.

      1.0  -> `given` fully determines `target` (hard functional dependency)
      0.9+ -> near-determinism; will produce tautological rules
      0.0  -> independent
    """
    h = _entropy(target)
    if h == 0:
        return 1.0  # constant column: trivially determined
    return float((h - _conditional_entropy(target, given)) / h)


# ------------------------------------------------------------- findings


@dataclass
class LeakageFinding:
    kind: str
    columns: tuple
    severity: float
    detail: str

    def __str__(self):
        cols = " / ".join(self.columns)
        return f"[{self.severity:.2f}] {self.kind:<22} {cols:<32} {self.detail}"


@dataclass
class LeakageReport:
    findings: list = field(default_factory=list)
    n_columns: int = 0

    @property
    def hard(self):
        return [f for f in self.findings if f.kind != "strong_association"]

    @property
    def advisory(self):
        return [f for f in self.findings if f.kind == "strong_association"]

    @property
    def clean(self) -> bool:
        """True when no HARD leakage was found. Advisory flags do not count."""
        return not self.hard

    def implicated_columns(self) -> set:
        out = set()
        for f in self.findings:
            out.update(f.columns)
        return out

    def summary(self) -> str:
        lines = []
        if self.hard:
            lines += [
                f"HARD LEAKAGE: {len(self.hard)} issue(s). These pairs generate",
                "rules that are true by construction. Drop or merge them BEFORE",
                "mining, or your findings are tautologies.",
                "",
            ]
            lines += [str(f) for f in sorted(self.hard, key=lambda f: -f.severity)]
        else:
            lines += ["No hard leakage detected (no functional dependencies,",
                      "duplicate encodings, row sorting, or arithmetic derivations)."]

        if self.advisory:
            lines += [
                "",
                f"ADVISORY: {len(self.advisory)} strongly-associated pair(s).",
                "Not proof of leakage -- but if this data has been binned or",
                "engineered, the hard detectors may have been laundered. Inspect",
                "these against the RAW data before trusting rules that use them.",
                "",
            ]
            lines += [str(f) for f in sorted(self.advisory,
                                             key=lambda f: -f.severity)[:15]]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["kind", "columns", "severity", "detail"])
        return pd.DataFrame(
            [
                {
                    "kind": f.kind,
                    "columns": " / ".join(f.columns),
                    "severity": round(f.severity, 4),
                    "detail": f.detail,
                }
                for f in self.findings
            ]
        ).sort_values("severity", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------- detectors


def detect_functional_dependencies(df: pd.DataFrame, threshold=0.95,
                                   max_cardinality=50):
    """
    Find column pairs where one determines the other.

    `has_low = ball1 <= 10` scores U = 1.0 against ball1. Any rule involving
    both is arithmetic dressed up as a discovery.
    """
    findings = []
    usable = [c for c in df.columns if df[c].nunique(dropna=False) <= max_cardinality]

    for a, b in permutations(usable, 2):
        u = uncertainty_coefficient(df[b], df[a])
        if u >= threshold:
            kind = "functional_dependency" if u >= 0.999 else "near_determinism"
            findings.append(
                LeakageFinding(
                    kind=kind,
                    columns=(a, b),
                    severity=u,
                    detail=f"{a} determines {b} (U={u:.3f})",
                )
            )
    return findings


def detect_duplicate_encodings(df: pd.DataFrame, threshold=0.95,
                               max_cardinality=50):
    """
    Find pairs that determine each other in BOTH directions -- the same
    variable under two names or two encodings.
    """
    findings = []
    usable = [c for c in df.columns if df[c].nunique(dropna=False) <= max_cardinality]

    for a, b in combinations(usable, 2):
        fwd = uncertainty_coefficient(df[b], df[a])
        rev = uncertainty_coefficient(df[a], df[b])
        both = min(fwd, rev)
        if both >= threshold:
            findings.append(
                LeakageFinding(
                    kind="duplicate_encoding",
                    columns=(a, b),
                    severity=both,
                    detail=f"mutually determining (U={fwd:.3f}/{rev:.3f}); keep one",
                )
            )
    return findings


def detect_row_ordering(df: pd.DataFrame, min_fraction=0.95, min_width=3):
    """
    Detect a row-wise sorted BLOCK of numeric columns.

    This is the lottery trap. Sorting six random draws makes ball1 <= ball2 a
    hard constraint, so every adjacent pair becomes 'associated'. The
    association is real and tells you only that you called sorted().

    Scans every contiguous window of columns rather than testing all of them
    at once -- otherwise a single appended column (a total, a spread) breaks
    global monotonicity and hides the sorted block inside it.
    """
    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    if len(cols) < min_width:
        return []

    arr = numeric.to_numpy(dtype=float)
    valid = ~np.isnan(arr).any(axis=1)
    if valid.sum() < 10:
        return []
    arr = arr[valid]

    best = None
    # Longest window first: report the maximal sorted block, not every subset.
    for width in range(len(cols), min_width - 1, -1):
        for start in range(0, len(cols) - width + 1):
            block = arr[:, start : start + width]
            d = np.diff(block, axis=1)
            nondec = float(np.mean(np.all(d >= 0, axis=1)))
            noninc = float(np.mean(np.all(d <= 0, axis=1)))
            frac = max(nondec, noninc)
            if frac >= min_fraction:
                best = (
                    tuple(cols[start : start + width]),
                    frac,
                    "ascending" if nondec >= noninc else "descending",
                )
                break
        if best:
            break

    if not best:
        return []

    block_cols, frac, direction = best
    return [
        LeakageFinding(
            kind="row_ordering",
            columns=block_cols,
            severity=frac,
            detail=(
                f"{frac:.1%} of rows are sorted {direction} across these "
                f"{len(block_cols)} columns; adjacency rules are forced by the sort"
            ),
        )
    ]


def detect_arithmetic_derivation(df: pd.DataFrame, tolerance=1e-6,
                                 max_columns=12):
    """
    Detect a numeric column that is the sum or difference of two others.

    Totals, subtotals, margins, and spreads are the usual offenders. Cheap
    exhaustive search over triples, capped for tractability.
    """
    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)[:max_columns]
    if len(cols) < 3:
        return []

    findings = []
    arr = {c: numeric[c].to_numpy(dtype=float) for c in cols}

    for target in cols:
        others = [c for c in cols if c != target]
        for a, b in combinations(others, 2):
            for op, sym in ((arr[a] + arr[b], "+"), (abs(arr[a] - arr[b]), "-")):
                diff = np.abs(arr[target] - op)
                scale = max(float(np.nanstd(arr[target])), 1e-12)
                if np.nanmean(diff) / scale < tolerance:
                    findings.append(
                        LeakageFinding(
                            kind="arithmetic_derivation",
                            columns=(target, a, b),
                            severity=1.0,
                            detail=f"{target} == {a} {sym} {b}",
                        )
                    )
                    break
    return findings


# ------------------------------------------------------------- entry point


def detect_strong_association(df: pd.DataFrame, threshold=0.15,
                              max_cardinality=50, max_pairs=400):
    """
    ADVISORY tier: pairs sharing a lot of information without meeting the
    determinism bar.

    Why this tier exists -- a lesson learned the hard way:

    The lottery data is binned into categories before mining. Binning breaks
    the exact functional dependency (`has_low` is no longer perfectly implied
    by a coarse bucket of ball1) and it breaks the sort constraint in code
    space (each column is binned over its own range). So the hard detectors
    above see a clean dataset -- while mining still returns 280 significant
    rules driven entirely by preprocessing.

    The lesson: TRANSFORMATION LAUNDERS LEAKAGE. Binning, scaling, and
    encoding all weaken the signatures the hard detectors look for, without
    weakening the spurious rules that leakage produces.

    Two consequences, and both belong in your report:
      1. Always run check_leakage on the RAW data, before any binning or
         feature engineering.
      2. On already-transformed data, use this advisory tier. It will not
         prove leakage, but it tells you which pairs to go and inspect.

    Symmetric normalised mutual information: I(A;B) / min(H(A), H(B)).
    """
    findings = []
    usable = [c for c in df.columns if df[c].nunique(dropna=False) <= max_cardinality]
    pairs = list(combinations(usable, 2))[:max_pairs]

    for a, b in pairs:
        ha, hb = _entropy(df[a]), _entropy(df[b])
        if ha == 0 or hb == 0:
            continue
        mi = ha - _conditional_entropy(df[a], df[b])
        nmi = mi / min(ha, hb)
        if nmi >= threshold:
            findings.append(
                LeakageFinding(
                    kind="strong_association",
                    columns=(a, b),
                    severity=float(nmi),
                    detail=(
                        f"NMI={nmi:.3f} -- real dependence; verify it comes from "
                        f"the phenomenon and not from preprocessing"
                    ),
                )
            )
    return findings


def check_leakage(df: pd.DataFrame, threshold=0.95, advisory=True,
                  advisory_threshold=0.15, verbose=True) -> LeakageReport:
    """
    Run every detector. Call this before mining anything.

    Returns a LeakageReport; .clean is True when nothing was found.
    """
    findings = []
    findings += detect_functional_dependencies(df, threshold=threshold)
    findings += detect_duplicate_encodings(df, threshold=threshold)
    findings += detect_row_ordering(df)
    findings += detect_arithmetic_derivation(df)

    hard_pairs = {frozenset(f.columns) for f in findings}
    if advisory:
        findings += [
            f for f in detect_strong_association(df, threshold=advisory_threshold)
            if frozenset(f.columns) not in hard_pairs
        ]

    # A mutually-determining pair also trips the one-way detector twice.
    # Keep the stronger label and drop the redundant one-way findings.
    dup_pairs = {frozenset(f.columns) for f in findings
                 if f.kind == "duplicate_encoding"}
    findings = [
        f for f in findings
        if not (f.kind in ("functional_dependency", "near_determinism")
                and frozenset(f.columns) in dup_pairs)
    ]

    report = LeakageReport(findings=findings, n_columns=df.shape[1])
    if verbose:
        print(report.summary())
    return report


def drop_leaking_columns(df: pd.DataFrame, report: LeakageReport,
                         verbose=True) -> pd.DataFrame:
    """
    Conservative automatic cleanup: for each flagged pair, drop the DERIVED
    column (the determined one) and keep the source.

    This is a starting point, not a substitute for judgement. Review what it
    removed -- sometimes the derived column is the one you care about and the
    source should go instead.
    """
    to_drop = set()
    for f in report.findings:
        if f.kind in ("functional_dependency", "near_determinism"):
            to_drop.add(f.columns[1])          # the determined column
        elif f.kind == "duplicate_encoding":
            to_drop.add(f.columns[1])          # keep the first arbitrarily
        elif f.kind == "arithmetic_derivation":
            to_drop.add(f.columns[0])          # the total

    to_drop &= set(df.columns)
    if verbose and to_drop:
        print(f"Dropping {len(to_drop)} derived column(s): {sorted(to_drop)}")
    return df.drop(columns=sorted(to_drop))
