"""
End-to-end analysis of one dataset: leakage check, calibrated association
rules, column attribution, calibrated clustering and calibrated anomalies.

`analyse()` does the computation and returns an `Analysis` object; it touches
no files. `save()` writes the CSVs, figures and reports. The CLI, the
`apply_to_your_data.py` script and the web app all call these two functions,
so every front end produces identical numbers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import clusters, rules
from .anomaly import AnomalyResult, anomalies_vs_null, relative_contrast
from .leakage import LeakageReport, check_leakage, drop_leaking_columns
from .nullmodels import IndependentShuffle
from .permutation import attribute_to_columns

VERDICT_ORDER = {"REAL (family-wise)": 0, "likely real (FDR)": 1, "GHOST": 2}
MAX_ATTRIBUTION_COLUMNS = 12      # attribution is O(columns x permutations)


@dataclass
class Options:
    """Every tunable of an analysis run."""

    target: str | None = None
    min_support: float = 0.05     # lower = more rules AND more ghosts
    min_confidence: float = 0.60
    min_lift: float = 1.20
    max_len: int = 3
    max_cardinality: int = 15     # drop columns with more distinct values
    n_null: int = 100             # null replicates; 100+ for final results
    alpha: float = 0.05
    seed: int = 1
    drop_leaking: bool = False
    attribution: bool = True
    null_model: type = IndependentShuffle

    def mine_kwargs(self) -> dict:
        return dict(min_support=self.min_support,
                    min_confidence=self.min_confidence,
                    min_lift=self.min_lift, max_len=self.max_len,
                    max_cardinality=self.max_cardinality)


@dataclass
class RuleResults:
    n_naive: int = 0
    n_candidates: int = 0
    n_distinct: int = 0
    ghost_rate: float = 0.0
    ghost_max: int = 0
    null_lifts: np.ndarray = field(default_factory=lambda: np.array([]))
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def survivors(self) -> pd.DataFrame:
        if self.table.empty:
            return self.table
        return self.table[self.table.verdict != "GHOST"]

    @property
    def null_p95(self) -> float | None:
        return (float(np.percentile(self.null_lifts, 95))
                if len(self.null_lifts) else None)

    @property
    def null_max(self) -> float | None:
        return float(self.null_lifts.max()) if len(self.null_lifts) else None


@dataclass
class ClusterResults:
    silhouette: pd.DataFrame
    gap: pd.DataFrame
    stability: list = field(default_factory=list)

    @property
    def good_k(self) -> list[int]:
        return self.silhouette[self.silhouette.beats_noise].k.tolist()


@dataclass
class Analysis:
    """Everything one run produced. Pure data; see save() for output."""

    name: str
    n_rows: int
    n_columns: int
    options: Options
    dropped_constant: list = field(default_factory=list)
    dropped_leaking: list = field(default_factory=list)
    leakage: LeakageReport | None = None
    rules: RuleResults | None = None
    attribution: pd.DataFrame | None = None
    clusters: ClusterResults | None = None
    relative_contrast: float | None = None
    anomalies: list[AnomalyResult] = field(default_factory=list)
    seconds: float = 0.0

    def top_findings(self, n: int = 24) -> pd.DataFrame:
        """
        The rules to show first. With a target, take the best rules for EACH
        target value; otherwise the class with the highest base-rate lift
        fills the whole list and the other classes never appear.
        """
        if self.rules is None:
            return pd.DataFrame()
        surv = self.rules.survivors
        if not self.options.target or surv.empty:
            return surv.head(n)
        consequent = surv.rule.str.split(" -> ").str[1]
        per_class = max(1, n // max(consequent.nunique(), 1))
        return surv.groupby(consequent, sort=False, group_keys=False).head(
            per_class)

    def headline(self) -> str:
        """One-sentence plain-English summary of the run."""
        parts = []
        if self.rules is not None:
            r = self.rules
            parts.append(
                f"{len(r.survivors)} of {r.n_distinct} distinct rules beat "
                f"the null (noise alone yields {r.ghost_rate:.1f} rules)")
        if self.clusters is not None:
            k = self.clusters.good_k
            parts.append(f"clusters beat noise at k = {k}" if k
                         else "no clustering beats noise")
        if self.anomalies:
            real = [a.method for a in self.anomalies if a.real]
            parts.append(f"real anomaly excess by {', '.join(real)}" if real
                         else "no anomaly excess over noise")
        return "; ".join(parts) + "." if parts else "Nothing to analyse."


# ----------------------------------------------------------------- helpers


def rank_rules(table: pd.DataFrame) -> pd.DataFrame:
    """Strongest evidence first, then biggest effect, then widest coverage."""
    if table.empty:
        return table
    order = table.verdict.map(VERDICT_ORDER).fillna(3)
    return (table.assign(_v=order)
            .sort_values(["_v", "lift", "support"],
                         ascending=[True, False, False])
            .drop(columns="_v")
            .reset_index(drop=True))


def _section(title: str, log: Callable) -> None:
    log("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------- steps


def _leakage(df, opts, log):
    _section("STEP 0 -- LEAKAGE CHECK", log)
    rep = check_leakage(df, verbose=False)
    log(rep.summary())
    dropped = []
    if opts.drop_leaking and not rep.clean:
        protected = {opts.target} if opts.target else set()
        # Never drop the column the whole analysis is about.
        keep = [f for f in rep.findings
                if not (set(f.columns) & protected
                        and f.kind != "arithmetic_derivation")]
        trimmed = LeakageReport(findings=keep, n_columns=rep.n_columns)
        before = set(df.columns)
        df = drop_leaking_columns(df, trimmed, verbose=False)
        dropped = sorted(before - set(df.columns))
        if dropped:
            log(f"Dropped leaking column(s): {dropped}")
    return df, rep, dropped


def _rules(categorical, opts, log) -> RuleResults:
    _section("STEP 1 -- ASSOCIATION RULES, CALIBRATED", log)
    t0 = time.time()
    res = RuleResults()
    mk = opts.mine_kwargs()
    found, names = rules.mine(categorical, **mk)
    res.n_naive = len(found)
    log(f"Naive mining: {len(found)} rules")
    found = rules.filter_by_target(found, names, opts.target)
    res.n_candidates = len(found)
    if opts.target:
        log(f"Predicting '{opts.target}': {len(found)} candidate rules")
    if not found:
        log("No rules at these thresholds. Lower min_support or min_lift.")
        return res

    minimal = rules.remove_redundant(found)
    res.n_distinct = len(minimal)
    log(f"After redundancy filtering: {len(minimal)} distinct findings")

    all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
        opts.null_model(categorical), n_replicates=opts.n_null,
        seed=opts.seed, target=opts.target, **mk)
    res.null_lifts = all_lifts
    res.ghost_rate = float(counts.mean())
    res.ghost_max = int(counts.max())

    log(f"\nGHOST RATE: a structureless copy of this dataset yields "
        f"{res.ghost_rate:.1f} rules (max {res.ghost_max})")
    if len(all_lifts):
        log(f"Null lift: median {np.median(all_lifts):.3f}, "
            f"95th pct {res.null_p95:.3f}, max {res.null_max:.3f}")

    rules.calibrate(minimal, all_lifts, max_lifts, alpha=opts.alpha)
    res.table = rank_rules(rules.rules_to_frame(minimal, names))

    log("\nVERDICTS")
    for v, c in res.table.verdict.value_counts().items():
        log(f"  {v:<22} {c}")
    if len(all_lifts) == 0:
        log("\nThe null produced no rules, so every rule beats the best "
            "ghost.\np_value/q_value are uninformative here; fwer_p is the "
            "test that matters.")
    log(f"({time.time() - t0:.0f}s)")
    return res


def _attribution(categorical, survivors, opts, log) -> pd.DataFrame | None:
    _section("STEP 2 -- COLUMN ATTRIBUTION", log)
    cols = [c for c in categorical.columns if c != opts.target]
    if len(cols) > MAX_ATTRIBUTION_COLUMNS:
        # Too many to shuffle one by one: test the columns the survivors use.
        used = []
        for rule in survivors.rule:
            for c in cols:
                if f"{c}=" in rule and c not in used:
                    used.append(c)
        cols = used[:MAX_ATTRIBUTION_COLUMNS]
        log(f"Testing the {len(cols)} columns most used by survivors.")
    if not cols:
        log("No columns to attribute.")
        return None

    def n_strong(d):
        r, names = rules.mine(d, min_support=opts.min_support,
                              min_confidence=opts.min_confidence,
                              min_lift=1.5, max_len=2,
                              max_cardinality=opts.max_cardinality)
        return len(rules.filter_by_target(r, names, opts.target))

    attr = attribute_to_columns(categorical, n_strong, columns=cols,
                                n_permutations=50, seed=3)
    log(attr.to_string(index=False))
    log("\nColumns with load_bearing=False contribute nothing.")
    return attr


def _clusters(numeric, log) -> ClusterResults:
    _section("STEP 3 -- CLUSTERING, CALIBRATED", log)
    sil = clusters.silhouette_vs_null(numeric, k_range=range(2, 9),
                                      n_null=25, seed=2)
    log(sil.to_string(index=False))
    log("\n'excess' is real silhouette minus null silhouette. Near zero")
    log("means your clusters are an artefact of k-means, not the data.")
    out = ClusterResults(silhouette=sil, gap=pd.DataFrame())
    if out.good_k:
        log(f"\nk values beating noise: {out.good_k}. Stability check:")
        for k in out.good_k:
            s = clusters.stability(numeric, k=k, n_runs=15, n_null=20, seed=2)
            log(f"  {s}")
            out.stability.append(str(s))
    else:
        log("\nNo k beats the null. Report this as a negative result --")
        log("it is a finding, not a failure.")
    out.gap = clusters.gap_statistic(numeric, k_range=range(1, 10),
                                     n_null=25, seed=2)
    log(f"\nGap statistic chooses k = {out.gap.attrs.get('chosen_k')}")
    log(out.gap.to_string(index=False))
    return out


def _anomalies(numeric, log):
    _section("STEP 4 -- ANOMALIES, CALIBRATED", log)
    from sklearn.preprocessing import StandardScaler
    rc = relative_contrast(StandardScaler().fit_transform(numeric))
    log(f"Relative contrast: {rc:.3f}  ({contrast_reading(rc)})")
    results = []
    for method in ("isolation_forest", "lof", "mahalanobis"):
        try:
            res = anomalies_vs_null(numeric, method=method, n_null=25, seed=4)
        except Exception as e:  # one broken detector must not end the run
            log(f"  {method}: failed ({e})")
            continue
        log(f"  {res}")
        results.append(res)
    return rc, results


def contrast_reading(rc: float) -> str:
    return ("distances informative" if rc > 1.0 else "degrading" if rc > 0.3
            else "COLLAPSED -- distance-based detection is meaningless here")


# ----------------------------------------------------------------- entry


def analyse(df: pd.DataFrame, options: Options | None = None,
            name: str = "dataset", log: Callable = print,
            progress: Callable[[float, str], None] | None = None) -> Analysis:
    """
    Run every applicable step on `df` and return the results.

    log       receives each line of human-readable output (default: print)
    progress  optional callback(fraction_done, step_name) for front ends
    """
    opts = options or Options()
    step = progress or (lambda f, s: None)
    t0 = time.time()

    if opts.target and opts.target not in df.columns:
        raise ValueError(f"target {opts.target!r} is not a column. "
                         f"Columns: {list(df.columns)}")

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        log(f"Dropping constant column(s): {constant}")
        df = df.drop(columns=constant)

    result = Analysis(name=name, n_rows=len(df), n_columns=df.shape[1],
                      options=opts, dropped_constant=constant)
    log(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")

    step(0.02, "Checking for leakage")
    df, result.leakage, result.dropped_leaking = _leakage(df, opts, log)
    categorical = df.select_dtypes(exclude=[np.number])
    numeric = df.select_dtypes(include=[np.number])
    log(f"  {categorical.shape[1]} categorical, {numeric.shape[1]} numeric")

    if categorical.shape[1] >= 2:
        step(0.10, "Mining rules and building the null distribution")
        result.rules = _rules(categorical, opts, log)
        surv = result.rules.survivors
        if opts.attribution and len(surv):
            step(0.55, "Attributing structure to columns")
            result.attribution = _attribution(categorical, surv, opts, log)
    if numeric.shape[1] >= 2:
        step(0.70, "Testing clusters against noise")
        result.clusters = _clusters(numeric, log)
        step(0.85, "Testing anomalies against noise")
        result.relative_contrast, result.anomalies = _anomalies(numeric, log)

    result.seconds = time.time() - t0
    step(1.0, "Done")
    _section("SUMMARY", log)
    log(result.headline())
    return result


def save(result: Analysis, outdir: str) -> dict[str, str]:
    """
    Write CSVs, figures, report.md and report.html into `outdir`.
    Returns {artifact name: path}.
    """
    from . import report, viz

    fig = os.path.join(outdir, "figures")
    os.makedirs(fig, exist_ok=True)
    paths = {}

    def csv(df, name):
        if df is not None and not df.empty:
            p = os.path.join(outdir, name)
            df.to_csv(p, index=False)
            paths[name] = p

    if result.leakage is not None:
        csv(result.leakage.to_frame(), "leakage.csv")

    r = result.rules
    if r is not None and not r.table.empty:
        csv(r.table, "rules_all.csv")
        csv(r.survivors, "rules_survivors.csv")
        top = result.top_findings()
        paths["lift_null.png"] = viz.plot_lift_distribution(
            r.survivors.lift.tolist(), r.null_lifts,
            os.path.join(fig, "lift_null.png"))
        paths["ghost_rate.png"] = viz.plot_ghost_rate(
            [r.n_distinct, int(round(r.ghost_rate))], [len(r.survivors), 0],
            ["real data", "structureless copy"],
            os.path.join(fig, "ghost_rate.png"))
        paths["top_rules.png"] = viz.plot_top_rules(
            top, os.path.join(fig, "top_rules.png"), top=len(top),
            null_p95=r.null_p95, null_max=r.null_max)

    if result.attribution is not None:
        csv(result.attribution, "column_attribution.csv")
        paths["column_attribution.png"] = viz.plot_column_attribution(
            result.attribution, os.path.join(fig, "column_attribution.png"))

    if result.clusters is not None:
        csv(result.clusters.silhouette, "clusters_silhouette.csv")
        csv(result.clusters.gap, "clusters_gap.csv")
        paths["silhouette.png"] = viz.plot_silhouette_vs_null(
            result.clusters.silhouette, os.path.join(fig, "silhouette.png"))

    if result.anomalies:
        csv(pd.DataFrame([{
            "method": a.method, "n_flagged": a.n_flagged,
            "observed_rate": round(a.excess_rate, 4),
            "null_rate": round(a.null_rate, 4),
            "p_value": round(a.p_value, 4), "real": a.real,
        } for a in result.anomalies]), "anomalies.csv")
        for a in result.anomalies:
            name = f"anomaly_{a.method}.png"
            paths[name] = viz.plot_anomaly_scores(a, os.path.join(fig, name))

    paths["report.md"] = report.write_markdown(
        result, os.path.join(outdir, "report.md"))
    paths["report.html"] = report.write_html(
        result, os.path.join(outdir, "report.html"), figures=paths)
    return paths
