"""
Association rule mining, plus the part everybody skips: deciding which of the
returned rules are real.

Apriori on a dataset with 20 columns will happily return thousands of rules
with lift > 2. So will Apriori on pure noise. The mining is the easy half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .significance import (  # re-exported for convenience
    benjamini_hochberg,
    check_replicates,
    empirical_pvalue,
    minimum_replicates,
)


# ---------------------------------------------------------------- encoding


def one_hot(df: pd.DataFrame, max_cardinality: int = 20):
    """
    Turn a categorical DataFrame into a boolean item matrix.

    Each item is "column=value". Columns with more distinct values than
    max_cardinality are dropped -- high-cardinality columns generate enormous
    numbers of low-support items and are a major ghost source.

    Returns (matrix [n x n_items] bool, item_names list[str]).
    """
    cols, names = [], []
    for col in df.columns:
        values = df[col].astype("object")
        uniques = pd.unique(values.dropna())
        if len(uniques) > max_cardinality:
            continue
        for v in sorted(uniques, key=str):
            cols.append((values == v).to_numpy())
            names.append(f"{col}={v}")
    if not cols:
        raise ValueError("No usable columns after cardinality filtering")
    return np.column_stack(cols), names


# ---------------------------------------------------------------- mining


def frequent_itemsets(matrix: np.ndarray, min_support: float, max_len: int = 3):
    """
    Apriori over a boolean matrix.

    Returns dict {frozenset_of_item_indices: support_float}.
    Uses numpy column ANDs, which is fast enough for a few hundred items.
    """
    n = matrix.shape[0]
    min_count = int(np.ceil(min_support * n))

    counts = matrix.sum(axis=0)
    current = {(i,): int(c) for i, c in enumerate(counts) if c >= min_count}
    all_sets = dict(current)

    k = 1
    while current and k < max_len:
        k += 1
        prev_keys = sorted(current)
        candidates = set()
        # Standard join step: merge two (k-1)-sets sharing a k-2 prefix.
        for i in range(len(prev_keys)):
            a = prev_keys[i]
            for j in range(i + 1, len(prev_keys)):
                b = prev_keys[j]
                if a[:-1] != b[:-1]:
                    break
                cand = a + (b[-1],)
                # Prune: every (k-1)-subset must itself be frequent.
                if all(sub in current for sub in combinations(cand, k - 1)):
                    candidates.add(cand)

        current = {}
        for cand in candidates:
            mask = matrix[:, cand[0]]
            for idx in cand[1:]:
                mask = mask & matrix[:, idx]
                if mask.sum() < min_count:
                    break
            c = int(mask.sum())
            if c >= min_count:
                current[cand] = c
        all_sets.update(current)

    return {frozenset(k_): v / n for k_, v in all_sets.items()}


@dataclass
class Rule:
    antecedent: frozenset
    consequent: frozenset
    support: float
    confidence: float
    lift: float
    leverage: float
    count: int
    p_value: float = float("nan")
    q_value: float = float("nan")
    fwer_p: float = float("nan")
    verdict: str = "untested"

    def label(self, names) -> str:
        a = ", ".join(sorted(names[i] for i in self.antecedent))
        c = ", ".join(sorted(names[i] for i in self.consequent))
        return f"{{{a}}} -> {{{c}}}"


def association_rules(itemsets, n_rows: int, min_confidence: float = 0.5,
                      min_lift: float = 1.0):
    """Generate rules from frequent itemsets. Standard definitions."""
    rules = []
    for items, support in itemsets.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in combinations(sorted(items), r):
                ante = frozenset(ante)
                cons = items - ante
                sa = itemsets.get(ante)
                sc = itemsets.get(cons)
                if not sa or not sc:
                    continue
                conf = support / sa
                lift = conf / sc
                if conf < min_confidence or lift < min_lift:
                    continue
                rules.append(
                    Rule(
                        antecedent=ante,
                        consequent=cons,
                        support=support,
                        confidence=conf,
                        lift=lift,
                        leverage=support - sa * sc,
                        count=int(round(support * n_rows)),
                    )
                )
    return rules


def mine(df: pd.DataFrame, min_support=0.05, min_confidence=0.5, min_lift=1.0,
         max_len=3, max_cardinality=20):
    """Convenience wrapper: DataFrame in, (rules, item_names) out."""
    matrix, names = one_hot(df, max_cardinality=max_cardinality)
    sets = frequent_itemsets(matrix, min_support, max_len=max_len)
    rules = association_rules(sets, len(df), min_confidence, min_lift)
    return rules, names


# ------------------------------------------------- null calibration


def target_items(names: list[str], target: str) -> set[int]:
    """Indices of every item that encodes a value of the `target` column."""
    prefix = f"{target}="
    return {i for i, n in enumerate(names) if n.startswith(prefix)}


def filter_by_target(rules: list[Rule], names: list[str],
                     target: str | None) -> list[Rule]:
    """
    Keep only rules that PREDICT the target column: the consequent is a
    single target item and the antecedent does not mention the target.

    Focusing on one question shrinks the hypothesis family from thousands of
    arbitrary column pairs to the handful you actually care about, which makes
    every correction downstream far less punishing. Pass target=None to keep
    everything.
    """
    if target is None:
        return list(rules)
    items = target_items(names, target)
    if not items:
        raise ValueError(f"target column {target!r} produced no items")
    return [
        r for r in rules
        if len(r.consequent) == 1 and r.consequent <= items
        and not (r.antecedent & items)
    ]


def ghost_lift_distribution(null_model, n_replicates=100, seed=0,
                            target: str | None = None, **mine_kwargs):
    """
    Run the identical mining pipeline on null replicates.

    If `target` is given, null rules are filtered exactly like the real ones
    (see filter_by_target), so the ghost distribution answers the same
    question the real rules do.

    Returns:
      all_lifts : every lift value seen across all replicates (for per-rule
                  p-values, controlling FDR)
      max_lifts : the single highest lift in each replicate (for family-wise
                  p-values -- the strict test)
      counts    : how many rules each replicate produced (the GHOST RATE)
    """
    all_lifts, max_lifts, counts = [], [], []
    for rep in null_model.replicates(n_replicates, seed=seed):
        rules, names = mine(rep, **mine_kwargs)
        rules = filter_by_target(rules, names, target)
        lifts = [r.lift for r in rules]
        counts.append(len(rules))
        all_lifts.extend(lifts)
        max_lifts.append(max(lifts) if lifts else 0.0)
    return np.array(all_lifts), np.array(max_lifts), np.array(counts)


def calibrate(rules, all_null_lifts, max_null_lifts, alpha=0.05):
    """
    Score every real rule against the ghost distribution.

    Two tests, deliberately:

      p_value / q_value  -- per-rule. How often does a null rule reach this
                            lift? Corrected across all rules with
                            Benjamini-Hochberg to control false discovery
                            rate. This is the practical test.

      fwer_p             -- family-wise. How often does the BEST rule in an
                            entire null run reach this lift? Surviving this
                            means the rule beats the best ghost the pipeline
                            can produce. This is the strict test.

    A rule that passes FWER is a headline result. One that passes only FDR is
    worth reporting with a caveat. One that fails both is a ghost.
    """
    if len(rules) == 0:
        return rules

    check_replicates(len(max_null_lifts), alpha=alpha, context="calibrate")

    n_all = max(len(all_null_lifts), 1)
    n_max = max(len(max_null_lifts), 1)

    raw_p = []
    for rule in rules:
        # +1 correction: never report p = 0 from a finite null sample.
        p = (np.sum(all_null_lifts >= rule.lift) + 1) / (n_all + 1)
        f = (np.sum(max_null_lifts >= rule.lift) + 1) / (n_max + 1)
        rule.p_value = float(p)
        rule.fwer_p = float(f)
        raw_p.append(p)

    q, rejected = benjamini_hochberg(np.array(raw_p), alpha=alpha)
    for rule, qv, rej in zip(rules, q, rejected):
        rule.q_value = float(qv)
        if rule.fwer_p <= alpha:
            rule.verdict = "REAL (family-wise)"
        elif rej:
            rule.verdict = "likely real (FDR)"
        else:
            rule.verdict = "GHOST"
    return rules


def remove_redundant(rules, z: float = 1.96, tolerance: float | None = None):
    """
    Drop rules whose antecedent contains useless padding.

    If {rain} -> {delay} has lift 2.70, then {rain, noise07=v2} -> {delay}
    exists too, and so do a hundred other variants with an irrelevant
    condition bolted on. They are all statistically significant. They are all
    the SAME finding. This is the second big ghost source and p-values do not
    touch it -- every padded rule is genuinely non-random.

    THE SUBTLE PART, and the reason a fixed tolerance fails:

    Adding a condition subsamples the data. Lift estimated on a smaller sample
    is noisier, so the MAXIMUM lift over many random paddings sits noticeably
    ABOVE the parent rule's lift. Padded rules do not merely tie the parent --
    they routinely beat it by chance. A fixed tolerance like 0.02 lets every
    one of them through.

    So the threshold has to scale with sampling error. The relative standard
    error of a confidence estimated from m supporting rows is roughly
    1/sqrt(m), giving

        margin = z * lift_parent / sqrt(count_child)

    A child rule is kept only if it beats its parent by more than that margin
    -- that is, only if the extra condition buys real predictive power rather
    than sampling noise. Pass an explicit `tolerance` to override with a fixed
    value if you want to show the comparison in your report.

    Run this BEFORE calibrate(), so multiple-comparison correction is applied
    to a sane number of genuinely distinct hypotheses.
    """
    by_consequent = {}
    for r in rules:
        by_consequent.setdefault(r.consequent, []).append(r)

    kept = []
    for _cons, group in by_consequent.items():
        group = sorted(group, key=lambda r: len(r.antecedent))
        for r in group:
            redundant = False
            for s in group:
                if not (s.antecedent < r.antecedent):  # strict subset only
                    continue
                if tolerance is not None:
                    margin = tolerance
                else:
                    margin = z * s.lift / np.sqrt(max(r.count, 1))
                if r.lift <= s.lift + margin:
                    redundant = True
                    break
            if not redundant:
                kept.append(r)
    return kept


def rules_to_frame(rules, names) -> pd.DataFrame:
    """Sorted, readable results table."""
    if not rules:
        return pd.DataFrame(
            columns=["rule", "support", "confidence", "lift", "p_value",
                     "q_value", "fwer_p", "verdict", "leverage", "count"]
        )
    rows = [
        {
            "rule": r.label(names),
            "support": round(r.support, 4),
            "confidence": round(r.confidence, 4),
            "lift": round(r.lift, 4),
            "p_value": round(r.p_value, 5),
            "q_value": round(r.q_value, 5),
            "fwer_p": round(r.fwer_p, 5),
            "verdict": r.verdict,
            "leverage": round(r.leverage, 5),
            "count": r.count,
        }
        for r in rules
    ]
    return pd.DataFrame(rows).sort_values(
        ["lift"], ascending=False
    ).reset_index(drop=True)
