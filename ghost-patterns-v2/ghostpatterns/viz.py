"""
Figures for the report.

Five plots, each making one argument. All save to PNG at 200 dpi with a
consistent style, so they drop straight into a paper or slide deck.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REAL = "#1a5f7a"
GHOST = "#c44536"
NEUTRAL = "#8d99ae"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


# --------------------------------------------------------------- figures


def plot_lift_distribution(real_lifts, null_lifts, path="figures/lift_null.png",
                           title="Real findings against the ghost distribution"):
    """
    THE core figure. Null lift distribution with real rules overlaid.

    Whatever sits inside the grey mass was always going to be found. Only the
    right tail beyond it is a result.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if len(null_lifts) == 0:
        ax.text(0.5, 0.5,
                "The null produced no rules at all at these thresholds.\n"
                "That is itself the result: every real finding clears a bar\n"
                "that noise never reaches. Loosen min_support/min_lift to\n"
                "see the ghost distribution.",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
    else:
        ax.hist(null_lifts, bins=60, color=NEUTRAL, alpha=0.65,
                label=f"null rules (n={len(null_lifts)})", density=True)

    if len(null_lifts):
        p95 = float(np.percentile(null_lifts, 95))
        ax.axvline(p95, color=GHOST, ls="--", lw=1.6,
                   label=f"null 95th pct = {p95:.2f}")
        ax.axvline(float(np.max(null_lifts)), color="black", ls=":", lw=1.4,
                   label=f"strongest ghost = {np.max(null_lifts):.2f}")

    for i, lift in enumerate(sorted(real_lifts, reverse=True)[:25]):
        ax.axvline(lift, color=REAL, lw=1.4, alpha=0.85,
                   label="real findings" if i == 0 else None)

    _style(ax, title, "lift", "density")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_ghost_rate(naive_counts, survivor_counts, labels,
                    path="figures/ghost_rate.png"):
    """Before/after bars: how much of each haul survived validation."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    w = 0.38

    ax.bar(x - w / 2, naive_counts, w, color=NEUTRAL, label="naive mining output")
    ax.bar(x + w / 2, survivor_counts, w, color=REAL, label="survived validation")

    for i, (nv, sv) in enumerate(zip(naive_counts, survivor_counts)):
        pct = f"{sv / nv:.0%}" if nv else "n/a"
        ax.text(i + w / 2, sv, f" {sv}\n ({pct})", ha="center", va="bottom",
                fontsize=9)
        ax.text(i - w / 2, nv, f"{nv} ", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style(ax, "How many findings survive calibration", "", "rules")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_power_curve(power_df, path="figures/power_curve.png"):
    """
    Sensitivity. The 80% line is the minimum detectable effect -- quote it
    with every negative result.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(power_df.effect_size, power_df.power, "o-", color=REAL, lw=2,
            markersize=7, label="recovery rate")
    ax.axhline(0.8, color=GHOST, ls="--", lw=1.4, label="80% power")

    mde = power_df.attrs.get("min_detectable_effect")
    if mde is not None:
        ax.axvline(mde, color="black", ls=":", lw=1.4,
                   label=f"min detectable effect = {mde}")

    ax.set_ylim(-0.03, 1.05)
    _style(ax, "Validator sensitivity", "planted effect size",
           "fraction recovered")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_dimensionality_curse(curve_df, path="figures/dimensionality.png"):
    """
    Relative contrast collapsing while the detector keeps flagging its quota.
    Every flagged point is a ghost by construction.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(curve_df.dimensions, curve_df.relative_contrast, "o-",
             color=REAL, lw=2, markersize=6)
    ax1.set_yscale("log")
    ax1.axhline(0.3, color=GHOST, ls="--", lw=1.3)
    ax1.text(curve_df.dimensions.iloc[-1], 0.32,
             "distances collapsed", ha="right", fontsize=8.5, color=GHOST)
    ax1.set_xscale("log")
    _style(ax1, "Anomaly detection degrades with dimensionality",
           "dimensions", "relative contrast (log)")
    ax1.yaxis.label.set_color(REAL)
    ax1.tick_params(axis="y", colors=REAL)

    ax2 = ax1.twinx()
    ax2.bar(curve_df.dimensions, curve_df.flag_rate, alpha=0.22,
            color=GHOST, width=curve_df.dimensions * 0.5)
    ax2.set_ylabel("fraction flagged (all ghosts)", fontsize=10, color=GHOST)
    ax2.tick_params(axis="y", colors=GHOST)
    ax2.set_ylim(0, max(curve_df.flag_rate) * 2.2)
    ax2.spines[["top"]].set_visible(False)

    return _save(fig, path)


def plot_silhouette_vs_null(sil_df, path="figures/silhouette.png"):
    """
    Real silhouette against the null's. The gap between the lines is the only
    part that is evidence; the height of either line is not.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(sil_df.k, sil_df.silhouette, "o-", color=REAL, lw=2,
            markersize=7, label="observed")
    ax.plot(sil_df.k, sil_df.null_mean, "s--", color=NEUTRAL, lw=1.8,
            markersize=6, label="null (structureless)")
    ax.fill_between(
        sil_df.k,
        sil_df.null_mean - 2 * sil_df.null_std,
        sil_df.null_mean + 2 * sil_df.null_std,
        color=NEUTRAL, alpha=0.25, label="null +/- 2 sd",
    )

    for _, r in sil_df[sil_df.beats_noise].iterrows():
        ax.annotate("beats null", (r.k, r.silhouette),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5, color=REAL)

    _style(ax, "Cluster separation: observed vs what noise produces",
           "k", "silhouette")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_top_rules(table: pd.DataFrame, path="figures/top_rules.png", top=15,
                   null_p95: float | None = None, null_max: float | None = None):
    """
    Horizontal bars for the strongest validated rules, coloured by verdict.

    The reference lines show where noise stops. A bar that ends left of the
    strongest ghost would not have survived; the visual gap is the evidence.
    """
    shown = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.38 * len(shown) + 1.2)))

    if shown.empty:
        ax.text(0.5, 0.5, "No rule survived validation.", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_yticks([])
    else:
        colors = [REAL if v == "REAL (family-wise)" else NEUTRAL
                  for v in shown.verdict]
        y = np.arange(len(shown))
        ax.barh(y, shown.lift, color=colors)
        labels = [r if len(r) <= 70 else r[:67] + "..." for r in shown.rule]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for yi, (lift, conf) in enumerate(zip(shown.lift, shown.confidence)):
            ax.text(lift, yi, f" {lift:.2f} (conf {conf:.0%})", va="center",
                    fontsize=7.5)
        ax.set_xlim(0, shown.lift.max() * 1.25)

    if null_p95 is not None:
        ax.axvline(null_p95, color=GHOST, ls="--", lw=1.3,
                   label=f"null 95th pct = {null_p95:.2f}")
    if null_max is not None:
        ax.axvline(null_max, color="black", ls=":", lw=1.3,
                   label=f"strongest ghost = {null_max:.2f}")

    _style(ax, f"Top {len(shown)} validated rules", "lift", "")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    if null_p95 is not None or null_max is not None:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_column_attribution(attr_df: pd.DataFrame,
                            path="figures/column_attribution.png"):
    """
    How much structure disappears when each column is shuffled. Columns with
    no drop are decoration and can be removed from the analysis.
    """
    df = attr_df.sort_values("drop")
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(df) + 1.2)))
    colors = [REAL if lb else NEUTRAL for lb in df.load_bearing]
    ax.barh(np.arange(len(df)), df["drop"], color=colors)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.column, fontsize=8.5)
    _style(ax, "Which columns carry the structure",
           "drop in strong-rule count when column is shuffled", "")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.bar(0, 0, color=REAL, label="load-bearing (p <= 0.05)")
    ax.bar(0, 0, color=NEUTRAL, label="not significant")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_anomaly_scores(result, path="figures/anomaly_scores.png"):
    """
    Real anomaly-score distribution over the null's. Only mass to the right
    of the threshold, in EXCESS of the null's, is evidence of anomalies.
    """
    if result.scores is None or result.null_scores is None:
        raise ValueError("result carries no scores; rerun anomalies_vs_null")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo = min(result.scores.min(), result.null_scores.min())
    hi = max(result.scores.max(), result.null_scores.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(result.null_scores, bins=bins, density=True, color=NEUTRAL,
            alpha=0.6, label="null (structureless)")
    ax.hist(result.scores, bins=bins, density=True, histtype="step",
            color=REAL, lw=1.8, label="observed")
    ax.axvline(result.threshold, color=GHOST, ls="--", lw=1.4,
               label=f"null 99th pct ({result.n_flagged} points above)")
    verdict = "real excess" if result.real else "no excess"
    _style(ax, f"{result.method}: {verdict} (p={result.p_value:.3f})",
           "anomaly score (higher = stranger)", "density")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def generate_all(outdir="figures", quick=True):
    """
    Build the full figure set from live computation -- nothing hardcoded.
    """
    from . import clusters, rules
    from .anomaly import dimensionality_curse_curve
    from .benchmark import power_curve as bench_power
    from .nullmodels import IndependentShuffle

    print(f"Generating figures into {outdir}/ ...")
    paths = []
    rng = np.random.default_rng(7)
    n = 1200

    # --- a dataset with one planted rule buried in noise
    df = pd.DataFrame({"weather": rng.choice(["sun", "rain", "cloud"], n,
                                             p=[0.5, 0.25, 0.25])})
    df["delay"] = np.where(
        np.where(df.weather == "rain", rng.random(n) < 0.8, rng.random(n) < 0.15),
        "yes", "no")
    df["promo"] = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    df["weekend"] = rng.choice(["yes", "no"], n, p=[0.28, 0.72])
    combo = (df.promo == "yes") & (df.weekend == "yes")
    df["basket"] = np.where(combo,
                            np.where(rng.random(n) < 0.75, "large", "small"),
                            np.where(rng.random(n) < 0.20, "large", "small"))
    for i in range(18):
        k = rng.integers(2, 5)
        df[f"noise{i:02d}"] = rng.choice([f"v{j}" for j in range(k)], n)

    # max_len=3 with many columns is what actually generates ghosts -- the
    # whole point of the figure is to show that distribution.
    params = dict(min_support=0.04, min_confidence=0.55, min_lift=1.2, max_len=3)
    found, names = rules.mine(df, **params)
    minimal = rules.remove_redundant(found)
    nm = IndependentShuffle(df)
    all_lifts, max_lifts, counts = rules.ghost_lift_distribution(
        nm, n_replicates=30 if quick else 100, seed=1, **params)
    rules.calibrate(minimal, all_lifts, max_lifts)
    table = rules.rules_to_frame(minimal, names)
    survivors = table[table.verdict != "GHOST"]

    paths.append(plot_lift_distribution(
        survivors.lift.tolist(), all_lifts, f"{outdir}/lift_null.png"))
    paths.append(plot_ghost_rate(
        [len(found), int(counts.mean())], [len(survivors), 0],
        ["real data", "structureless copy"], f"{outdir}/ghost_rate.png"))

    # --- dimensionality
    paths.append(plot_dimensionality_curse(
        dimensionality_curse_curve(n=400), f"{outdir}/dimensionality.png"))

    # --- clustering on noise
    noise = pd.DataFrame(rng.normal(size=(500, 6)),
                         columns=[f"f{i}" for i in range(6)])
    sil = clusters.silhouette_vs_null(noise, k_range=range(2, 8),
                                      n_null=12 if quick else 25, seed=2)
    paths.append(plot_silhouette_vs_null(sil, f"{outdir}/silhouette.png"))

    # --- power curve
    pw = bench_power(effects=(0.1, 0.2, 0.3, 0.5),
                     n_trials=4 if quick else 15,
                     n_null=24 if quick else 40, verbose=False)
    paths.append(plot_power_curve(pw, f"{outdir}/power_curve.png"))

    print(f"Done: {len(paths)} figures.")
    return paths
