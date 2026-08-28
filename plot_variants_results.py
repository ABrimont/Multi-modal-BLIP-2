#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Academic Grid Search Figures — One per metric.

Reads all results.csv files under <root> (one subfolder per variant) and generates:
  - A separate figure per metric (BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE)
  - Figures for raw sum and normalized sum
  - Mean ± standard deviation plotted on each variant
  - Sensitivity and coefficient of variation figures
  - Length penalty x repetition penalty heatmaps per metric
  - Console prints and CSV exports:
        * Mean ± standard deviation table (variant × metric)
        * Best configuration of each variant with all its metrics

All figures are saved via save_fig() in high quality.
Configurable format and resolution: --ext {jpg,png,pdf,tiff} --dpi 400 --quality 95

Usage:
    python plot_grid.py
    python plot_grid.py --root hyper_output/grid_msrvtt_variants --outdir figures
    python plot_grid.py --ext pdf              # vector format (publication)
    python plot_grid.py --ext png --dpi 600       # lossless raster, high sharpness
    python plot_grid.py --order alpha --metrics cider meteor
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
METRIC_COLS = ["bleu4", "meteor", "rouge", "cider", "spice"]
METRIC_LABELS = {
    "bleu4": "BLEU-4", "meteor": "METEOR", "rouge": "ROUGE-L",
    "cider": "CIDEr", "spice": "SPICE",
    "sum_raw": "Raw sum", "sum_norm": "Normalized sum",
}
PARAM_COLS = ["variant", "batch", "beam", "lp", "rp", "min", "max"]

# Okabe-Ito colorblind-safe palette
TOL = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442", "#000000"]

INK   = "#1A1A1A"
MUTED = "#555555"
GRID  = "#D7D7D7"

SERIF = "DejaVu Serif"

# Save quality defaults (overridden by CLI arguments)
SAVE_DPI     = 400
JPEG_QUALITY = 95


# -----------------------------------------------------------------------------
# Color Utilities
# -----------------------------------------------------------------------------
def _mix(c, o, t):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(o))
    return tuple((1 - t) * a + t * b)

def lighten(c, t=0.6):
    return _mix(c, "white", t)

def darken(c, t=0.25):
    return _mix(c, "black", t)


# -----------------------------------------------------------------------------
# Centralized Saving Function
# -----------------------------------------------------------------------------
def save_fig(fig, outpath, dpi=None):
    """Save and close the figure. JPEG uses maximum quality + white background."""
    dpi = dpi or SAVE_DPI
    ext = os.path.splitext(outpath)[1].lower()
    kw = dict(dpi=dpi, bbox_inches="tight")

    if ext in (".jpg", ".jpeg"):
        kw["facecolor"] = "white"
        kw["pil_kwargs"] = {"quality": JPEG_QUALITY, "optimize": True, "progressive": True}
    elif ext in (".tif", ".tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}

    fig.savefig(outpath, **kw)
    plt.close(fig)
    print(f"  → {outpath}")


# -----------------------------------------------------------------------------
# Academic Style Setup
# -----------------------------------------------------------------------------
def setup_style(fontdir="fonts"):
    global SERIF
    if os.path.isdir(fontdir):
        for ttf in glob.glob(os.path.join(fontdir, "*.ttf")):
            try:
                fm.fontManager.addfont(ttf)
            except Exception:
                pass
    avail = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Times New Roman", "Tinos", "Liberation Serif",
                 "Nimbus Roman", "STIX Two Text", "DejaVu Serif"]:
        if cand in avail:
            SERIF = cand
            break
    plt.rcParams.update({
        "font.family":      SERIF,
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "text.color":       INK,
        "axes.edgecolor":   INK,
        "axes.linewidth":   1.0,
        "axes.labelcolor":  INK,
        "xtick.color":      INK,
        "ytick.color":      INK,
        "axes.labelsize":   14,
        "xtick.labelsize":  13,
        "ytick.labelsize":  12,
        "axes.titlesize":   16,
        "figure.dpi":       120,
    })
    print(f"  font: {SERIF}")


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_all(root):
    paths = sorted(glob.glob(os.path.join(root, "*", "results.csv")))
    if not paths and os.path.isfile(os.path.join(root, "results.csv")):
        paths = [os.path.join(root, "results.csv")]
    if not paths:
        raise SystemExit(f"No results.csv found under: {root}")

    frames = []
    for p in paths:
        variant = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        for m in METRIC_COLS:
            df[m] = pd.to_numeric(df[m], errors="coerce") if m in df.columns else np.nan
        df["variant"] = variant
        df = df.dropna(subset=METRIC_COLS, how="all")
        print(f"  [{variant:<14}] {len(df)} valid runs")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_sums(data, metrics):
    d = data.copy()
    d["sum_raw"] = d[metrics].sum(axis=1, skipna=False)

    norm = d[metrics].copy()
    for m in metrics:
        lo, hi = d[m].min(), d[m].max()
        norm[m] = 0.0 if hi == lo else (d[m] - lo) / (hi - lo)

    d["sum_norm"] = norm.sum(axis=1, skipna=False)
    return d


# -----------------------------------------------------------------------------
# Variant Ordering
# -----------------------------------------------------------------------------
TARGET_ORDER = [
    "Aligned_6",
    "Aligned_12",
    "Alternating",
    "Increased",
    "Random_initialization",
    "Separate",
    "unimodal"
]

def order_variants(data, value_col, how="median"):
    available_variants = data["variant"].unique()
    ordered = [v for v in TARGET_ORDER if v in available_variants]
    ordered += [v for v in available_variants if v not in ordered]
    return ordered


# -----------------------------------------------------------------------------
# Figure Generation per Metric / Sum
# -----------------------------------------------------------------------------
def make_figure(data, value_col, variants, cmap, outpath, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=(1.55 * len(variants) + 1.8, 4.9))

    for i, v in enumerate(variants):
        vals = np.sort(data.loc[data.variant == v, value_col].dropna().values)[::-1]
        if len(vals) == 0:
            continue
        base = cmap[v]

        ax.boxplot(
            vals, positions=[i], widths=0.46, patch_artist=True,
            showcaps=True, showfliers=False, whis=1.5, zorder=2,
            medianprops=dict(color=INK, linewidth=1.8),
            whiskerprops=dict(color=INK, linewidth=1.0),
            capprops=dict(color=INK, linewidth=1.0),
            boxprops=dict(facecolor=lighten(base, 0.62),
                          edgecolor=darken(base, 0.1), linewidth=1.2),
        )
        rng = np.random.default_rng(i)
        jit = rng.uniform(-0.085, 0.085, len(vals)) - 0.13
        ax.scatter(np.full(len(vals), i) + jit, vals, s=13,
                   color=darken(base, 0.05), alpha=0.4,
                   edgecolor="none", zorder=3)

        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.errorbar(i + 0.27, mu, yerr=sd, fmt="o", ms=5.5,
                    color=INK, ecolor=INK, elinewidth=1.1,
                    capsize=3.5, capthick=1.1, zorder=4)

    ax.set_xticks(range(len(variants)))
    clean_variants = [v[2:] if v.startswith("0_") else v for v in variants]
    ax.set_xticklabels(clean_variants, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(variants) - 0.45 + 0.3)
    ax.set_ylabel(ylabel or METRIC_LABELS.get(value_col, value_col))
    ax.set_title(title or METRIC_LABELS.get(value_col, value_col),
                 fontweight="bold", pad=14)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=4, width=1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.05)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor=INK, markersize=6, label="mean ± s.d."),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.015),
              frameon=False, fontsize=11, handletextpad=0.4, borderaxespad=0.0)

    fig.tight_layout()
    save_fig(fig, outpath)


# -----------------------------------------------------------------------------
# Reporting & CSV Export
# -----------------------------------------------------------------------------
def report(data, metrics, outdir):
    cols = metrics + ["sum_raw", "sum_norm"]

    g = data.groupby("variant")[cols]
    mean, std = g.mean(), g.std(ddof=1)
    rows = []
    for v in mean.index:
        rows.append({"variant": v,
                     **{METRIC_LABELS.get(c, c): f"{mean.loc[v, c]:.3f} ± {std.loc[v, c]:.3f}"
                        for c in cols}})
    tbl = pd.DataFrame(rows).set_index("variant")
    tbl = tbl.loc[mean["sum_norm"].sort_values(ascending=False).index]

    print("\n" + "=" * 78)
    print("MEAN ± STD  (over the whole grid, per variant)")
    print("=" * 78)
    print(tbl.to_string())

    flat = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    flat = flat.loc[mean["sum_norm"].sort_values(ascending=False).index]
    flat.to_csv(os.path.join(outdir, "stats_mean_std.csv"))
    print(f"\n  → {os.path.join(outdir, 'stats_mean_std.csv')}")

    best_rows = []
    for v in data["variant"].unique():
        sub = data[data.variant == v].dropna(subset=["cider"])
        if len(sub):
            best_rows.append(sub.loc[sub["cider"].idxmax()])
    best = pd.DataFrame(best_rows).sort_values("cider", ascending=False)

    print("\n" + "=" * 78)
    print("BEST CONFIG PER VARIANT  (argmax CIDEr)")
    print("=" * 78)
    show = [c for c in PARAM_COLS if c in best.columns] + cols
    disp = best[show].copy()
    for c in cols:
        disp[c] = disp[c].map(lambda x: f"{x:.3f}")
    disp = disp.rename(columns=METRIC_LABELS).reset_index(drop=True)
    print(disp.to_string(index=False))

    best[show].to_csv(os.path.join(outdir, "best_configs.csv"), index=False)
    print(f"\n  → {os.path.join(outdir, 'best_configs.csv')}")


# -----------------------------------------------------------------------------
# Sensitivity & CV Analysis
# -----------------------------------------------------------------------------
def sensitivity_frame(data, metrics, variant=None):
    d = data if variant is None else data[data.variant == variant]
    rows = []
    for _, sub in d.groupby("variant"):
        for m in metrics:
            col = sub[m].dropna()
            if len(col) == 0 or col.mean() == 0:
                continue
            mu = col.mean()
            for x in col:
                rows.append({"metric": METRIC_LABELS[m],
                             "pct": 100.0 * (x - mu) / mu})
    return pd.DataFrame(rows)


def cv_table(data, metrics):
    cv = data.groupby("variant")[metrics].agg(
        lambda c: c.std(ddof=1) / c.mean() * 100.0)
    summary = pd.DataFrame({
        "CV_mean_%": cv.mean(),
        "CV_std_%":  cv.std(ddof=1),
    })
    summary.index = [METRIC_LABELS.get(m, m) for m in summary.index]
    return cv, summary.sort_values("CV_mean_%", ascending=False)


def make_sensitivity_figure(data, metrics, outpath, variant=None):
    df = sensitivity_frame(data, metrics, variant)
    if df.empty:
        return
    iqr = df.groupby("metric")["pct"].agg(lambda s: s.quantile(.75) - s.quantile(.25))
    order = list(iqr.sort_values(ascending=False).index)
    cmap = {lbl: TOL[i % len(TOL)] for i, lbl in enumerate(order)}

    fig, ax = plt.subplots(figsize=(1.35 * len(order) + 1.8, 4.9))
    ax.axhline(0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)

    for i, lbl in enumerate(order):
        vals = df.loc[df.metric == lbl, "pct"].values
        base = cmap[lbl]
        ax.boxplot(
            vals, positions=[i], widths=0.5, patch_artist=True,
            showcaps=True, showfliers=False, whis=1.5, zorder=2,
            medianprops=dict(color=INK, linewidth=1.8),
            whiskerprops=dict(color=INK, linewidth=1.0),
            capprops=dict(color=INK, linewidth=1.0),
            boxprops=dict(facecolor=lighten(base, 0.62),
                          edgecolor=darken(base, 0.1), linewidth=1.2),
        )
        rng = np.random.default_rng(i)
        jit = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=12,
                   color=darken(base, 0.05), alpha=0.35,
                   edgecolor="none", zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("deviation from mean (%)")
    ttl = "Metric sensitivity to hyperparameters"
    ttl += f" — {variant}" if variant else ""
    ax.set_title(ttl, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=4, width=1.0)
    fig.tight_layout()
    save_fig(fig, outpath)


def make_cv_figure(summary, outpath):
    fig, ax = plt.subplots(figsize=(1.25 * len(summary) + 1.6, 4.6))
    labels = list(summary.index)
    means = summary["CV_mean_%"].values
    errs = summary["CV_std_%"].fillna(0).values
    colors = [lighten(TOL[i % len(TOL)], 0.35) for i in range(len(labels))]
    ax.bar(range(len(labels)), means, width=0.62, color=colors,
           edgecolor=[darken(TOL[i % len(TOL)], 0.1) for i in range(len(labels))],
           linewidth=1.2, zorder=2)
    ax.errorbar(range(len(labels)), means, yerr=errs, fmt="none",
                ecolor=INK, elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=3)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.annotate(f"{m:.3f}", (i, m + e), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", fontsize=11, color=INK)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("coefficient of variation (%)")
    ax.set_title("Metric variability across the hyperparameter grid",
                 fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=4, width=1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.1)
    fig.tight_layout()
    save_fig(fig, outpath)


# -----------------------------------------------------------------------------
# Heatmaps lp × rp
# -----------------------------------------------------------------------------
def _text_color(rgb):
    r, g, b = rgb[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1A1A1A" if lum > 0.6 else "white"

def _decimals_for(span):
    if not np.isfinite(span) or span <= 0:
        return 2
    d = int(np.ceil(-np.log10(span))) + 1
    return int(np.clip(d, 1, 4))


def make_heatmaps(data, metrics, outpath, center=False):
    if "lp" not in data.columns or "rp" not in data.columns:
        print("  [heatmap] missing lp/rp columns")
        return

    n = len(metrics)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, metrics):
        piv = data.pivot_table(index="lp", columns="rp", values=m, aggfunc="mean")
        piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)
        M = piv.values.astype(float)
        base = TOL[metrics.index(m) % len(TOL)]

        if center:
            mu = np.nanmean(M)
            M = 100.0 * (M - mu) / mu
            cmap = LinearSegmentedColormap.from_list(
                "div", [darken(base, 0.3), "#F7F7F7", darken(base, 0.05)])
            vmax = np.nanmax(np.abs(M))
            vmin = -vmax
            dec = _decimals_for((np.nanmax(M) - np.nanmin(M)) / 6 or 1)
            fmt = lambda x, d=dec: f"{x:+.{d}f}"
        else:
            cmap = LinearSegmentedColormap.from_list(
                "seq", [lighten(base, 0.85), darken(base, 0.28)])
            vmin = np.nanmin(M)
            vmax = np.nanmax(M)
            dec = _decimals_for(vmax - vmin)
            fmt = lambda x, d=dec: f"{x:.{d}f}"

        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        for x in range(1, M.shape[1]):
            ax.axvline(x - 0.5, color="white", linewidth=1.6)
        for y in range(1, M.shape[0]):
            ax.axhline(y - 0.5, color="white", linewidth=1.6)

        raw = data.pivot_table(index="lp", columns="rp", values=m, aggfunc="mean")
        raw = raw.sort_index().reindex(sorted(raw.columns), axis=1).values
        bi, bj = np.unravel_index(np.nanargmax(raw), raw.shape)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                rgba = im.cmap(im.norm(M[i, j]))
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center",
                        fontsize=9.5, color=_text_color(rgba),
                        fontweight="bold" if (i == bi and j == bj) else "normal")

        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=2.4, zorder=5))

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{x:g}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{x:g}" for x in piv.index])
        ax.set_title(METRIC_LABELS.get(m, m), fontweight="bold", fontsize=13)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.supxlabel("Repetition penalty", fontsize=15, y=0.02)
    fig.supylabel("Length penalty", fontsize=15, x=0.005)

    fig.tight_layout(rect=[0.02, 0.04, 1, 1])
    save_fig(fig, outpath)


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------
def main():
    global SAVE_DPI, JPEG_QUALITY

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="hyper_output/grid_msrvtt_variants")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--fontdir", default="fonts")
    ap.add_argument("--metrics", nargs="+", default=METRIC_COLS, choices=METRIC_COLS)
    ap.add_argument("--order", default="median", choices=["median", "alpha"])
    ap.add_argument("--ext", default="jpg",
                    choices=["jpg", "jpeg", "png", "pdf", "tif", "tiff"],
                    help="figure format (default: jpg)")
    ap.add_argument("--dpi", type=int, default=400,
                    help="raster resolution (default: 400)")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality 1-95 (default: 95)")
    ap.add_argument("--sens-variant", default=None,
                    help="restrict sensitivity analysis to a single variant")
    ap.add_argument("--heatmap-center", action="store_true",
                    help="relative percentage heatmaps instead of raw values")
    args = ap.parse_args()

    SAVE_DPI = args.dpi
    JPEG_QUALITY = args.quality
    ext = args.ext

    os.makedirs(args.outdir, exist_ok=True)
    setup_style(args.fontdir)

    print(f"\nReading directory: {args.root}")
    data = load_all(args.root)
    data = add_sums(data, args.metrics)
    allv = sorted(data["variant"].unique())
    cmap = {v: TOL[i % len(TOL)] for i, v in enumerate(allv)}
    print(f"\nVariants: {', '.join(allv)}")
    print(f"Output    : {ext.upper()} @ {args.dpi} dpi"
          + (f", quality {args.quality}" if ext in ("jpg", "jpeg") else "") + "\n")

    for m in args.metrics:
        order = order_variants(data, m, args.order)
        make_figure(data, m, order, cmap,
                    os.path.join(args.outdir, f"fig_{m}.{ext}"))

    for s in ("sum_raw", "sum_norm"):
        order = order_variants(data, s, args.order)
        yl = METRIC_LABELS[s] + (f"  (max = {len(args.metrics)})"
                                 if s == "sum_norm" else "")
        make_figure(data, s, order, cmap,
                    os.path.join(args.outdir, f"fig_{s}.{ext}"),
                    ylabel=yl, title=METRIC_LABELS[s])

    report(data, args.metrics, args.outdir)

    make_sensitivity_figure(data, args.metrics,
                            os.path.join(args.outdir, f"fig_metric_sensitivity.{ext}"),
                            variant=args.sens_variant)
    cv, summary = cv_table(data, args.metrics)
    make_cv_figure(summary, os.path.join(args.outdir, f"fig_metric_cv.{ext}"))
    summary.to_csv(os.path.join(args.outdir, "metric_cv.csv"))
    print("\n" + "=" * 78)
    print("METRIC SENSITIVITY  (coefficient of variation, % — higher = more sensitive)")
    print("=" * 78)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  → {os.path.join(args.outdir, 'metric_cv.csv')}")

    make_heatmaps(data, args.metrics,
                  os.path.join(args.outdir, f"fig_heatmaps_lp_rp.{ext}"),
                  center=args.heatmap_center)

    print("\nFinished successfully.")


if __name__ == "__main__":
    main()
