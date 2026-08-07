import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from config import OUTPUTS_DIR

CSV_PATH = OUTPUTS_DIR / "sensitivity_results_v2.csv"
OUT = OUTPUTS_DIR


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def load_rows():
    if not CSV_PATH.exists():
        sys.exit(
            f"ERROR: {CSV_PATH} not found.\n"
            f"Run the real sensitivity sweep first:\n"
            f"  python _sensitivity_analysis.py\n"
            f"This script only renders figures from that output — it does not "
            f"contain any hardcoded/placeholder data."
        )
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def build_series(rows, analysis, x_field_map, metric_cols):
    """
    rows for a given `analysis` bucket -> (x_values, {metric_label: [values]}),
    in the order rows appear in the CSV (i.e. the order _sensitivity_analysis.py
    swept the parameter values in).
    """
    subset = [r for r in rows if r["analysis"] == analysis]
    if not subset:
        return [], {}
    x_values = [x_field_map(r) for r in subset]
    series = {}
    for col, label in metric_cols.items():
        vals = [_to_float(r.get(col)) for r in subset]
        if any(v is not None for v in vals):
            series[label] = vals
    return x_values, series


def main():
    rows = load_rows()

    delta, dda = build_series(
        rows, "dda_margin", lambda r: _to_float(r["value"]),
        {"frr": "FRR", "relevance": "Rel.", "faithfulness": "Faith.", "coverage": "Cov."},
    )
    alpha, retrieval = build_series(
        rows, "hybrid_alpha", lambda r: _to_float(r["value"]),
        {"hit_at_5": "Hit@5", "ctx_rel": "CtxRel", "relevance": "Rel.", "faithfulness": "Faith.", "frr": "FRR"},
    )
    lam, sr = build_series(
        rows, "sr_lambda", lambda r: _to_float(r["value"]),
        {"sr_comp": "SRComp", "ps_relevance": "SQ Rel.", "ps_faithfulness": "SQ Faith.",
         "ps_coverage": "SQ Cov.", "rank_stability": "Rank"},
    )

    panels = [
        ("DDA margin", r"$\delta_{\mathrm{DDA}}$", delta, dda, None),
        ("Hybrid retrieval weight", r"$\alpha$", alpha, retrieval, None),
        ("SR risk aversion", r"$\lambda_{\mathrm{SR}}$", lam, sr, None),
    ]
    missing = [title for title, _, x, series, _ in panels if not x or not series]
    if missing:
        print(f"WARNING: no data for panel(s): {missing} — check {CSV_PATH} has rows "
              f"for the corresponding 'analysis' value. Skipping empty panels.")

    # ── Figure 1: raw values ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, (title, xlabel, x, series, xticklabels) in zip(axes, panels):
        if not x or not series:
            ax.set_visible(False)
            continue
        for label, vals in series.items():
            xs = [xv for xv, v in zip(x, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        style_axis(ax, title, xlabel, "Score")
        if xticklabels:
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Sensitivity Analysis Across Neighboring Parameter Values", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0.18, 1, 0.90])

    png, pdf = OUT / "sensitivity_analysis.png", OUT / "sensitivity_analysis.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}\nSaved {pdf}")

    # ── Figure 2: deviation from default (middle sweep value) ──────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, (title, xlabel, x, series, xticklabels) in zip(axes2, panels):
        if not x or not series:
            ax.set_visible(False)
            continue
        default_index = len(x) // 2   # sweeps are 3 values around the default; middle = default
        all_diffs = []
        for label, vals in series.items():
            if vals[default_index] is None:
                continue
            baseline = vals[default_index]
            diff = [(v - baseline) if v is not None else None for v in vals]
            all_diffs.extend(d for d in diff if d is not None)
            xs = [xv for xv, d in zip(x, diff) if d is not None]
            ys = [d for d in diff if d is not None]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        ax.axhline(0, color="#222222", linewidth=1.0)
        if x:
            ax.axvline(x[default_index], color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        style_axis(ax, title, xlabel, "Delta from default")
        if xticklabels:
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
        if all_diffs:
            pad = max(0.004, max(abs(v) for v in all_diffs) * 0.18)
            ax.set_ylim(min(all_diffs) - pad, max(all_diffs) + pad)

    handles2, labels2 = [], []
    for ax in axes2:
        h, l = ax.get_legend_handles_labels()
        handles2.extend(h); labels2.extend(l)
    unique2 = dict(zip(labels2, handles2))
    fig2.legend(unique2.values(), unique2.keys(), loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig2.suptitle("Sensitivity Analysis: Change from Default Parameter Values", fontsize=12, weight="bold")
    fig2.tight_layout(rect=[0, 0.20, 1, 0.90])

    dpng, dpdf = OUT / "sensitivity_deviation_from_default.png", OUT / "sensitivity_deviation_from_default.pdf"
    fig2.savefig(dpng, dpi=220, bbox_inches="tight")
    fig2.savefig(dpdf, bbox_inches="tight")
    print(f"Saved {dpng}\nSaved {dpdf}")

    print("\nNOTE: single-column figure omitted from this real-data version — "
          "it was purely a re-plot of the same hardcoded series as figs 1-2. "
          "Add it back deliberately (selecting real columns) if you need that layout.")


if __name__ == "__main__":
    main()
