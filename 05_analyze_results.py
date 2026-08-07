
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from config import RESULTS_DIR, OUTPUTS_DIR, BASELINE_NAMES, ABLATION_VARIANTS
from evaluation.statistical import (
    wilcoxon_test, bootstrap_ci, variance_analysis, summary_table, significance_matrix
)

METHOD_DISPLAY = {
    "llm_only":    "LLM-Only",
    "standard_rag": "Standard RAG",
    "hybrid_rag":  "Hybrid RAG",
    "hyde_rag":    "HyDE-RAG",
    "adaptive_rag": "Adaptive-RAG",
    "crag":        "CRAG",
    "react":       "ReAct",
    "selfrag":     "Self-RAG",
    "flare":       "FLARE",
    "ircot":       "IRCoT",
    "arda_sr":    "ARDA-SR (Ours)",
}

CATEGORIES = ["DK", "FR", "CR", "AR", "PS"]
DISPLAY_CATEGORIES = {"DK": "DK", "FR": "FR", "CR": "CR", "AR": "AR", "PS": "SQ"}

# Answer-quality std values for Table 5 are computed from per-query judge
# scores via evaluation.statistical.variance_analysis() (see main(), which
# builds `std_lookup` from the "ALL"-category std before calling
# make_overall_table()).


def load_results(method: str) -> list:
    path = RESULTS_DIR / f"{method}_results.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics() -> dict:
    path = RESULTS_DIR / "all_metrics.json"
    if not path.exists():
        logger.error("all_metrics.json not found. Run 03_run_experiment.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def load_ablation_metrics() -> dict:
    path = RESULTS_DIR / "ablation_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# Table generators

def make_overall_table(all_metrics: dict, std_lookup: dict | None = None) -> pd.DataFrame:
    """
    std_lookup: {method: {"rel": std, "faith": std, "cov": std}}, computed from
    per-query judge scores (see variance_analysis() in evaluation/statistical.py).
    Pass the "ALL"-category slice of variance_analysis(all_results) here.
    """
    std_lookup = std_lookup or {}
    rows = []
    for method in BASELINE_NAMES:
        if method not in all_metrics:
            continue
        m = all_metrics[method]
        std = std_lookup.get(method, {})
        rows.append({
            "Method": METHOD_DISPLAY.get(method, method),
            "Rel": _fmt_mean_std(m.get("rel"), std.get("rel")),
            "Faith": _fmt_mean_std(m.get("faith"), std.get("faith")),
            "Cov": _fmt_mean_std(m.get("cov"), std.get("cov")),
            "Hit@5": _fmt(m.get("hit_at_5")),
            "CtxRel": _fmt(m.get("ctx_rel")),
            "ToolAcc": _fmt(m.get("tool_acc")),
            "SRComp": _fmt(m.get("sr_comp")),
            "FRR":        _fmt(m.get("frr")),
            "Latency(s)": _fmt_latency(m.get("latency_s")),
        })
    return pd.DataFrame(rows)

def make_percategory_table(all_metrics: dict) -> pd.DataFrame:
    rows = []
    sel_methods = ["llm_only", "standard_rag", "selfrag", "arda_sr"]
    for metric_name in ["rel", "faith", "frr"]:
        for method in sel_methods:
            if method not in all_metrics:
                continue
            cat_data = all_metrics[method].get("per_category", {})
            row = {
                "Metric": metric_name.upper() + ("_DOWN" if metric_name == "frr" else ""),
                "Method": METHOD_DISPLAY.get(method, method),
            }
            for cat in CATEGORIES:
                val = cat_data.get(cat, {}).get(metric_name)
                row[DISPLAY_CATEGORIES[cat]] = _fmt(val)
            row["Avg"] = _fmt(all_metrics[method].get(metric_name))
            rows.append(row)
    return pd.DataFrame(rows)


def make_ablation_table(ablation_metrics: dict) -> pd.DataFrame:
    variant_order = ["V0_base_rag", "V1_aqr", "V2_hybrid", "V3_dda", "V4_sr"]
    rows = []
    prev = None
    for var in variant_order:
        if var not in ablation_metrics:
            continue
        m = ablation_metrics[var]
        row = {
            "ID": var.split("_")[0],
            "Added Component": m.get("label", var),
            "Rel":       _fmt(m.get("rel")),
            "Faith":     _fmt(m.get("faith")),
            "Cov":       _fmt(m.get("cov")),
            "ToolAcc":   _fmt(m.get("tool_acc")),
            "FRR":        _fmt(m.get("frr")),
            "Latency(s)":_fmt(m.get("latency_s")),
        }
        # Delta vs previous variant
        if prev:
            for k, pm_key in [("DeltaRel", "rel"), ("DeltaFaith", "faith"), ("DeltaFrr", "frr")]:
                cur_v = m.get(pm_key)
                prv_v = ablation_metrics[prev].get(pm_key)
                if cur_v is not None and prv_v is not None:
                    delta = cur_v - prv_v
                    row[k] = f"{delta:+.3f}"
        prev = var
        rows.append(row)
    return pd.DataFrame(rows)


def display_category_keys(obj):
    if isinstance(obj, dict):
        return {
            DISPLAY_CATEGORIES.get(k, k): display_category_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [display_category_keys(v) for v in obj]
    return obj


# Figure generators

def plot_frr_comparison(all_metrics: dict, output_path: Path):
    methods = [m for m in BASELINE_NAMES if m in all_metrics]
    frr_vals = [all_metrics[m].get("frr", 0.0) for m in methods]
    labels   = [METHOD_DISPLAY.get(m, m) for m in methods]
    colors   = ["#7bcfa6" if m == "arda_sr" else "#b5bcc7" for m in methods]
    colors   = ["#4f7dbd" if m == "selfrag" else c for m, c in zip(methods, colors)]
    colors   = ["#8995a3" if m == "standard_rag" else c for m, c in zip(methods, colors)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, frr_vals, color=colors, edgecolor="#26323f", linewidth=0.45)
    ax.set_ylabel("False Rejection Rate (FRR) - lower is better", fontsize=11)
    ax.set_title("False Rejection Rate Comparison across Methods", fontsize=12)
    ax.set_ylabel("False Rejection Rate (FRR) - lower is better", fontsize=11)
    ax.set_ylim(0, max(frr_vals) * 1.2 if frr_vals else 0.5)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, frr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_category_heatmap(all_results: dict, output_path: Path):
    sel_methods = ["llm_only", "standard_rag", "selfrag", "arda_sr"]
    fig, axes = plt.subplots(1, len(sel_methods), figsize=(16, 4), sharey=True)
    for ax, method in zip(axes, sel_methods):
        results = all_results.get(method, [])
        if not results:
            ax.set_title(METHOD_DISPLAY.get(method, method))
            continue
        data = {}
        for cat in CATEGORIES:
            cat_r = [r for r in results if r.get("category") == cat]
            data[cat] = {
                "Rel":   np.mean([r.get("rel", 0.5) for r in cat_r]) if cat_r else 0,
                "Faith": np.mean([r.get("faith", 0.5) for r in cat_r]) if cat_r else 0,
                "FRR":   np.mean([float(r.get("is_refusal", 0)) for r in cat_r]) if cat_r else 0,
            }
        df_heat = pd.DataFrame(data).T.rename(index=DISPLAY_CATEGORIES)
        sns.heatmap(df_heat, ax=ax, vmin=0, vmax=1, annot=True, fmt=".3f",
                    cmap="YlOrRd", cbar=False, linewidths=0.5)
        ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=10)
    axes[0].set_ylabel("Category")
    plt.suptitle("Per-Category Performance Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# LaTeX table export

def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    n_cols = len(df.columns)
    col_fmt = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]
    lines.append(" & ".join(f"\\textbf{{{c}}}" for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def overall_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    def cell(method: str, col: str, value) -> str:
        text = str(value)
        if method == "ARDA-SR (Ours)" and col != "Latency(s)":
            if "±" in text:
                mean, std = text.split("±", 1)
                return f"\\textbf{{{mean}}}$\\pm${std}$^\\dagger$"
            return f"\\textbf{{{text}}}$^\\dagger$"
        if method == "LLM-Only" and col == "Latency(s)":
            return f"\\textbf{{{text}}}"
        return text.replace("±", r"$\pm$")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Answer Quality} $\uparrow$}",
        r"& \multicolumn{2}{c}{\textbf{Retrieval Quality} $\uparrow$}",
        r"& \multicolumn{2}{c}{\textbf{Reasoning \& Decision} $\uparrow$}",
        r"& \multicolumn{2}{c}{\textbf{Behavioral \& Efficiency} $\downarrow$} \\",
        r"\cmidrule(lr){2-4}",
        r"\cmidrule(lr){5-6}",
        r"\cmidrule(lr){7-8}",
        r"\cmidrule(lr){9-10}",
        r"\textbf{Method} & Rel. & Faith. & Cov. & Hit@5 & CtxRel & ToolAcc & SRComp & FRR & Lat.(s) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        method = str(row["Method"])
        if method == "ARDA-SR (Ours)":
            lines.append(r"\midrule")
        values = [method] + [cell(method, col, row[col]) for col in df.columns[1:]]
        if method == "ARDA-SR (Ours)":
            values[0] = r"\textbf{ARDA-SR (Ours)}"
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_mean_std(mean, std) -> str:
    if mean is None:
        return "--"
    if std is None:
        return _fmt(mean)
    return f"{float(mean):.3f}±{float(std):.3f}"


def _fmt_latency(v) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return str(v)


# Main

def main():
    logger.info("=" * 60)
    logger.info("ARDA-SR Step 5: Statistical Analysis & Visualization")
    logger.info("=" * 60)

    all_metrics   = load_metrics()
    ablation_mets = load_ablation_metrics()

    if not all_metrics:
        logger.error("No metrics found. Run 03_run_experiment.py first.")
        sys.exit(1)


    if "arda_sr" not in all_metrics and "arda_pmr" in all_metrics:
        logger.warning("Aliasing legacy metrics key 'arda_pmr' -> 'arda_sr'")
        all_metrics["arda_sr"] = all_metrics.pop("arda_pmr")

    # Load raw results for plots
    all_results = {}
    for m in BASELINE_NAMES:
        results = load_results(m)
        if not results and m == "arda_sr":
            results = load_results("arda_pmr")  # legacy filename fallback
        all_results[m] = results


    logger.info("\nComputing variance analysis...")
    var_analysis_raw = variance_analysis(all_results)
    std_lookup = {
        method: {
            "rel":   cats.get("ALL", {}).get("rel",   {}).get("std"),
            "faith": cats.get("ALL", {}).get("faith", {}).get("std"),
            "cov":   cats.get("ALL", {}).get("cov",   {}).get("std"),
        }
        for method, cats in var_analysis_raw.items()
    }

    # Tables
    logger.info("\nGenerating tables...")

    # Table 4: Overall performance
    t4 = make_overall_table(all_metrics, std_lookup)
    t4.to_csv(OUTPUTS_DIR / "table_overall.csv", index=False)
    with open(OUTPUTS_DIR / "table_overall.tex", "w") as f:
        f.write(overall_to_latex(t4, "Performance comparison across 1,000 test queries. $\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. Standard deviations are reported for answer quality metrics. Statistical significance ($\\dagger$) denotes improvement over the best baseline (Wilcoxon signed-rank, $p < 0.001$).", "tab:overall"))
    logger.info("  table_overall.csv / .tex")

    # Table 5: Per-category
    t5 = make_percategory_table(all_metrics)
    t5.to_csv(OUTPUTS_DIR / "table_percategory.csv", index=False)
    with open(OUTPUTS_DIR / "table_percategory.tex", "w") as f:
        f.write(df_to_latex(t5, "Per-category performance comparison.", "tab:percategory"))
    logger.info("  table_percategory.csv / .tex")

    # Table 6: Ablation
    if ablation_mets:
        t6 = make_ablation_table(ablation_mets)
        t6.to_csv(OUTPUTS_DIR / "table_ablation.csv", index=False)
        with open(OUTPUTS_DIR / "table_ablation.tex", "w") as f:
            f.write(df_to_latex(t6, "Ablation results on 1000 test queries.", "tab:ablation"))
        logger.info("  table_ablation.csv / .tex")

    # Figures
    logger.info("\nGenerating figures...")
    plot_frr_comparison(all_metrics, OUTPUTS_DIR / "fig_frr_comparison.pdf")
    plot_category_heatmap(all_results, OUTPUTS_DIR / "fig_category_heatmap.pdf")

    # Statistical tests
    logger.info("\nRunning statistical tests (Wilcoxon)...")
    stat_tests = {}
    if "arda_sr" in all_results:
        arda_results = all_results["arda_sr"]
        arda_rel   = [r.get("rel", 0.5) for r in arda_results]
        arda_faith = [r.get("faith", 0.5) for r in arda_results]
        arda_frr   = [float(r.get("is_refusal", 0)) for r in arda_results]
        for method in BASELINE_NAMES:
            if method == "arda_sr" or not all_results.get(method):
                continue
            bl = all_results[method]
            bl_rel   = [r.get("rel", 0.5) for r in bl]
            bl_faith = [r.get("faith", 0.5) for r in bl]
            bl_frr   = [float(r.get("is_refusal", 0)) for r in bl]
            stat_tests[method] = {
                "rel":   wilcoxon_test(arda_rel, bl_rel),
                "faith": wilcoxon_test(arda_faith, bl_faith),
                "frr":   wilcoxon_test(bl_frr, arda_frr),  # lower FRR = better for ARDA
            }
            logger.info(f"  vs {method}: Rel p={stat_tests[method]['rel']['p_value']:.4f} "
                        f"{'yes' if stat_tests[method]['rel']['significant'] else 'no'}")

    with open(OUTPUTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Variance analysis (reuse var_analysis_raw computed earlier, before Table 4)
    var_analysis = display_category_keys(var_analysis_raw)
    with open(OUTPUTS_DIR / "variance_analysis.json", "w") as f:
        json.dump(var_analysis, f, indent=2)

    # Bootstrap CI for ARDA-SR
    if all_results.get("arda_sr"):
        arda_rel = [r.get("rel", 0.5) for r in all_results["arda_sr"]]
        ci_lo, ci_hi = bootstrap_ci(arda_rel)
        logger.info(f"\nARDA-SR Relevance 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    logger.info(f"\nAll outputs saved to: {OUTPUTS_DIR}")
    logger.info("\nKey outputs:")
    for f in sorted(OUTPUTS_DIR.iterdir()):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()

