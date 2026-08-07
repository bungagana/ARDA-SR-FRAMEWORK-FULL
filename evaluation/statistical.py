"""
Statistical analysis for experiment results.
Wilcoxon signed-rank test, Cohen's d, bootstrap CI, variance analysis.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats

from config import WILCOXON_ALPHA, BOOTSTRAP_SAMPLES, RANDOM_SEED

logger = logging.getLogger(__name__)

rng = np.random.default_rng(RANDOM_SEED)


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = WILCOXON_ALPHA,
) -> Dict:
    """
    Wilcoxon signed-rank test between method A and method B.
    Returns: {statistic, p_value, significant, effect_size_d}
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    if n < 5:
        return {"statistic": None, "p_value": 1.0, "significant": False, "effect_size_d": 0.0}

    try:
        stat, p = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        d = cohens_d(a, b)
        return {
            "statistic": round(float(stat), 4),
            "p_value":   round(float(p), 6),
            "significant": bool(p < alpha),
            "effect_size_d": round(d, 4),
        }
    except Exception as exc:
        logger.warning(f"Wilcoxon test failed: {exc}")
        return {"statistic": None, "p_value": 1.0, "significant": False, "effect_size_d": 0.0}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    diff = a - b
    if diff.std() < 1e-10:
        return 0.0
    return float(diff.mean() / diff.std())


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for the mean.
    Returns (lower, upper) at the specified CI level.
    """
    scores_arr = np.array(scores)
    boot_means = [rng.choice(scores_arr, size=len(scores_arr), replace=True).mean()
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return round(lower, 4), round(upper, 4)


def variance_analysis(results_by_method: Dict[str, List[Dict]]) -> Dict:
    """
    Compute per-method, per-category variance in key metrics.
    Addresses reviewer concern: variance should be consistent with query complexity.
    """
    metrics_of_interest = ["rel", "faith", "cov", "frr"]
    out = {}
    for method, results in results_by_method.items():
        out[method] = {}
        for cat in ["DK", "FR", "CR", "AR", "PS", "ALL"]:
            if cat == "ALL":
                subset = results
            else:
                subset = [r for r in results if r.get("category") == cat]
            if not subset:
                continue
            cat_stats = {}
            for m in metrics_of_interest:
                vals = [r.get(m) for r in subset if r.get(m) is not None]
                if vals:
                    cat_stats[m] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "std":  round(float(np.std(vals)), 4),
                        "min":  round(float(np.min(vals)), 4),
                        "max":  round(float(np.max(vals)), 4),
                    }
            out[method][cat] = cat_stats
    return out


def significance_matrix(
    method_scores: Dict[str, List[float]],
    alpha: float = WILCOXON_ALPHA,
) -> Dict:
    """
    Compute pairwise Wilcoxon significance matrix.
    Returns dict: {(method_a, method_b): {p_value, significant, effect_size_d}}
    """
    methods = list(method_scores.keys())
    matrix = {}
    for i, m_a in enumerate(methods):
        for j, m_b in enumerate(methods):
            if i == j:
                continue
            key = f"{m_a}_vs_{m_b}"
            matrix[key] = wilcoxon_test(method_scores[m_a], method_scores[m_b], alpha)
    return matrix


def summary_table(
    all_metrics: Dict[str, Dict],
    method_order: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Format metrics dict into a list-of-rows table for CSV/LaTeX export.
    all_metrics: {method_name: {rel, faith, cov, hit_at_5, ctx_rel, tool_acc, frr, far, latency_s}}
    """
    order = method_order or list(all_metrics.keys())
    rows = []
    for m in order:
        if m not in all_metrics:
            continue
        met = all_metrics[m]
        rows.append({
            "Method":       m,
            "Rel":          met.get("rel", "--"),
            "Faith":        met.get("faith", "--"),
            "Cov":          met.get("cov", "--"),
            "Hit@5":        met.get("hit_at_5", "--"),
            "CtxRel":       met.get("ctx_rel", "--"),
            "ToolAcc":      met.get("tool_acc", "--"),
            "FRR":          met.get("frr", "--"),
            "FAR":          met.get("far", "--"),
            "Latency(s)":   met.get("latency_s", "--"),
        })
    return rows
