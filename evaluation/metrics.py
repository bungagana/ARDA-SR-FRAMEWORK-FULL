"""
Automatic evaluation metrics:
FRR, FAR, Hit@K, CtxRel, ToolAcc, SRComp, Latency.
LLM-judge metrics (Rel, Faith, Cov) are in llm_judge.py.
"""

import json
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_frr(results: List[Dict]) -> float:
    """
    False Rejection Rate: fraction of answerable queries refused.
    FRR = #{r̂=1 ∧ r=0} / #{r=0}
    r=0: query is answerable (should_be_answerable=True); r̂=1: system refused (is_refusal=True).

    should_be_answerable must come from real evidence/human-validation labeling
    (see merge_human_validation.py) — a query missing this field is treated as
    unresolved and excluded from the denominator rather than defaulted to True,
    so an unlabeled dataset can't silently inflate/deflate FRR.
    """
    answerable = [r for r in results if r.get("should_be_answerable") is True]
    if not answerable:
        return 0.0
    refused = sum(1 for r in answerable if r.get("is_refusal", False))
    return refused / len(answerable)


def compute_far(results: List[Dict]) -> float:
    """
    False Acceptance Rate: fraction of unanswerable queries where system answered.
    FAR = #{r̂=0 ∧ r=1} / #{r=1}
    """
    unanswerable = [r for r in results if r.get("should_be_answerable") is False]
    if not unanswerable:
        # If no explicitly unanswerable items, use AR category with refusal ground-truth
        ar_items = [r for r in results if r.get("category") == "AR"]
        if not ar_items:
            return 0.0
        # In AR category, assume ~30% should legitimately return "uncertain"
        # Items where system confidently answered ambiguous queries = FAR proxy
        accepted = sum(1 for r in ar_items if not r.get("is_refusal", False)
                       and len(r.get("answer", "")) < 50)
        return accepted / len(ar_items)
    answered = sum(1 for r in unanswerable if not r.get("is_refusal", False))
    return answered / len(unanswerable)


def compute_hit_at_k(results: List[Dict], k: int = 5) -> float:
    """
    Hit@K: fraction of queries where ≥1 relevant chunk appears in Top-K evidence.
    Relevance is estimated by checking if any evidence mentions query keywords.
    """
    retrieval_queries = [r for r in results if r.get("evidence")]
    if not retrieval_queries:
        return 0.0
    hits = 0
    for r in retrieval_queries:
        query_tokens = set(r["query"].lower().split())
        evidence = r.get("evidence", [])
        for chunk in evidence[:k]:
            chunk_tokens = set(chunk.get("text", "").lower().split())
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            if overlap > 0.15:  # ≥15% token overlap → relevant
                hits += 1
                break
    return hits / len(retrieval_queries)


def compute_ctx_rel(results: List[Dict], llm_judge=None) -> float:
    """
    Context Relevance: average LLM-judged relevance of retrieved evidence to
    the query, normalised to [0, 1] (raw 1-5 / 5). Per paper Section 2.4.2:
    "CtxRel is assessed by an LLM evaluator on a scale of 1-5 based on the
    adequacy of information supporting the answer."

    Preferred path: each result dict already carries a per-query "ctx_rel"
    score (1-5) populated by LLMJudge.judge_ctx_rel_batch() in the runner —
    this function just averages those. Only if results have NO "ctx_rel"
    field at all (e.g. judge scoring was skipped for cost) does it fall back
    to the cheap lexical token-overlap heuristic in _ctx_rel_heuristic(),
    which does NOT match the paper's stated LLM-judged methodology and should
    be treated as an approximation only.
    """
    retrieval_queries = [r for r in results if r.get("evidence")]
    if not retrieval_queries:
        return 0.0

    have_llm_scores = any(r.get("ctx_rel") is not None for r in retrieval_queries)
    if have_llm_scores:
        scores = [r["ctx_rel"] / 5.0 for r in retrieval_queries if r.get("ctx_rel") is not None]
        return float(np.mean(scores)) if scores else 0.0

    logger.warning(
        "compute_ctx_rel: no LLM-judged 'ctx_rel' found on results — falling back "
        "to the lexical-overlap heuristic, which does NOT match the paper's "
        "stated LLM-judged CtxRel methodology (Section 2.4.2)."
    )
    return _ctx_rel_heuristic(retrieval_queries)


def _ctx_rel_heuristic(retrieval_queries: List[Dict]) -> float:
    """Cheap fallback only — NOT the paper's methodology. See compute_ctx_rel()."""
    scores = []
    for r in retrieval_queries:
        query_tokens = set(r["query"].lower().split())
        evidence = r.get("evidence", [])
        chunk_scores = []
        for chunk in evidence[:5]:
            chunk_tokens = set(chunk.get("text", "").lower().split())
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            chunk_scores.append(min(overlap * 2.5, 1.0))
        scores.append(np.mean(chunk_scores) if chunk_scores else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_tool_acc(results: List[Dict]) -> float:
    """
    Tool Accuracy: fraction of queries where predicted routing mode matches reference.
    Reference mode is inferred from query category.
    """
    category_to_mode = {
        "DK": "m1",
        "FR": "m2",
        "CR": "m3",
        "AR": "m3",
        "PS": "m4",
    }
    routed = [r for r in results if "routing" in r and r["routing"]]
    if not routed:
        return 0.0
    correct = 0
    for r in routed:
        cat = r.get("category", "")
        ref_mode = category_to_mode.get(cat, "m2")
        pred_mode = r.get("mode") or r.get("routing", {}).get("mode", "")
        if pred_mode == ref_mode:
            correct += 1
    return correct / len(routed)


def _sr_schema_score(answer: str) -> float:
    """
    Schema-based Scenario Reasoning compliance score, matching Eq. (23) of the
    paper exactly: SRComp(a) = (1/5) * sum_k 1[c_k in a], for the 5 required
    output components (Section 2.4.2 / Table 12):
      c_1: recommended policy action
      c_2: alternative comparison
      c_3: risk mitigation
      c_4: implementation steps
      c_5: stated assumptions
    Deterministic checklist rather than LLM-judged, same as the reference
    implementation this metric approximates.
    """
    text = (answer or "").lower()
    checks = [
        # c_1: recommended policy action
        any(w in text for w in [
            "recommend", "recommendation", "rekomendasi", "direkomendasikan",
            "prioritas", "priority", "keputusan tunggal", "single recommendation",
        ]),
        # c_2: alternative comparison
        any(w in text for w in [
            "compare", "comparison", "banding", "dibanding", "vs.", "versus",
            "rank", "peringkat", "argmax", "alternatif", "alternative", "skenario", "scenario",
        ]),
        # c_3: risk mitigation
        any(w in text for w in [
            "risk", "risiko", "trade-off", "tradeoff", "mitigasi", "mitigation",
        ]),
        # c_4: implementation steps
        any(w in text for w in [
            "implementation", "implementasi", "langkah", "step", "tahapan",
            "action plan", "rencana aksi", "pelaksanaan", "roadmap",
        ]),
        # c_5: stated assumptions
        any(w in text for w in [
            "assumption", "asumsi", "assume", "diasumsikan", "limitation", "keterbatasan",
        ]),
    ]
    return sum(checks) / len(checks)


def compute_sr_compliance(results: List[Dict]) -> float:
    """
    SRComp: average schema-compliance score over policy-scenario queries.

    For each PS answer, SRComp(a_i) is computed by the deterministic checklist
    in _sr_schema_score(). If the pipeline provides structured sr_info with an
    explicit sr_compliant flag, that flag is used as a full-score shortcut.
    """
    ps_results = [r for r in results if r.get("category") == "PS"]
    if not ps_results:
        return 0.0
    scores = []
    for r in ps_results:
        sr_info = r.get("sr_info", {})
        if sr_info.get("sr_compliant"):
            scores.append(1.0)
            continue
        scores.append(_sr_schema_score(r.get("answer", "")))
    return float(np.mean(scores))


def compute_pmr_compliance(results: List[Dict]) -> float:
    """Backward-compatible alias for older result scripts."""
    return compute_sr_compliance(results)


def compute_latency(results: List[Dict]) -> float:
    """Average response latency in seconds."""
    latencies = [r.get("latency_s", 0.0) for r in results if "latency_s" in r]
    return float(np.mean(latencies)) if latencies else 0.0


def compute_all_metrics(results: List[Dict], llm_scores: Dict | None = None) -> Dict:
    """
    Compute all automatic metrics for a list of results.
    llm_scores: dict of {query_id: {rel, faith, cov}} from LLMJudge.
    """
    metrics = {
        "hit_at_5":     round(compute_hit_at_k(results, k=5), 4),
        "ctx_rel":      round(compute_ctx_rel(results), 4),
        "tool_acc":     round(compute_tool_acc(results), 4),
        "sr_comp":      round(compute_sr_compliance(results), 4),
        "frr":          round(compute_frr(results), 4),
        "far":          round(compute_far(results), 4),
        "latency_s":    round(compute_latency(results), 3),
        "n_queries":    len(results),
    }

    if llm_scores:
        rel_scores   = [llm_scores[r["query_id"]]["rel"] / 5.0
                        for r in results if r.get("query_id") in llm_scores]
        faith_scores = [llm_scores[r["query_id"]]["faith"] / 5.0
                        for r in results if r.get("query_id") in llm_scores]
        cov_scores   = [llm_scores[r["query_id"]]["cov"] / 5.0
                        for r in results if r.get("query_id") in llm_scores]
        metrics["rel"]   = round(float(np.mean(rel_scores)), 4)   if rel_scores   else 0.0
        metrics["faith"] = round(float(np.mean(faith_scores)), 4) if faith_scores else 0.0
        metrics["cov"]   = round(float(np.mean(cov_scores)), 4)   if cov_scores   else 0.0
    else:
        metrics["rel"] = metrics["faith"] = metrics["cov"] = None

    return metrics


def per_category_metrics(results: List[Dict], llm_scores: Dict | None = None) -> Dict:
    """Compute metrics broken down by query category."""
    categories = list(set(r.get("category", "unknown") for r in results))
    out = {}
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_llm = None
        if llm_scores:
            cat_llm = {k: v for k, v in llm_scores.items()
                       if any(r.get("query_id") == k and r.get("category") == cat for r in results)}
        out[cat] = compute_all_metrics(cat_results, cat_llm)
    return out
