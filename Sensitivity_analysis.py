# -*- coding: utf-8 -*-
"""
Sensitivity analysis for ARDA-SR's fixed hyperparameters, per Section 2.3.4 /
Fig. 6: robustness is validated by perturbing each parameter within a small
range around its fixed value and checking metric stability.

This is a stability check, not a tuning sweep: it re-runs a fixed query
subset at each perturbed value and reports how much the metrics move.
It does not search for a "best" value and does not feed back into the
fixed defaults in config.py / arda_sr/*.py.

Four sweeps, mirroring Fig. 6's panels:
  1. dda_margin  — δ (DDA_DECISION_MARGIN): 0.03 / 0.05 / 0.07
  2. dda_beta    — β = (β1,β2,β3,β4): three vectors around the fixed default
  3. hybrid_alpha— α (HYBRID_ALPHA):   0.40 / 0.60 / 0.80
  4. sr_lambda   — λ_SR (SR_LAMBDA):   0.30 / 0.50 / 0.70

Design notes:
  - Sweeps 1-3 run the full pipeline (AQR routing intact) over a general
    query subset (categories DK/FR/CR/AR — excludes PS, which routes to SR
    not DDA) and report frr, far, rel, faith, cov (+ hit_at_5, ctx_rel for
    the alpha sweep, since alpha only affects retrieval).
  - Sweep 4 (λ_SR) only affects the SR module (mode m4 / PS queries), so it
    runs SR directly over a PS-only subset, and reports sr_comp + rel/faith/
    cov on PS answers + rank_stability (fraction of queries whose top-ranked
    scenario is unchanged from the λ=0.5 baseline).

Costs API calls (Gemini). Output is written to outputs/sensitivity_results_v2.csv.

Usage:
    python _sensitivity_analysis.py                  # default subset sizes
    python _sensitivity_analysis.py --n-general 24 --n-ps 15
    python _sensitivity_analysis.py --dry-run         # print plan, no API calls
"""
import argparse
import csv
import json
import logging
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED, SR_NUM_SCENARIOS


from utils.kb_builder import KnowledgeBase
from utils.llm_client import GeminiClient
from utils.openai_client import GPTJudgeClient
import arda_sr.dda as dda_mod
import arda_sr.retrieval as retrieval_mod
from arda_sr.dda import DEFAULT_BETAS
from arda_sr.pipeline import ARDASRPipeline
from arda_sr.sr import SR
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import compute_all_metrics

QA_PATH = DATA_DIR / "qa_dataset.json"
OUT_CSV = OUTPUTS_DIR / "sensitivity_results_v2.csv"

DELTA_VALUES = [0.03, 0.05, 0.07]
BETA_VALUES = [
    {"b1": 0.35, "b2": 0.20, "b3": 0.20, "b4": 0.25},
    {"b1": 0.30, "b2": 0.25, "b3": 0.20, "b4": 0.25},  # paper default
    {"b1": 0.25, "b2": 0.30, "b3": 0.20, "b4": 0.25},
]
ALPHA_VALUES = [0.40, 0.60, 0.80]
LAMBDA_VALUES = [0.30, 0.50, 0.70]


def load_subset(n_general: int, n_ps: int):
    with open(QA_PATH, encoding="utf-8") as f:
        qa_data = json.load(f)
    rng = random.Random(RANDOM_SEED)

    by_cat = {}
    for qa in qa_data:
        by_cat.setdefault(qa.get("category", ""), []).append(qa)

    general_cats = [c for c in by_cat if c != "PS"]
    per_cat_n = max(1, n_general // max(1, len(general_cats)))
    general_subset = []
    for c in general_cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        general_subset.extend(pool[:per_cat_n])

    ps_pool = by_cat.get("PS", [])[:]
    rng.shuffle(ps_pool)
    ps_subset = ps_pool[:n_ps]

    return general_subset, ps_subset


def run_general_sweep(name, values, apply_fn, qa_subset, kb, client, judge, need_retrieval_metrics=False):
    """Re-run the full pipeline over qa_subset once per value in `values`."""
    rows = []
    for value in values:
        restore = apply_fn(value)
        try:
            pipeline = ARDASRPipeline(kb, client)  # fresh instance; picks up patched module constants
            results = []
            for qa in qa_subset:
                res = pipeline.run(query=qa["question"], reference_answer=qa.get("reference_answer", ""))
                res["query_id"] = qa["query_id"]
                res["category"] = qa.get("category", "")
                results.append(res)

            llm_scores = judge.judge_batch(results, show_progress=False)
            for r in results:
                if r["query_id"] in llm_scores:
                    r.update(llm_scores[r["query_id"]])

            metrics = compute_all_metrics(results, llm_scores)
            row = {
                "analysis": name,
                "parameter": _param_label(name),
                "value": _value_label(name, value),
                "frr": metrics["frr"],
                "far": metrics["far"],
                "relevance": metrics.get("rel"),
                "faithfulness": metrics.get("faith"),
                "coverage": metrics.get("cov"),
            }
            if need_retrieval_metrics:
                ctx_rel_scores = judge.judge_ctx_rel_batch(results, show_progress=False)
                for r in results:
                    if r["query_id"] in ctx_rel_scores:
                        r["ctx_rel"] = ctx_rel_scores[r["query_id"]]
                row["hit_at_5"] = metrics["hit_at_5"]
                row["ctx_rel"] = round(
                    sum(ctx_rel_scores.values()) / (5.0 * len(ctx_rel_scores)), 4
                ) if ctx_rel_scores else None
            rows.append(row)
            logger.info(f"[{name}] value={value} -> frr={row['frr']} rel={row['relevance']} faith={row['faithfulness']}")
        finally:
            restore()
    return rows


def run_lambda_sweep(values, ps_subset, kb, client, judge):
    """
    Isolates the effect of lambda_sr on scenario RANKING: scenarios (and their
    p_success/utility/risk/loss estimates) are generated ONCE per query and
    cached, so the only thing that changes across lambda values is the EU
    formula's argmax and the resulting policy answer — not a fresh,
    independently-sampled scenario set each time (which would confound
    lambda's effect with plain LLM sampling variance).
    """
    retriever = retrieval_mod.HybridRetriever(kb)
    sr_helper = SR(client, lam=0.5, n_scenarios=SR_NUM_SCENARIOS)  # only used for its static/instance helpers

    evidence_cache, scenarios_cache = {}, {}
    for qa in ps_subset:
        meta_filter = retrieval_mod.HybridRetriever.extract_metadata_from_query(qa["question"])
        evidence = retriever.retrieve(qa["question"], metadata_filter=meta_filter or None)
        evidence_cache[qa["query_id"]] = evidence
        evidence_text = sr_helper._format_evidence(evidence)
        scenarios_cache[qa["query_id"]] = sr_helper._generate_scenarios(qa["question"], evidence_text)

    baseline_top = {}
    rows = []
    for lam in values:
        results = []
        top_scenarios = {}
        for qa in ps_subset:
            qid = qa["query_id"]
            scenarios = scenarios_cache[qid]
            evidence = evidence_cache[qid]
            if not scenarios:
                fb = sr_helper._fallback(qa["question"])
                results.append({
                    "query": qa["question"], "query_id": qid, "category": "PS",
                    "reference": qa.get("reference_answer", ""), "answer": fb["answer"],
                    "evidence": evidence, "is_refusal": True, "sr_info": fb,
                })
                continue

            for s in scenarios:
                p, u = float(s.get("p_success", 0.5)), float(s.get("utility", 0.5))
                r, lo = float(s.get("risk", 0.5)), float(s.get("loss", 0.5))
                s["eu"] = round(p * u - lam * r * lo, 4)
            optimal = max(scenarios, key=lambda s: s["eu"])
            top_scenarios[qid] = optimal.get("name")

            scenarios_text = sr_helper._format_scenarios(scenarios)
            evidence_text = sr_helper._format_evidence(evidence)
            answer = sr_helper._generate_answer(qa["question"], optimal, scenarios_text, evidence_text)
            compliant = sr_helper._check_compliance(answer)
            results.append({
                "query": qa["question"], "query_id": qid, "category": "PS",
                "reference": qa.get("reference_answer", ""), "answer": answer,
                "evidence": evidence, "is_refusal": not bool(answer.strip()),
                "sr_info": {"optimal_scenario": optimal, "sr_compliant": compliant},
            })

        llm_scores = judge.judge_batch(results, show_progress=False)
        metrics = compute_all_metrics(results, llm_scores)

        if lam == 0.50:
            baseline_top = dict(top_scenarios)
        if baseline_top:
            matches = sum(1 for qid, s in top_scenarios.items() if baseline_top.get(qid) == s)
            rank_stability = round(matches / len(top_scenarios), 4) if top_scenarios else None
        else:
            rank_stability = None  # backfilled below once the 0.50 baseline is known

        rows.append({
            "analysis": "sr_lambda",
            "parameter": "lambda_sr",
            "value": lam,
            "ctx_rel": None,
            "sr_comp": metrics["sr_comp"],
            "ps_relevance": metrics.get("rel"),
            "ps_faithfulness": metrics.get("faith"),
            "ps_coverage": metrics.get("cov"),
            "rank_stability": rank_stability,
            "_top_scenarios": top_scenarios,  # stripped before CSV write
        })
        logger.info(f"[sr_lambda] value={lam} -> sr_comp={metrics['sr_comp']} rank_stability={rank_stability}")
    if baseline_top:
        for row in rows:
            if row["rank_stability"] is None:
                ts = row.pop("_top_scenarios")
                matches = sum(1 for qid, s in ts.items() if baseline_top.get(qid) == s)
                row["rank_stability"] = round(matches / len(ts), 4) if ts else None
    for row in rows:
        row.pop("_top_scenarios", None)
    return rows


def _param_label(name):
    return {"dda_margin": "delta_dda", "dda_beta": "beta_vector", "hybrid_alpha": "alpha"}[name]


def _value_label(name, value):
    if name == "dda_beta":
        b = value
        return f"{b['b1']:.2f}/{b['b2']:.2f}/{b['b3']:.2f}/{b['b4']:.2f}"
    return value


def apply_delta(value):
    original = dda_mod.DDA_DECISION_MARGIN
    dda_mod.DDA_DECISION_MARGIN = value
    return lambda: setattr(dda_mod, "DDA_DECISION_MARGIN", original)


def apply_beta(value):
    # betas is passed via pipeline constructor, not a module patch; no-op restore.
    return lambda: None


def apply_alpha(value):
    original = retrieval_mod.HYBRID_ALPHA
    retrieval_mod.HYBRID_ALPHA = value
    return lambda: setattr(retrieval_mod, "HYBRID_ALPHA", original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-general", type=int, default=24, help="subset size for delta/beta/alpha sweeps (non-PS)")
    ap.add_argument("--n-ps", type=int, default=200,
                     help="subset size for lambda_sr sweep (PS only) — matches the paper's "
                          "stated n=200 for this sweep (Section 2.3.5)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and subset sizes, make no API calls")
    args = ap.parse_args()

    general_subset, ps_subset = load_subset(args.n_general, args.n_ps)
    logger.info(f"General subset (delta/beta/alpha): {len(general_subset)} queries "
                f"({[qa['category'] for qa in general_subset].count('DK')} DK / "
                f"non-PS categories mixed)")
    logger.info(f"PS subset (lambda_sr): {len(ps_subset)} queries")
    logger.info(f"delta sweep: {DELTA_VALUES}")
    logger.info(f"beta sweep: {[_value_label('dda_beta', b) for b in BETA_VALUES]}")
    logger.info(f"alpha sweep: {ALPHA_VALUES}")
    logger.info(f"lambda sweep: {LAMBDA_VALUES}")
    total_calls_est = (len(DELTA_VALUES) + len(BETA_VALUES) + len(ALPHA_VALUES)) * len(general_subset) * 3 \
        + len(LAMBDA_VALUES) * len(ps_subset) * 2
    logger.info(f"Estimated LLM calls: ~{total_calls_est} (rough; DDA/AQR/judge each call once or twice per query)")

    if args.dry_run:
        logger.info("Dry run only — no API calls made. Re-run without --dry-run to execute.")
        return

    client = GeminiClient()
    kb = KnowledgeBase().load()
    judge = LLMJudge(GPTJudgeClient())

    all_rows = []
    t0 = time.time()

    all_rows += run_general_sweep("dda_margin", DELTA_VALUES, apply_delta, general_subset, kb, client, judge)

    for betas in BETA_VALUES:
        # betas can't be applied via module patch (constructor param), so build
        # the pipeline directly here rather than through run_general_sweep's apply_fn.
        pipeline = ARDASRPipeline(kb, client, betas=betas)
        results = []
        for qa in general_subset:
            res = pipeline.run(query=qa["question"], reference_answer=qa.get("reference_answer", ""))
            res["query_id"] = qa["query_id"]
            res["category"] = qa.get("category", "")
            results.append(res)
        llm_scores = judge.judge_batch(results, show_progress=False)
        metrics = compute_all_metrics(results, llm_scores)
        row = {
            "analysis": "dda_beta", "parameter": "beta_vector",
            "value": _value_label("dda_beta", betas),
            "frr": metrics["frr"], "far": metrics["far"],
            "relevance": metrics.get("rel"), "faithfulness": metrics.get("faith"),
            "coverage": metrics.get("cov"),
        }
        all_rows.append(row)
        logger.info(f"[dda_beta] value={row['value']} -> frr={row['frr']} rel={row['relevance']}")

    all_rows += run_general_sweep("hybrid_alpha", ALPHA_VALUES, apply_alpha, general_subset, kb, client, judge,
                                   need_retrieval_metrics=True)

    all_rows += run_lambda_sweep(LAMBDA_VALUES, ps_subset, kb, client, judge)

    fieldnames = ["analysis", "parameter", "value", "frr", "far", "relevance", "faithfulness",
                  "coverage", "hit_at_5", "ctx_rel", "sr_comp", "ps_relevance", "ps_faithfulness",
                  "ps_coverage", "rank_stability"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info(f"Done in {time.time()-t0:.1f}s. Saved {len(all_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
