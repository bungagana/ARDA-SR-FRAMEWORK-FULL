"""
Step 3: Run Main Experiment
==============================
Runs all 10 baselines + ARDA-SR on the full QA dataset.
Computes automatic metrics + LLM-judge scores.
Saves per-method result JSONs and a summary metrics CSV.

Run: python 03_run_experiment.py [--method METHOD] [--smoke]
  --method METHOD  : run only one method (e.g. --method arda_sr)
  --smoke          : run on first 10 queries only (sanity check)

Output: results/{method}_results.json, results/all_metrics.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import RESULTS_DIR, DATA_DIR, BASELINE_NAMES, RANDOM_SEED
from utils.llm_client import GeminiClient
from utils.openai_client import GPTJudgeClient
from utils.kb_builder import KnowledgeBase
from baselines import ALL_BASELINES
from arda_sr.pipeline import ARDASRPipeline
from arda_sr.dda import DEFAULT_BETAS, grid_search_betas
from evaluation.metrics import compute_all_metrics, per_category_metrics
from evaluation.llm_judge import LLMJudge
from evaluation.statistical import summary_table, significance_matrix


def load_qa_dataset(path: Path, smoke: bool = False) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        logger.info("Smoke mode: using first 10 queries per category")
        from collections import defaultdict
        by_cat = defaultdict(list)
        for qa in data:
            by_cat[qa.get("category", "?")].append(qa)
        data = []
        for cat_items in by_cat.values():
            data.extend(cat_items[:10])
    return data


def _ckpt_path(method_name: str) -> Path:
    return RESULTS_DIR / f"{method_name}_ckpt.json"


def _load_ckpt(method_name: str) -> dict:
    """Load per-query checkpoint. Returns {query_id: result_dict}."""
    p = _ckpt_path(method_name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        return {r["query_id"]: r for r in rows}
    return {}


def _save_ckpt(method_name: str, done: dict) -> None:
    p = _ckpt_path(method_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(done.values()), f, ensure_ascii=False)


def run_method(method_name: str, qa_data: list, kb: KnowledgeBase, client: GeminiClient,
               betas=None) -> list:
    """Run a single method on all QA pairs, with per-query checkpoint."""
    done = _load_ckpt(method_name)
    if done:
        logger.info(f"  Resuming {method_name}: {len(done)}/{len(qa_data)} already done")

    if method_name == "arda_sr":
        pipeline = ARDASRPipeline(kb, client, betas=betas)
    else:
        PipelineClass = ALL_BASELINES[method_name]
        pipeline = PipelineClass(kb, client)

    remaining = [qa for qa in qa_data if qa["query_id"] not in done]
    for qa in tqdm(remaining, desc=f"  {method_name}", leave=False):
        res = pipeline.run(
            query=qa["question"],
            reference_answer=qa.get("reference_answer", ""),
        )
        res["query_id"] = qa["query_id"]
        res["category"] = qa.get("category", "")
        res["method"]   = method_name
        done[qa["query_id"]] = res
        _save_ckpt(method_name, done)

    # preserve original order
    id_order = {qa["query_id"]: i for i, qa in enumerate(qa_data)}
    return sorted(done.values(), key=lambda r: id_order.get(r["query_id"], 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        help="Run only this method (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run on first 10 queries per category only")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge scoring (faster, auto-metrics only)")
    parser.add_argument("--dda-dev-path", type=str, default=None,
                        help="Optional separate development JSON for DDA beta tuning. Do not pass the final test set.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ARDA-SR Step 3: Main Experiment")
    logger.info("=" * 60)

    qa_path = DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        logger.error("QA dataset not found. Run 02_generate_qa.py first.")
        sys.exit(1)

    qa_data = load_qa_dataset(qa_path, smoke=args.smoke)
    logger.info(f"Loaded {len(qa_data)} QA pairs")

    client = GeminiClient()
    kb     = KnowledgeBase().load()
    # Judge is gpt-5.4-mini (separate model family from Gemini, which generates
    # the answers being judged) — per Section 2.4.2: "gpt-5.4-mini as an
    # independent evaluator... reducing potential bias resulting from the use
    # of the same model for both generation and evaluation."
    judge  = LLMJudge(GPTJudgeClient()) if not args.skip_judge else None

    best_betas = DEFAULT_BETAS
    if args.dda_dev_path:
        dev_path = Path(args.dda_dev_path)
        with open(dev_path, encoding="utf-8") as f:
            dev_data = json.load(f)
        val_items = [{"query": q["question"], "reference_answer": q.get("reference_answer", "")}
                     for q in dev_data]
        logger.info(f"\nRunning DDA beta grid search on {len(val_items)} separate development samples...")
        best_betas = grid_search_betas(val_items, client)
        logger.info(f"Best beta: {best_betas}")
    else:
        logger.info(f"\nUsing fixed DDA beta design values: {best_betas}")

    # ── Determine which methods to run ────────────────────────────────────
    requested = [args.method] if args.method else BASELINE_NAMES

    # Resume: skip methods whose result file already exists (unless --smoke)
    methods_to_run = []
    for m in requested:
        out_path = RESULTS_DIR / f"{m}_results.json"
        if not args.smoke and out_path.exists():
            logger.info(f"  Skipping {m}: results already exist ({out_path.name})")
        else:
            methods_to_run.append(m)

    if not methods_to_run:
        logger.info("All methods already completed. Re-running aggregation only.")
    else:
        logger.info(f"\nRunning {len(methods_to_run)} methods: {methods_to_run}")

    all_metrics: dict = {}
    all_results: dict = {}

    # Load existing results for methods we skipped
    for m in requested:
        out_path = RESULTS_DIR / f"{m}_results.json"
        if m not in methods_to_run and out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                all_results[m] = json.load(f)

    for method in methods_to_run:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Running: {method}")
        t = time.time()

        results = run_method(method, qa_data, kb, client,
                             betas=best_betas if method == "arda_sr" else None)
        elapsed = time.time() - t
        logger.info(f"  Completed in {elapsed:.1f}s")

        # LLM judge scoring
        llm_scores = None
        if judge and not args.skip_judge:
            logger.info(f"  Running LLM judge on {len(results)} answers...")
            llm_scores = judge.judge_batch(results, show_progress=True)
            for r in results:
                qid = r.get("query_id", "")
                if qid in llm_scores:
                    r.update({
                        "rel":   llm_scores[qid]["rel"] / 5.0,
                        "faith": llm_scores[qid]["faith"] / 5.0,
                        "cov":   llm_scores[qid]["cov"] / 5.0,
                    })

            # CtxRel: separate LLM judge pass over retrieval quality (Section
            # 2.4.2 — "assessed by an LLM evaluator on a scale of 1-5").
            # Only meaningful for queries where retrieval actually ran.
            logger.info(f"  Running CtxRel judge on retrieval evidence...")
            ctx_rel_scores = judge.judge_ctx_rel_batch(results, show_progress=True)
            for r in results:
                qid = r.get("query_id", "")
                if qid in ctx_rel_scores:
                    r["ctx_rel"] = ctx_rel_scores[qid]

        # Compute metrics
        metrics = compute_all_metrics(results, llm_scores)
        cat_metrics = per_category_metrics(results, llm_scores)
        all_metrics[method] = {**metrics, "per_category": cat_metrics}
        all_results[method] = results

        # Save final result and remove checkpoint
        out_path = RESULTS_DIR / f"{method}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        ckpt = _ckpt_path(method)
        if ckpt.exists():
            ckpt.unlink()
        logger.info(f"  Saved → {out_path.name}")

        logger.info(f"  FRR={metrics.get('frr','--')} | "
                    f"Rel={metrics.get('rel','--')} | "
                    f"Faith={metrics.get('faith','--')}")

    # ── Re-compute metrics for any loaded (not freshly run) methods ──────
    for method, results in all_results.items():
        if method not in all_metrics:
            llm_scores_loaded = {
                r["query_id"]: {"rel": int(r.get("rel", 0.5) * 5),
                                "faith": int(r.get("faith", 0.5) * 5),
                                "cov": int(r.get("cov", 0.5) * 5)}
                for r in results if r.get("rel") is not None
            } or None
            metrics = compute_all_metrics(results, llm_scores_loaded)
            cat_metrics = per_category_metrics(results, llm_scores_loaded)
            all_metrics[method] = {**metrics, "per_category": cat_metrics}

    # ── Aggregate outputs ─────────────────────────────────────────────────
    with open(RESULTS_DIR / "all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Summary table
    rows = summary_table(all_metrics)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    logger.info(f"\n✓ Results saved to: {RESULTS_DIR}")
    logger.info("\nMetrics Summary:")
    logger.info(df.to_string(index=False))

    # Statistical significance: ARDA-SR vs each baseline
    if "arda_sr" in all_results and len(methods_to_run) > 1:
        logger.info("\nStatistical Significance (ARDA-SR vs baselines):")
        arda_rel = [r.get("rel", 0.5) for r in all_results["arda_sr"]]
        for m in methods_to_run:
            if m == "arda_sr":
                continue
            bl_rel = [r.get("rel", 0.5) for r in all_results.get(m, [])]
            if bl_rel:
                from evaluation.statistical import wilcoxon_test
                test = wilcoxon_test(arda_rel, bl_rel)
                sig = "✓" if test["significant"] else "✗"
                logger.info(f"  vs {m}: p={test['p_value']:.4f} {sig} (d={test['effect_size_d']:.3f})")


if __name__ == "__main__":
    main()
