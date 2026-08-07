"""
Step 4: Ablation Study (Bottom-Up)
=====================================
Evaluates 5 ablation variants, each adding one ARDA-SR component:

  V0: Base RAG            (standard retrieve-then-generate)
  V1: + AQR module       (adaptive routing)
  V2: + Hybrid retrieval  (AQR + dense/BM25 hybrid)
  V3: + DDA module       (arbitration layer)
  V4: + SR module        (full ARDA-SR)

Run: python 04_ablation.py [--smoke]
Output: results/ablation_{variant}_results.json, results/ablation_metrics.json
"""

import argparse
import json
import logging
import sys
import time

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ablation.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import RESULTS_DIR, DATA_DIR, ABLATION_VARIANTS
from utils.llm_client import GeminiClient
from utils.openai_client import GPTJudgeClient
from utils.kb_builder import KnowledgeBase
from arda_sr.pipeline import ARDASRPipeline
from arda_sr.dda import DEFAULT_BETAS, grid_search_betas
from evaluation.metrics import compute_all_metrics, per_category_metrics
from evaluation.llm_judge import LLMJudge
from evaluation.statistical import summary_table


# ── Ablation variant configs ───────────────────────────────────────────────
# Each variant enables/disables ARDA-SR modules
VARIANT_CONFIGS = {
    "V0_base_rag": {
        "use_aqr":             False,
        "use_dda":             False,
        "use_sr":              False,
        "use_hybrid_retrieval": False,
        "label":                "Base RAG",
    },
    "V1_aqr": {
        "use_aqr":             True,
        "use_dda":             False,
        "use_sr":              False,
        "use_hybrid_retrieval": False,
        "label":                "+ AQR module",
    },
    "V2_hybrid": {
        "use_aqr":             True,
        "use_dda":             False,
        "use_sr":              False,
        "use_hybrid_retrieval": True,
        "label":                "+ Hybrid retrieval",
    },
    "V3_dda": {
        "use_aqr":             True,
        "use_dda":             True,
        "use_sr":              False,
        "use_hybrid_retrieval": True,
        "label":                "+ DDA module",
    },
    "V4_sr": {
        "use_aqr":             True,
        "use_dda":             True,
        "use_sr":              True,
        "use_hybrid_retrieval": True,
        "label":                "+ SR module (full ARDA-SR)",
    },
}


def run_variant(variant_name: str, config: dict, qa_data: list,
                kb: KnowledgeBase, client: GeminiClient, betas=None) -> list:
    pipeline = ARDASRPipeline(
        kb=kb,
        client=client,
        betas=betas,
        use_aqr=config["use_aqr"],
        use_dda=config["use_dda"],
        use_sr=config["use_sr"],
        use_hybrid_retrieval=config["use_hybrid_retrieval"],
    )
    results = []
    for qa in tqdm(qa_data, desc=f"  {variant_name}", leave=False):
        res = pipeline.run(
            query=qa["question"],
            reference_answer=qa.get("reference_answer", ""),
        )
        res["query_id"] = qa["query_id"]
        res["category"] = qa.get("category", "")
        res["method"]   = variant_name
        res["variant_label"] = config["label"]
        results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run on first 10 queries per category only")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge scoring")
    parser.add_argument("--dda-dev-path", type=str, default=None,
                        help="Optional separate development JSON for DDA beta tuning. Do not pass the final test set.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ARDA-SR Step 4: Ablation Study (Bottom-Up)")
    logger.info("=" * 60)

    qa_path = DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        logger.error("QA dataset not found. Run 02_generate_qa.py first.")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    if args.smoke:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for qa in qa_data:
            by_cat[qa.get("category", "?")].append(qa)
        qa_data = []
        for items in by_cat.values():
            qa_data.extend(items[:10])
        logger.info(f"Smoke mode: {len(qa_data)} queries")

    client = GeminiClient()
    kb     = KnowledgeBase().load()
    judge  = LLMJudge(GPTJudgeClient()) if not args.skip_judge else None

    best_betas = DEFAULT_BETAS
    if args.dda_dev_path:
        with open(args.dda_dev_path, encoding="utf-8") as f:
            dev_data = json.load(f)
        val_items = [{"query": q["question"], "reference_answer": q.get("reference_answer", "")}
                     for q in dev_data]
        logger.info(f"Running DDA beta grid search on {len(val_items)} separate development samples...")
        best_betas = grid_search_betas(val_items, client)
    else:
        logger.info(f"Using fixed DDA beta design values: {best_betas}")

    ablation_metrics = {}

    for variant_name, config in VARIANT_CONFIGS.items():
        out_path = RESULTS_DIR / f"ablation_{variant_name}_results.json"

        # Resume: skip variant if result file already exists (not smoke mode)
        if not args.smoke and out_path.exists():
            logger.info(f"  Skipping {variant_name}: results already exist ({out_path.name})")
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            llm_scores_loaded = {
                r["query_id"]: {"rel": int(r.get("rel", 0.5) * 5),
                                "faith": int(r.get("faith", 0.5) * 5),
                                "cov": int(r.get("cov", 0.5) * 5)}
                for r in results if r.get("rel") is not None
            } or None
            metrics = compute_all_metrics(results, llm_scores_loaded)
            cat_metrics = per_category_metrics(results, llm_scores_loaded)
            ablation_metrics[variant_name] = {**metrics, "label": config["label"], "per_category": cat_metrics}
            continue

        logger.info(f"\n{'─'*40}")
        logger.info(f"Variant: {variant_name} — {config['label']}")
        t = time.time()

        results = run_variant(variant_name, config, qa_data, kb, client, best_betas)
        elapsed = time.time() - t
        logger.info(f"  Completed in {elapsed:.1f}s")

        llm_scores = None
        if judge:
            logger.info(f"  LLM judging {len(results)} answers...")
            llm_scores = judge.judge_batch(results)
            for r in results:
                qid = r.get("query_id", "")
                if qid in llm_scores:
                    r.update({
                        "rel":   llm_scores[qid]["rel"] / 5.0,
                        "faith": llm_scores[qid]["faith"] / 5.0,
                        "cov":   llm_scores[qid]["cov"] / 5.0,
                    })

        metrics = compute_all_metrics(results, llm_scores)
        cat_metrics = per_category_metrics(results, llm_scores)
        ablation_metrics[variant_name] = {
            **metrics,
            "label": config["label"],
            "per_category": cat_metrics,
        }

        # Save immediately (resume support)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved → {out_path.name}")

        logger.info(f"  Rel={metrics.get('rel','--')} | "
                    f"Faith={metrics.get('faith','--')} | "
                    f"FRR={metrics.get('frr','--')}")

    # ── Save and display ───────────────────────────────────────────────────
    with open(RESULTS_DIR / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ablation_metrics, f, indent=2)

    rows = summary_table(ablation_metrics, method_order=list(VARIANT_CONFIGS.keys()))
    df = pd.DataFrame(rows)
    # Add label column
    df.insert(1, "Added Component",
              [VARIANT_CONFIGS[m]["label"] for m in VARIANT_CONFIGS if m in ablation_metrics])
    df.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)

    logger.info("\nAblation Study Results:")
    logger.info(df.to_string(index=False))
    logger.info(f"\n✓ Ablation results saved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
