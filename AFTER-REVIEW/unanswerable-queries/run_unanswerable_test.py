import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(THIS_DIR / "unanswerable_test.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from utils.llm_client import GeminiClient
from utils.kb_builder import KnowledgeBase
from baselines import ALL_BASELINES
from arda_sr.pipeline import ARDASRPipeline
from arda_sr.dda import DEFAULT_BETAS

DEFAULT_METHODS = ["standard_rag", "selfrag", "arda_sr"]
ALL_METHOD_NAMES = list(ALL_BASELINES.keys()) + ["arda_sr"]

EXTRA_REFUSAL_PHRASES = [
    "tidak tersedia", "tidak memuat informasi", "tidak ditemukan",
    "tidak dibahas", "tidak mencakup", "tidak dicakup",
    "belum tersedia", "tidak terdapat informasi", "tidak disebutkan",
    "tidak ada dalam bukti", "tidak ada di dalam bukti",
    "bukti yang diberikan tidak", "tidak memberikan informasi",
    "not covered", "not mentioned", "does not contain", "does not mention",
    "no relevant information", "not found in the",
]


def is_refusal_corrected(answer: str, raw_is_refusal: bool) -> bool:
    if raw_is_refusal:
        return True
    text = (answer or "").lower()
    return any(p in text for p in EXTRA_REFUSAL_PHRASES)


def load_unanswerable_queries(smoke: bool = False) -> list:
    path = THIS_DIR / "unanswerable_queries.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        # 6 items = ~1-2 per unans_type, quick sanity check before a full run
        data = data[:2] + data[15:17] + data[30:32] + data[45:47]
    return data


def _ckpt_path(method_name: str) -> Path:
    return RESULTS_DIR / f"{method_name}_unanswerable_ckpt.json"


def _load_ckpt(method_name: str) -> dict:
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


def run_method(method_name: str, queries: list, kb: KnowledgeBase, client: GeminiClient) -> list:
    done = _load_ckpt(method_name)
    if done:
        logger.info(f"  Resuming {method_name}: {len(done)}/{len(queries)} already done")

    if method_name == "arda_sr":
        pipeline = ARDASRPipeline(kb, client, betas=DEFAULT_BETAS)
    else:
        PipelineClass = ALL_BASELINES[method_name]
        pipeline = PipelineClass(kb, client)

    remaining = [q for q in queries if q["query_id"] not in done]
    for q in tqdm(remaining, desc=f"  {method_name}", leave=False):
        res = pipeline.run(query=q["question"], reference_answer="")
        res["query_id"] = q["query_id"]
        res["unans_type"] = q["unans_type"]
        res["reason"] = q["reason"]
        res["method"] = method_name
        # Ground truth for this whole dataset: every item is unanswerable
        # (r_i = 1), the complement of the main benchmark's r_i = 0 case.
        res["should_be_answerable"] = False
        # Keep the pipeline's own is_refusal (for auditability / comparison
        # with how the main experiment defines refusal), and compute a
        # corrected version with the extra Indonesian "no evidence" phrases
        # this experiment surfaced. ARR/FAR below use the corrected field.
        res["is_refusal_raw"] = res.get("is_refusal", False)
        res["is_refusal"] = is_refusal_corrected(res.get("answer", ""), res["is_refusal_raw"])
        done[q["query_id"]] = res
        _save_ckpt(method_name, done)

    id_order = {q["query_id"]: i for i, q in enumerate(queries)}
    return sorted(done.values(), key=lambda r: id_order.get(r["query_id"], 0))


def compute_far(results: list) -> float:
    """False Acceptance Rate: fraction of unanswerable queries the system answered
    instead of refusing. Complement of the paper's FRR (Eq. 24). Lower is better."""
    if not results:
        return 0.0
    accepted = sum(1 for r in results if not r.get("is_refusal", False))
    return accepted / len(results)


def compute_arr(results: list) -> float:
    """Appropriate Refusal Rate = 1 - FAR: fraction of unanswerable queries the
    system correctly refused. Higher is better — this is the number to report
    alongside FRR (also reported ↑-is-good) so both metrics in the paper read
    the same direction."""
    return 1.0 - compute_far(results)


def arr_by_type(results: list) -> dict:
    by_type = {}
    types = sorted(set(r["unans_type"] for r in results))
    for t in types:
        subset = [r for r in results if r["unans_type"] == t]
        by_type[t] = round(compute_arr(subset), 4)
    return by_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names, e.g. standard_rag,selfrag,arda_sr")
    parser.add_argument("--all", action="store_true",
                        help="Run all 10 baselines + arda_sr")
    parser.add_argument("--smoke", action="store_true",
                        help="Sanity check on ~6 queries only")
    args = parser.parse_args()

    if args.all:
        methods = ALL_METHOD_NAMES
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        unknown = [m for m in methods if m not in ALL_METHOD_NAMES]
        if unknown:
            logger.error(f"Unknown method(s): {unknown}. Valid: {ALL_METHOD_NAMES}")
            sys.exit(1)
    else:
        methods = DEFAULT_METHODS

    logger.info("=" * 60)
    logger.info("Unanswerable Query Robustness Test (post-review addition)")
    logger.info(f"Methods: {methods}")
    logger.info("=" * 60)

    queries = load_unanswerable_queries(smoke=args.smoke)
    logger.info(f"Loaded {len(queries)} unanswerable queries")

    client = GeminiClient()
    kb = KnowledgeBase().load()

    all_results = {}
    all_arr = {}
    all_far = {}
    all_arr_by_type = {}

    for method in methods:
        logger.info(f"\n{'-'*40}\nRunning: {method}")
        t = time.time()
        results = run_method(method, queries, kb, client)
        elapsed = time.time() - t
        logger.info(f"  Completed in {elapsed:.1f}s")

        out_path = RESULTS_DIR / f"{method}_unanswerable_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        ckpt = _ckpt_path(method)
        if ckpt.exists():
            ckpt.unlink()

        arr = compute_arr(results)
        far = 1.0 - arr
        arr_type = arr_by_type(results)
        all_results[method] = results
        all_arr[method] = round(arr, 4)
        all_far[method] = round(far, 4)
        all_arr_by_type[method] = arr_type

        n_refused = sum(1 for r in results if r.get("is_refusal", False))
        logger.info(f"  ARR (overall) = {arr:.4f}  ({n_refused}/{len(results)} correctly refused)  [FAR = {far:.4f}]")
        for t_, v in arr_type.items():
            logger.info(f"    {t_:28s} ARR = {v:.4f}")

    # ── Aggregate summary ────────────────────────────────────────────────
    types = sorted(set(q["unans_type"] for q in queries))
    rows = []
    for method in methods:
        row = {"method": method, "arr_overall": all_arr[method], "far_overall": all_far[method],
               "n": len(all_results[method])}
        for t_ in types:
            row[f"arr_{t_}"] = all_arr_by_type[method].get(t_, None)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"arr_overall": all_arr, "far_overall": all_far, "arr_by_type": all_arr_by_type}, f, indent=2)

    # ── False-acceptance examples for manual inspection / manuscript ──────
    # (is_refusal here is already the corrected value from run_method)
    fa_rows = []
    corrected_rows = []
    for method in methods:
        for r in all_results[method]:
            if r.get("is_refusal_raw") != r.get("is_refusal"):
                corrected_rows.append({
                    "method": method, "query_id": r["query_id"], "unans_type": r["unans_type"],
                    "question": r["query"], "answer": r.get("answer", ""),
                    "raw_is_refusal": r.get("is_refusal_raw"), "corrected_is_refusal": r.get("is_refusal"),
                })
            if not r.get("is_refusal", False):
                fa_rows.append({
                    "method": method,
                    "query_id": r["query_id"],
                    "unans_type": r["unans_type"],
                    "question": r["query"],
                    "reason_unanswerable": r["reason"],
                    "answer": r.get("answer", ""),
                })
    fa_df = pd.DataFrame(fa_rows)
    fa_df.to_csv(RESULTS_DIR / "false_acceptances.csv", index=False, encoding="utf-8-sig")

    # Still-remaining false acceptances after the extra phrase pass are the
    # real candidates for a "manual review needed" pass — some may yet be
    # genuine refusals phrased in a way neither list catches. Flag rather
    # than silently trust the automated FAR/ARR numbers.
    if corrected_rows:
        corr_df = pd.DataFrame(corrected_rows)
        corr_df.to_csv(RESULTS_DIR / "refusal_detection_corrections.csv", index=False, encoding="utf-8-sig")
        logger.info(f"  {len(corrected_rows)} case(s) where the extra refusal-phrase pass changed the "
                    f"is_refusal label -> results/refusal_detection_corrections.csv (audit trail)")

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY (ARR = Appropriate Refusal Rate — higher is better; FAR = 1-ARR)")
    logger.info(summary_df.to_string(index=False))
    logger.info(f"\n✓ Results dir: {RESULTS_DIR}")
    logger.info(f"✓ {len(fa_df)} false-acceptance cases logged to false_acceptances.csv for manual review")


if __name__ == "__main__":
    main()
