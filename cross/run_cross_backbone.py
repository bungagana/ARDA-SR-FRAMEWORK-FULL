from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR
from cross.hf_client import HFInferenceClient
from cross.ollama_client import OllamaClient

OUT_DIR = ROOT / "cross" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3-mini-4k-instruct"]

# Methods run per backbone — mirrors the rows of the paper's cross-backbone
# table (Table 7 / \label{tab:cross-backbone}): Standard RAG and Self-RAG as
# baselines, ARDA-SR as the proposed method. Order matters for logging only.
DEFAULT_METHODS = ["standard_rag", "selfrag", "arda_sr"]


def build_pipeline(method: str, kb, client, betas=None):
    """Instantiate the pipeline for `method` against the given backbone client."""
    if method == "arda_sr":
        from arda_sr.pipeline import ARDASRPipeline
        return ARDASRPipeline(kb=kb, client=client, betas=betas)
    if method == "standard_rag":
        from baselines.standard_rag import StandardRAGPipeline
        return StandardRAGPipeline(kb=kb, client=client)
    if method == "selfrag":
        from baselines.selfrag import SelfRAGPipeline
        return SelfRAGPipeline(kb=kb, client=client)
    raise ValueError(f"Unknown cross-backbone method: {method!r} (expected one of {DEFAULT_METHODS})")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "cross_backbone.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("cross_backbone")


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(".", "_")


def load_qa_dataset(smoke: bool = False, limit_per_category: int | None = None) -> list[dict]:
    qa_path = DATA_DIR / "qa_dataset.json"
    with open(qa_path, encoding="utf-8") as f:
        rows = json.load(f)

    if smoke and limit_per_category is None:
        limit_per_category = 2

    if limit_per_category:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_cat[row.get("category", "?")].append(row)
        limited = []
        for cat in sorted(by_cat):
            limited.extend(by_cat[cat][:limit_per_category])
        rows = limited

    return rows


def ckpt_path(model: str, method: str) -> Path:
    return OUT_DIR / f"{safe_model_name(model)}_{method}_ckpt.json"


def result_path(model: str, method: str) -> Path:
    return OUT_DIR / f"{safe_model_name(model)}_{method}_results.json"


def load_checkpoint(model: str, method: str) -> dict[str, dict]:
    path = ckpt_path(model, method)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["query_id"]: row for row in rows}


def save_checkpoint(model: str, method: str, done: dict[str, dict]) -> None:
    with open(ckpt_path(model, method), "w", encoding="utf-8") as f:
        json.dump(list(done.values()), f, ensure_ascii=False, indent=2)


def build_client(model: str, args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaClient(model=model, temperature=args.temperature)
    return HFInferenceClient(
        model=model,
        provider=args.hf_provider,
        temperature=args.temperature,
        timeout=args.timeout,
    )


def run_model(model: str, method: str, qa_data: list[dict], kb, args: argparse.Namespace) -> tuple[list[dict], dict]:
    from arda_sr.dda import DEFAULT_BETAS, grid_search_betas
    from evaluation.llm_judge import LLMJudge
    from evaluation.metrics import compute_all_metrics, per_category_metrics

    client = build_client(model, args)

    betas = DEFAULT_BETAS
    if method == "arda_sr" and args.grid_search:
        val_items = [
            {"query": q["question"], "reference_answer": q.get("reference_answer", "")}
            for q in qa_data[: max(5, len(qa_data) // 5)]
        ]
        betas = grid_search_betas(val_items, client)

    pipeline = build_pipeline(method, kb, client, betas=betas)

    # Reuse a finished result file if one already exists (e.g. arda_sr was run
    # in an earlier invocation) instead of re-running 1,000 queries against a
    # paid API. --force re-runs from scratch regardless.
    done: dict[str, dict] = {}
    finished_path = result_path(model, method)
    if finished_path.exists() and not args.force:
        with open(finished_path, encoding="utf-8") as f:
            done = {row["query_id"]: row for row in json.load(f)}
        logger.info(
            "[%s/%s] found existing finished results (%s rows) at %s — reusing, "
            "pass --force to re-run.",
            model, method, len(done), finished_path,
        )
    else:
        done = load_checkpoint(model, method)

    remaining = [qa for qa in qa_data if qa["query_id"] not in done]

    if done and remaining:
        logger.info("Resuming %s/%s: %s/%s completed", model, method, len(done), len(qa_data))

    for idx, qa in enumerate(remaining, 1):
        logger.info("[%s/%s] %s/%s %s", model, method, idx, len(remaining), qa["query_id"])
        row = pipeline.run(
            query=qa["question"],
            reference_answer=qa.get("reference_answer", ""),
        )
        row["query_id"] = qa["query_id"]
        row["category"] = qa.get("category", "")
        row["method"] = method
        row["backbone"] = model
        done[qa["query_id"]] = row
        save_checkpoint(model, method, done)

    id_order = {qa["query_id"]: i for i, qa in enumerate(qa_data)}
    results = sorted(done.values(), key=lambda row: id_order.get(row["query_id"], 0))

    llm_scores = None
    if args.judge:
        from utils.openai_client import GPTJudgeClient
        logger.info("[%s/%s] gpt-5.4-mini judge scoring %s answers", model, method, len(results))
        judge = LLMJudge(GPTJudgeClient())
        llm_scores = judge.judge_batch(results, show_progress=True)
        for row in results:
            qid = row.get("query_id", "")
            if qid in llm_scores:
                row.update(
                    {
                        "rel": llm_scores[qid]["rel"] / 5.0,
                        "faith": llm_scores[qid]["faith"] / 5.0,
                        "cov": llm_scores[qid]["cov"] / 5.0,
                    }
                )

    metrics = compute_all_metrics(results, llm_scores)
    metrics["per_category"] = per_category_metrics(results, llm_scores)
    metrics["backbone"] = model
    metrics["method"] = method
    metrics["provider"] = args.provider
    metrics["hf_provider"] = args.hf_provider if args.provider == "hf" else None
    metrics["n_queries"] = len(results)
    metrics["judge"] = f"same_{args.provider}_model" if args.judge else "none_auto_metrics_only"

    with open(result_path(model, method), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if ckpt_path(model, method).exists():
        ckpt_path(model, method).unlink()
    return results, metrics


def write_metrics(all_metrics: dict[str, dict]) -> None:
    """all_metrics is keyed by "{model}::{method}"."""
    with open(OUT_DIR / "cross_backbone_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    rows = []
    for key, metrics in all_metrics.items():
        model = metrics.get("backbone", key.split("::", 1)[0])
        method = metrics.get("method", key.split("::", 1)[-1])
        rows.append(
            {
                "Backbone": model,
                "Method": method,
                "N": metrics.get("n_queries"),
                "Rel": metrics.get("rel"),
                "Faith": metrics.get("faith"),
                "Cov": metrics.get("cov"),
                "Hit@5": metrics.get("hit_at_5"),
                "CtxRel": metrics.get("ctx_rel"),
                "ToolAcc": metrics.get("tool_acc"),
                "SRComp": metrics.get("sr_comp", metrics.get("pmr_comp")),
                "FRR": metrics.get("frr"),
                "FAR": metrics.get("far"),
                "Latency(s)": metrics.get("latency_s"),
                "Judge": metrics.get("judge"),
            }
        )

    import csv

    with open(OUT_DIR / "cross_backbone_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to run per backbone: standard_rag, selfrag, arda_sr "
             "(default: all three, matching Table 7's rows).",
    )
    parser.add_argument("--provider", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--hf-provider", default="auto", help="Hugging Face inference provider.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--smoke", action="store_true", help="Run two queries per category.")
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--judge", action="store_true", help="Use the same backbone model as LLM judge.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a finished *_results.json already exists for this model/method "
             "(default: reuse it and skip the API calls).",
    )
    parser.add_argument("--grid-search", action="store_true", help="Run DDAF beta search per backbone.")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in DEFAULT_METHODS:
            raise SystemExit(f"Unknown method {m!r} in --methods (expected one of {DEFAULT_METHODS})")

    qa_data = load_qa_dataset(smoke=args.smoke, limit_per_category=args.limit_per_category)
    logger.info("Loaded %s QA rows", len(qa_data))
    logger.info("Backbones: %s | Methods: %s", models, methods)

    from utils.kb_builder import KnowledgeBase

    kb = KnowledgeBase().load()

    # Merge into whatever's already recorded rather than overwriting it —
    # otherwise running e.g. --methods standard_rag,selfrag would drop the
    # arda_sr rows a previous invocation already produced.
    metrics_path = OUT_DIR / "cross_backbone_metrics.json"
    summary_csv_path = OUT_DIR / "cross_backbone_summary.csv"
    all_metrics: dict[str, dict] = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            all_metrics = json.load(f)
        logger.info("Loaded %s existing metric entries from %s", len(all_metrics), metrics_path)
    elif summary_csv_path.exists():
             import csv as csv_module
        with open(summary_csv_path, encoding="utf-8") as f:
            for csv_row in csv_module.DictReader(f):
                model_ = csv_row["Backbone"]
                method_ = csv_row.get("Method") or "arda_sr"  # old CSVs had no Method column
                key_ = f"{model_}::{method_}"
                def _num(v):
                    try:
                        return float(v) if v not in (None, "") else None
                    except ValueError:
                        return v
                all_metrics[key_] = {
                    "backbone": model_, "method": method_,
                    "n_queries": _num(csv_row.get("N")),
                    "rel": _num(csv_row.get("Rel")), "faith": _num(csv_row.get("Faith")),
                    "cov": _num(csv_row.get("Cov")), "hit_at_5": _num(csv_row.get("Hit@5")),
                    "ctx_rel": _num(csv_row.get("CtxRel")), "tool_acc": _num(csv_row.get("ToolAcc")),
                    "sr_comp": _num(csv_row.get("SRComp")), "frr": _num(csv_row.get("FRR")),
                    "far": _num(csv_row.get("FAR")), "latency_s": _num(csv_row.get("Latency(s)")),
                    "judge": csv_row.get("Judge"),
                }
        logger.info("Seeded %s existing rows from %s (no cross_backbone_metrics.json found)",
                    len(all_metrics), summary_csv_path)

    for model in models:
        for method in methods:
            key = f"{model}::{method}"
            logger.info("Running %s with %s backbone: %s", method, args.provider, model)
            start = time.time()
            try:
                _, metrics = run_model(model, method, qa_data, kb, args)
                all_metrics[key] = metrics
                logger.info(
                    "[%s/%s] done in %.1fs | FRR=%s ToolAcc=%s Lat=%s",
                    model,
                    method,
                    time.time() - start,
                    metrics.get("frr"),
                    metrics.get("tool_acc"),
                    metrics.get("latency_s"),
                )
            except Exception as exc:
                logger.exception("[%s/%s] failed: %s", model, method, exc)
                all_metrics[key] = {"backbone": model, "method": method, "error": str(exc), "n_queries": 0}

    write_metrics(all_metrics)
    logger.info("Cross-backbone outputs saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
