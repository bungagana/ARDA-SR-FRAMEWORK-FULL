"""
Step 2.5: Derive should_be_answerable from real evidence in the corpus.

Implements the paper's Section 2.4.1 "Stage 1" methodology: cosine-similarity
embedding between each reference_answer and its cited source_docs chunk(s) in
the real knowledge base. This is a deterministic, reproducible signal computed
against the actual corpus — not a hardcoded flag and not an LLM guess.

Rule:
  - DK (no source_docs): should_be_answerable = True by definition
    (mode m1 — direct conceptual answering, doesn't require retrieval).
  - FR/CR/AR/PS (has source_docs): should_be_answerable = True if
    cos(embed(reference_answer), embed(best-matching source chunk)) >= tau,
    else False (flagged as ungrounded — likely a generation artifact from
    QA creation, candidate for human review or regeneration).

This is Stage 1 only. Stage 2 (3 real human annotators, per the paper) should
still review — especially the items this script flags False — and can
override this label. Wire that override in via merge_human_validation.py.

Run: python verify_answerability.py [--threshold 0.45]
Reads:  data/qa_dataset.json (no should_be_answerable field)
Writes: data/qa_dataset.json (+ should_be_answerable, evidence_similarity)
        data/answerability_report.json (per-category summary, flagged items)
"""

import argparse
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config import DATA_DIR
from utils.kb_builder import KnowledgeBase


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.45,
                         help="Min cosine similarity to consider an answer grounded in its cited source_docs.")
    args = parser.parse_args()
    tau = args.threshold

    qa_path = DATA_DIR / "qa_dataset.json"
    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    logger.info(f"Loaded {len(qa_data)} QA pairs from {qa_path}")

    kb = KnowledgeBase().load()

    # Index chunks by filename for fast lookup
    chunks_by_file = defaultdict(list)
    for c in kb.chunks:
        chunks_by_file[c.get("filename", "")].append(c)
    logger.info(f"KB: {len(kb.chunks)} chunks across {len(chunks_by_file)} files")

    n_dk = 0
    n_grounded = 0
    n_ungrounded = 0
    n_no_source_match = 0
    flagged = []

    for qa in qa_data:
        cat = qa.get("category")
        source_docs = qa.get("source_docs") or []

        if cat == "DK" and not source_docs:
            qa["should_be_answerable"] = True
            qa["evidence_similarity"] = None   # not applicable — no retrieval required for m1
            n_dk += 1
            continue

        # Gather candidate chunks from the cited source_docs
        candidate_chunks = []
        for fn in source_docs:
            candidate_chunks.extend(chunks_by_file.get(fn, []))

        if not candidate_chunks:
            # Cited file(s) not found in KB at all — can't verify grounding
            qa["should_be_answerable"] = False
            qa["evidence_similarity"] = 0.0
            n_no_source_match += 1
            flagged.append({**qa, "reason": "source_docs not found in KB"})
            continue

        ans_vec = kb.embed_query(qa.get("reference_answer", ""))
        best_sim = 0.0
        for c in candidate_chunks:
            chunk_vec = kb.embed_query(c.get("text", "")[:2000])
            sim = cosine(ans_vec, chunk_vec)
            best_sim = max(best_sim, sim)

        qa["evidence_similarity"] = round(best_sim, 4)
        if best_sim >= tau:
            qa["should_be_answerable"] = True
            n_grounded += 1
        else:
            qa["should_be_answerable"] = False
            n_ungrounded += 1
            flagged.append({**qa, "reason": f"similarity {best_sim:.3f} < threshold {tau}"})

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    import pandas as pd
    pd.DataFrame(qa_data).to_csv(DATA_DIR / "qa_dataset.csv", index=False, encoding="utf-8")

    by_cat_answerable = Counter((q["category"], q["should_be_answerable"]) for q in qa_data)
    report = {
        "threshold": tau,
        "n_total": len(qa_data),
        "n_dk_by_definition": n_dk,
        "n_grounded_by_similarity": n_grounded,
        "n_ungrounded_flagged": n_ungrounded,
        "n_source_docs_not_in_kb": n_no_source_match,
        "by_category": {
            cat: {
                "answerable_true":  by_cat_answerable.get((cat, True), 0),
                "answerable_false": by_cat_answerable.get((cat, False), 0),
            }
            for cat in sorted(set(q["category"] for q in qa_data))
        },
        "flagged_items": [
            {"query_id": f["query_id"], "category": f["category"], "reason": f["reason"]}
            for f in flagged
        ],
    }
    with open(DATA_DIR / "answerability_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info(f"DK (answerable by definition):        {n_dk}")
    logger.info(f"Grounded (similarity >= {tau}):        {n_grounded}")
    logger.info(f"Flagged ungrounded (similarity < {tau}): {n_ungrounded}")
    logger.info(f"Source docs missing from KB:           {n_no_source_match}")
    logger.info(f"\n✓ Updated: {qa_path}")
    logger.info(f"✓ Report:  {DATA_DIR / 'answerability_report.json'} ({len(flagged)} items flagged for human review)")


if __name__ == "__main__":
    main()
