"""
Step 2: Generate QA Dataset — Claude Haiku edition
================================================================================
Generates the QA ground-truth dataset using claude-haiku-4-5 via the official
`anthropic` SDK, per the methodology described in Section 2.4.1. No LLM
validator, no annotator staging — this script only generates. There is no
should_be_answerable field in the output: that label isn't known at
generation time, so it's simply omitted rather than guessed or hardcoded.
Human validation (whatever protocol you run) is a separate step you do on
this output afterward.

Run: python 02_generate_qa_claude.py
  --cats CR AR PS  : only (re)generate these categories
Output: data/qa_dataset.json, data/qa_dataset.csv, data/qa_generation_stats.json
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qa_generation.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import (
    QA_CATEGORIES, QA_TARGET_PER_CATEGORY,
    QA_CATEGORY_DESC, DATA_DIR, KB_DIR, RANDOM_SEED
)
from utils.claude_client import ClaudeClient
from utils.kb_builder import KnowledgeBase

random.seed(RANDOM_SEED)

# ── Per-category generation prompts ────────────────────────────────────────

DK_PROMPT = """\
Generate {n} conceptual question-answer pairs about Indonesian transmigration.
These questions must be about concepts and policy frameworks that are ANSWERABLE FROM DOMAIN KNOWLEDGE
without looking up specific statistics or area-level facts.

Topic context (use this ONLY as inspiration for domain concepts — do NOT ask about specific numbers,
area sizes, or statistics that would require document lookup):
---
{document_text}
---

Generate questions about transmigration concepts, program classifications, policy goals,
eligibility criteria (K-1/K-2/K-3), and general frameworks that appear in or relate to the above context.

Return ONLY a JSON array — no explanation, no markdown:
[
  {{"question": "...", "reference_answer": "...", "category": "DK"}},
  ...
]"""

FR_PROMPT = """\
Generate {n} factual question-answer pairs that require looking up specific data from this document.

Document:
{document_text}

Questions should ask about specific facts: numbers, area names, productivity data, facility counts, etc.
Each answer must be directly verifiable from the document text.

Return ONLY a JSON array:
[
  {{"question": "...", "reference_answer": "...", "category": "FR", "source_docs": ["{filename}"]}},
  ...
]"""

CR_PROMPT = """\
Generate {n} complex multi-hop questions that REQUIRE information from BOTH of these documents.

Document A ({filename_a}):
{text_a}

Document B ({filename_b}):
{text_b}

Questions must require combining facts from BOTH documents. Examples:
- Compare commodity productivity between the two areas
- Which area has better infrastructure given [condition from doc A] and [fact from doc B]?

Return ONLY a JSON array:
[
  {{"question": "...", "reference_answer": "...", "category": "CR",
    "source_docs": ["{filename_a}", "{filename_b}"]}},
  ...
]"""

AR_PROMPT = """\
Generate {n} AMBIGUOUS questions about Indonesian transmigration where it is genuinely unclear
whether to answer from general knowledge or specific documents.

Context document:
{document_text}

Ambiguous examples:
- Questions where the answer might depend on a specific region OR could be general
- Questions about general procedures that may vary by local regulation
- Questions mixing policy concepts with specific area data

Return ONLY a JSON array:
[
  {{"question": "...", "reference_answer": "...", "category": "AR", "source_docs": ["{filename}"]}},
  ...
]"""

PS_PROMPT = """\
Generate {n} POLICY SCENARIO questions that require multi-alternative decision analysis.

Context:
{document_text}

Questions must:
1. Present a decision/policy problem
2. Require evaluating multiple alternatives/scenarios
3. Consider risks, benefits, and trade-offs
4. Be relevant to government/transmigration context

Example: "If the government wants to X in area Y, what intervention should be prioritized? Analyze trade-offs."

Return ONLY a JSON array:
[
  {{"question": "...", "reference_answer": "...", "category": "PS", "source_docs": ["{filename}"]}},
  ...
]"""


BATCH = 15        # QA pairs per API call — safe within ~4000 output tokens
QA_MAX_TOKENS = 8192   # generous output limit for list generation


class QAGenerator:
    def __init__(self, kb: KnowledgeBase, client: ClaudeClient):
        self.kb = kb
        self.client = client
        self.chunks = kb.chunks

    def _regional_chunks(self, n: int = 3) -> List[Dict]:
        regional = [c for c in self.chunks if c.get("category") == "regional_profile"]
        return random.sample(regional, min(n, len(regional)))

    def _batched_json(self, prompt: str) -> List[Dict]:
        """Call generate_json with QA_MAX_TOKENS; always returns a list."""
        try:
            raw = self.client.generate_json(prompt, max_tokens=QA_MAX_TOKENS)
            return raw if isinstance(raw, list) else []
        except Exception as exc:
            logger.warning(f"Batch generation failed: {exc}")
            return []

    def _while_generate(self, pool: List[Dict], prompt_fn, n: int) -> List[Dict]:
        """
        Generic while-loop generator: keep sampling chunks from pool until
        we have n items. Handles Claude returning fewer items than requested.
        """
        results: List[Dict] = []
        max_attempts = n * 6   # hard cap on API calls
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            chunk = random.choice(pool)
            want = min(BATCH, n - len(results))
            items = self._batched_json(prompt_fn(chunk, want))
            results.extend(items)
            logger.debug(f"    [{attempts}] got {len(items)} items, total {len(results)}/{n}")
        if len(results) < n:
            logger.warning(f"  Only generated {len(results)}/{n} items after {attempts} attempts")
        return results[:n]

    def generate_dk(self, n: int) -> List[Dict]:
        regional = [c for c in self.chunks if c.get("category") == "regional_profile"]
        def prompt_fn(chunk, want):
            return DK_PROMPT.format(n=want, document_text=chunk["text"][:1200])
        return self._while_generate(regional, prompt_fn, n)

    def generate_fr(self, n: int) -> List[Dict]:
        regional = [c for c in self.chunks if c.get("category") == "regional_profile"]
        def prompt_fn(chunk, want):
            return FR_PROMPT.format(n=want, document_text=chunk["text"][:1500],
                                    filename=chunk["filename"])
        return self._while_generate(regional, prompt_fn, n)

    def generate_cr(self, n: int) -> List[Dict]:
        regional = [c for c in self.chunks if c.get("category") == "regional_profile"]
        if len(regional) < 2:
            return []
        results: List[Dict] = []
        seen_pairs: set = set()
        max_attempts = n * 6
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            chunk_a, chunk_b = random.sample(regional, 2)
            if chunk_a["filename"] == chunk_b["filename"]:
                continue
            key = tuple(sorted([chunk_a["filename"], chunk_b["filename"]]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            want = min(BATCH // 2, n - len(results))
            items = self._batched_json(
                CR_PROMPT.format(
                    n=want,
                    filename_a=chunk_a["filename"], text_a=chunk_a["text"][:800],
                    filename_b=chunk_b["filename"], text_b=chunk_b["text"][:800],
                )
            )
            results.extend(items)
        return results[:n]

    def generate_ar(self, n: int) -> List[Dict]:
        regional = [c for c in self.chunks if c.get("category") == "regional_profile"]
        def prompt_fn(chunk, want):
            return AR_PROMPT.format(n=want, document_text=chunk["text"][:1200],
                                    filename=chunk["filename"])
        return self._while_generate(regional, prompt_fn, n)

    def generate_ps(self, n: int) -> List[Dict]:
        policy_chunks = [c for c in self.chunks
                         if c.get("category") in ("regional_profile", "regulations_policies")]
        def prompt_fn(chunk, want):
            return PS_PROMPT.format(n=want, document_text=chunk["text"][:1200],
                                    filename=chunk["filename"])
        return self._while_generate(policy_chunks, prompt_fn, n)


def assign_ids(qa_pairs: List[Dict], start_id: int = 0) -> List[Dict]:
    for i, qa in enumerate(qa_pairs):
        qa["query_id"] = f"qa_{start_id + i:04d}"
        # No should_be_answerable field here — it isn't known at generation
        # time. Whatever validation you run afterward should add it based on
        # real evidence/annotation, not have this script guess it.
        if not isinstance(qa.get("source_docs"), list):
            qa["source_docs"] = []
    return qa_pairs


CHECKPOINT_PATH = DATA_DIR / "qa_checkpoint.json"


def _load_checkpoint() -> Dict:
    """Load existing checkpoint if present. Returns {cat: {accepted: [...], kappa: float, stats: {...}}}"""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            ckpt = json.load(f)
        logger.info(f"  Resuming from checkpoint: {list(ckpt.keys())} already done")
        return ckpt
    return {}


def _save_checkpoint(checkpoint: Dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats", nargs="+", metavar="CAT",
                        help="Only generate for these categories (e.g. --cats CR AR PS)")
    args = parser.parse_args()

    cats_to_run = args.cats if args.cats else QA_CATEGORIES
    # Validate category names
    invalid = [c for c in cats_to_run if c not in QA_CATEGORIES]
    if invalid:
        logger.error(f"Unknown categories: {invalid}. Valid: {QA_CATEGORIES}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ARDA-SR Step 2: QA Dataset Generation (Claude, no validator)")
    logger.info(f"  Categories: {cats_to_run}")
    logger.info("=" * 60)

    # Generation uses Claude Haiku (claude-haiku-4-5), per Section 2.4.1.
    client    = ClaudeClient()
    kb        = KnowledgeBase().load()
    generator = QAGenerator(kb, client)

    # ── Load existing dataset to merge into (partial run support) ─────────
    existing_path = DATA_DIR / "qa_dataset.json"
    if existing_path.exists() and args.cats:
        with open(existing_path, encoding="utf-8") as f:
            existing = json.load(f)
        # Remove any existing entries for cats_to_run (will be regenerated)
        all_candidates: List[Dict] = [q for q in existing if q.get("category") not in cats_to_run]
        logger.info(f"  Loaded {len(all_candidates)} existing entries (keeping non-{cats_to_run} categories)")
    else:
        all_candidates: List[Dict] = []

    # ── Load checkpoint (resume support) ──────────────────────────────────
    checkpoint = _load_checkpoint()
    stats: Dict = {}
    global_id = len(all_candidates)

    # Reconstruct state from checkpoint for cats_to_run
    for cat in cats_to_run:
        if cat in checkpoint:
            cat_candidates = checkpoint[cat]["candidates"]
            all_candidates.extend(cat_candidates)
            stats[cat] = checkpoint[cat]["stats"]
            global_id += len(cat_candidates)
            logger.info(f"  Skipping {cat}: already generated ({len(cat_candidates)} candidates)")

    gen_methods = {
        "DK": generator.generate_dk,
        "FR": generator.generate_fr,
        "CR": generator.generate_cr,
        "AR": generator.generate_ar,
        "PS": generator.generate_ps,
    }

    for cat in cats_to_run:
        if cat in checkpoint:
            continue   # already done

        logger.info(f"\n── Category: {cat} ({QA_CATEGORY_DESC[cat]}) ──")
        # No filtering step anymore, so generate exactly the target count
        # directly instead of over-generating a buffer for a validator to trim.
        generate_n = QA_TARGET_PER_CATEGORY

        logger.info(f"  Generating {generate_n} QA pairs...")
        candidates = gen_methods[cat](generate_n)

        for qa in candidates:
            qa["category"] = cat

        candidates = assign_ids(candidates, global_id)
        global_id += len(candidates)

        all_candidates.extend(candidates)
        cat_stats = {"generated": len(candidates)}

        # ── Save checkpoint after each category ────────────────────────────
        checkpoint[cat] = {"candidates": candidates, "stats": cat_stats}
        _save_checkpoint(checkpoint)
        logger.info(f"  Checkpoint saved → {CHECKPOINT_PATH}")

    # ── Final outputs ─────────────────────────────────────────────────────
    with open(DATA_DIR / "qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(all_candidates)
    df.to_csv(DATA_DIR / "qa_dataset.csv", index=False, encoding="utf-8")

    with open(DATA_DIR / "qa_generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── Delete checkpoint (run complete) ───────────────────────────────────
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("QA Generation Summary:")
    logger.info(f"  TOTAL: {len(all_candidates)} QA pairs")
    from collections import Counter
    cat_counts = Counter(q.get("category") for q in all_candidates)
    for cat in QA_CATEGORIES:
        logger.info(f"  {cat}: {cat_counts.get(cat, 0)} pairs")
    logger.info(f"\n✓ Dataset saved: {DATA_DIR / 'qa_dataset.json'}")
    logger.info("  (no should_be_answerable field — add it via your own validation step)")


if __name__ == "__main__":
    main()
