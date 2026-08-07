"""
Step 1: Build the Knowledge Base
Run with: python 01_build_kb.py
(On Windows, set PYTHONUTF8=1 or use: python -X utf8 01_build_kb.py)
=================================
Extracts text from documents.zip, chunks, embeds, and indexes into FAISS + BM25.

Run: python 01_build_kb.py
Output: kb/faiss_index/, kb/bm25_index.pkl, kb/chunks.json
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kb_build.log"),
    ],
)
logger = logging.getLogger(__name__)

from config import DOCS_ZIP, KB_DIR
from utils.document_processor import DocumentProcessor
from utils.kb_builder import KnowledgeBaseBuilder


def main():
    logger.info("=" * 60)
    logger.info("ARDA-SR Step 1: Knowledge Base Construction")
    logger.info("=" * 60)

    if not DOCS_ZIP.exists():
        logger.error(f"documents.zip not found at: {DOCS_ZIP}")
        sys.exit(1)

    # ── Extract documents ──────────────────────────────────────────────────
    logger.info(f"Processing: {DOCS_ZIP}")
    processor = DocumentProcessor(DOCS_ZIP)
    documents = list(processor.iter_documents())
    logger.info(f"Extracted {len(documents)} documents")

    # Print distribution
    from collections import Counter
    dist = Counter(d["category"] for d in documents)
    for cat, count in sorted(dist.items()):
        logger.info(f"  {cat}: {count} docs")

    # ── Build KB ───────────────────────────────────────────────────────────
    builder = KnowledgeBaseBuilder(KB_DIR)
    builder.build(documents)

    # ── Verify ────────────────────────────────────────────────────────────
    import json
    with open(KB_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"\nKB Summary:")
    logger.info(f"  Total chunks: {len(chunks)}")
    cat_dist = Counter(c["category"] for c in chunks)
    for cat, count in sorted(cat_dist.items()):
        logger.info(f"  {cat}: {count} chunks")

    import faiss
    index = faiss.read_index(str(KB_DIR / "faiss_index" / "index.bin"))
    logger.info(f"  FAISS index: {index.ntotal} vectors, dim={index.d}")

    logger.info("\n✓ Knowledge Base built successfully!")
    logger.info(f"  Output: {KB_DIR}")


if __name__ == "__main__":
    main()
