"""Build and persist the FAISS + BM25 knowledge base from document chunks."""

import json
import math
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from config import (
    KB_DIR, EMBED_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


# ── Chunker ────────────────────────────────────────────────────────────────

def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Recursive character-level splitter respecting paragraph boundaries."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        parts = text.split(sep) if sep else list(text)
        chunks, buf = [], ""
        for part in parts:
            piece = (buf + sep + part).strip() if buf else part.strip()
            if len(piece) <= size:
                buf = piece
            else:
                if buf:
                    chunks.append(buf)
                # If part itself is too long, recurse with next separator
                buf = part.strip()
        if buf:
            chunks.append(buf)

        # Merge short chunks & enforce overlap
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) + len(c) + 1 <= size:
                merged[-1] = merged[-1] + (" " if sep == " " else "\n") + c
            else:
                merged.append(c)

        if all(len(c) <= size for c in merged):
            # Apply overlap: prepend tail of previous chunk
            result = []
            for i, c in enumerate(merged):
                if i > 0 and overlap > 0:
                    tail = merged[i - 1][-overlap:]
                    c = tail + " " + c
                result.append(c[:size])
            return [c for c in result if c.strip()]

    return [text[:size]]


# ── Knowledge Base Builder ─────────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """Builds a dual-index KB: FAISS (dense) + BM25 (sparse)."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(exist_ok=True)
        self._embed_model: SentenceTransformer | None = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, documents: List[Dict]) -> None:
        """
        Build KB from document dicts (with keys: filename, category, doc_type, text).
        Saves: kb/chunks.json, kb/faiss_index/, kb/bm25_index.pkl
        """
        logger.info(f"Building KB from {len(documents)} documents...")
        chunks = self._make_chunks(documents)
        logger.info(f"  -> {len(chunks)} total chunks")

        self._build_faiss(chunks)
        self._build_bm25(chunks)
        self._save_chunks(chunks)
        logger.info("KB build complete.")

    def _make_chunks(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            doc_chunks = _split_text(doc["text"])
            for idx, chunk_text in enumerate(doc_chunks):
                meta = self._extract_metadata(doc["text"], doc["filename"])
                chunks.append({
                    "chunk_id":    len(chunks),
                    "filename":    doc["filename"],
                    "category":    doc["category"],
                    "doc_type":    doc["doc_type"],
                    "chunk_index": idx,
                    "chunk_size":  len(chunk_text),
                    "text":        chunk_text,
                    **meta,
                })
        return chunks

    def _extract_metadata(self, text: str, filename: str) -> dict:
        meta = {}
        # Region
        m = re.search(r"KAWASAN\s+(?:TRANSMIGRASI\s+)?([A-Z][A-Z\s\-]+?)(?:\s*[–\-–]|\s*,|\n)", text)
        if m:
            meta["region"] = m.group(1).strip().title()

        # Province
        provinces = ["Sulawesi Tenggara", "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Utara",
                     "Sulawesi Selatan", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
                     "Kalimantan Timur", "Papua", "Papua Barat", "Maluku", "Maluku Utara",
                     "Sumatera Selatan", "Sumatera Utara", "Nusa Tenggara Timur", "Nusa Tenggara Barat"]
        for p in provinces:
            if p.lower() in text.lower():
                meta["province"] = p
                break

        # Year
        m = re.search(r"\b(20\d{2})\b", text)
        if m:
            meta["year"] = m.group(1)

        # Commodities
        comm_map = {
            "padi": "rice", "jagung": "corn", "kopi": "coffee",
            "kakao": "cocoa", "kelapa sawit": "palm_oil",
            "karet": "rubber", "sapi": "cattle", "udang": "shrimp",
        }
        found = [k for k in comm_map if k in text.lower()]
        if found:
            meta["commodities"] = found

        # Regulation
        if re.search(r"pp\s+nomor|peraturan pemerintah", text, re.I):
            meta["regulation_type"] = "PP"
        elif re.search(r"permen|peraturan menteri", text, re.I):
            meta["regulation_type"] = "Permen"
        elif re.search(r"uu\s+nomor|undang-undang", text, re.I):
            meta["regulation_type"] = "UU"

        return meta

    # ── FAISS ──────────────────────────────────────────────────────────────

    def _build_faiss(self, chunks: List[Dict]) -> None:
        texts = [c["text"] for c in chunks]
        logger.info(f"  Embedding {len(texts)} chunks (batch=64)...")
        vectors = self.embed_model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(EMBED_DIM)   # Inner product == cosine when normalized
        index.add(vectors)

        faiss_path = self.kb_dir / "faiss_index"
        faiss_path.mkdir(exist_ok=True)
        faiss.write_index(index, str(faiss_path / "index.bin"))
        np.save(str(faiss_path / "vectors.npy"), vectors)
        logger.info(f"  FAISS index saved ({index.ntotal} vectors).")

    # ── BM25 ───────────────────────────────────────────────────────────────

    def _build_bm25(self, chunks: List[Dict]) -> None:
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with open(self.kb_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        logger.info("  BM25 index saved.")

    def _save_chunks(self, chunks: List[Dict]) -> None:
        with open(self.kb_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"  chunks.json saved ({len(chunks)} entries).")


# ── KB Loader (used at inference time) ────────────────────────────────────

class KnowledgeBase:
    """Load and query a pre-built KB."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self._chunks: List[Dict] | None = None
        self._faiss: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None
        self._embed: SentenceTransformer | None = None

    def load(self) -> "KnowledgeBase":
        with open(self.kb_dir / "chunks.json", encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._faiss = faiss.read_index(str(self.kb_dir / "faiss_index" / "index.bin"))
        with open(self.kb_dir / "bm25_index.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        self._embed = SentenceTransformer(EMBED_MODEL)
        logger.info(f"KB loaded: {len(self._chunks)} chunks.")
        return self

    @property
    def chunks(self) -> List[Dict]:
        assert self._chunks is not None, "Call .load() first"
        return self._chunks

    def embed_query(self, query: str) -> np.ndarray:
        v = self._embed.encode([query], normalize_embeddings=True).astype("float32")
        return v

    def faiss_search(self, query_vec: np.ndarray, k: int) -> List[int]:
        _, indices = self._faiss.search(query_vec, k)
        return indices[0].tolist()

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self._bm25.get_scores(query.lower().split()))
