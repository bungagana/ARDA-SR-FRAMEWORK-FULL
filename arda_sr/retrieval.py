"""
Hybrid retrieval: FAISS dense search + BM25 sparse search + metadata filtering.
s(e|q) = α · cos(z_e, z_q) + (1-α) · BM25(e, q)
"""

import re
import logging
from typing import List, Dict, Optional

import numpy as np

from config import TOP_K, HYBRID_ALPHA
from utils.kb_builder import KnowledgeBase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieve Top-K evidence chunks using hybrid dense+sparse scoring with metadata filtering."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Returns top-k chunk dicts with added 'hybrid_score' field.
        metadata_filter: dict of {field: value} that chunks must match (subset).
        """
        chunks = self.kb.chunks

        # ── 1. Metadata filtering ──────────────────────────────────────────
        if metadata_filter:
            candidate_ids = [
                i for i, c in enumerate(chunks)
                if self._matches(c, metadata_filter)
            ]
        else:
            candidate_ids = list(range(len(chunks)))

        if not candidate_ids:
            candidate_ids = list(range(len(chunks)))   # fallback: no filter

        # ── 2. Dense scores (cosine via FAISS inner product on unit vectors) ─
        query_vec = self.kb.embed_query(query)
        # Use FAISS for dense retrieval over the full index
        q_vec_full = self.kb.embed_query(query)
        # Retrieve top-K*5 candidates from FAISS, then re-score with BM25
        n_faiss = min(len(chunks), max(k * 5, 50))
        faiss_scores_arr, top_indices = self.kb._faiss.search(q_vec_full, n_faiss)
        faiss_scores = {}
        for rank, idx in enumerate(top_indices[0]):
            if idx < 0:
                continue
            # faiss_scores_arr contains inner products (cosine when vectors are normalised)
            faiss_scores[idx] = float(faiss_scores_arr[0][rank])

        # ── 3. BM25 scores ─────────────────────────────────────────────────
        bm25_scores_all = self.kb.bm25_scores(query)
        max_bm25 = bm25_scores_all.max() or 1.0
        bm25_norm = bm25_scores_all / max_bm25

        # ── 4. Hybrid score (only for candidates) ─────────────────────────
        scored = []
        for i in candidate_ids:
            dense = faiss_scores.get(i, 0.0)
            bm25 = float(bm25_norm[i])
            hybrid = HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * bm25
            scored.append((i, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        results = []
        for idx, score in top_k:
            c = dict(chunks[idx])
            c["hybrid_score"] = round(score, 4)
            results.append(c)
        return results

    @staticmethod
    def _matches(chunk: Dict, filt: Dict) -> bool:
        for key, val in filt.items():
            chunk_val = chunk.get(key)
            if chunk_val is None:
                continue
            if isinstance(chunk_val, list):
                if val not in chunk_val:
                    return False
            elif str(chunk_val).lower() != str(val).lower():
                return False
        return True

    @staticmethod
    def extract_metadata_from_query(query: str) -> Dict:
        """Heuristically extract filter metadata from query text."""
        filt = {}
        provinces = [
            "Sulawesi Tenggara", "Sulawesi Tengah", "Sulawesi Barat",
            "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
            "Papua", "Papua Barat", "Maluku", "Maluku Utara",
            "Sumatera Selatan", "Nusa Tenggara Timur",
        ]
        for p in provinces:
            if p.lower() in query.lower():
                filt["province"] = p
                break

        comm_map = ["padi", "jagung", "kopi", "kakao", "kelapa sawit", "karet", "sapi"]
        for c in comm_map:
            if c in query.lower():
                filt["commodities"] = c
                break

        m = re.search(r"\b(20\d{2})\b", query)
        if m:
            filt["year"] = m.group(1)

        if re.search(r"pp\s+nomor|peraturan pemerintah", query, re.I):
            filt["regulation_type"] = "PP"
        elif re.search(r"permen|peraturan menteri", query, re.I):
            filt["regulation_type"] = "Permen"

        return filt
