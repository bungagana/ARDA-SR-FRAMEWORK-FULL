"""
ARDA-SR end-to-end inference pipeline.
Implements Algorithm 1 from the paper.

(q, D) → a*  via:
  1. AQR  — mode routing
  2. Retrieval — evidence construction (when required)
  3. DDA  — dual-draft arbitration (non-policy queries)
  4. SR   — policy-scenario reasoning (m4 queries)
"""

import logging
import time
from typing import Dict, List, Optional

from utils.llm_client import GeminiClient
from utils.kb_builder import KnowledgeBase
from arda_sr.aqr import AQR
from arda_sr.dda import DDA
from arda_sr.sr import SR
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = {"m2", "m3", "m4"}


class ARDASRPipeline:
    """
    Full ARDA-SR inference pipeline.

    Usage:
        pipeline = ARDASRPipeline(kb)
        result   = pipeline.run(query, reference_answer="...")
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        client: GeminiClient | None = None,
        betas: Dict | None = None,
        # ablation flags
        use_aqr: bool = True,
        use_dda: bool = True,
        use_sr:  bool = True,
        use_hybrid_retrieval: bool = True,
    ):
        self.client    = client or GeminiClient()
        self.kb        = kb
        self.retriever = HybridRetriever(kb)
        self.aqr      = AQR(self.client) if use_aqr else None
        self.dda      = DDA(self.client, betas=betas) if use_dda else None
        self.sr       = SR(self.client)  if use_sr  else None
        self.use_hybrid = use_hybrid_retrieval

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        """
        Run the full ARDA-SR pipeline.
        Returns a result dict ready for evaluation.
        """
        t_start = time.time()
        result = {
            "query":      query,
            "reference":  reference_answer,
            "method":     "arda_sr",
            "mode":       None,
            "answer":     "",
            "evidence":   [],
            "routing":    {},
            "dda_info":  {},
            "sr_info":   {},
            "is_refusal": False,
            "latency_s":  0.0,
        }

        # ── Step 1: AQR routing ──────────────────────────────────────────
        if self.aqr is not None:
            routing = self.aqr.classify(query)
        else:
            # Ablation: no AQR → default to m2 (always retrieve)
            routing = {"mode": "m2", "mode_probs": {}, "entropy": 0.0,
                       "features": {}, "hybrid_path": False, "reasoning": "AQR disabled"}

        mode   = routing["mode"]
        result["mode"]    = mode
        result["routing"] = routing

        # ── Step 2: Retrieval (when required) ────────────────────────────
        evidence: List[Dict] = []
        if mode in RETRIEVAL_MODES or routing.get("hybrid_path"):
            meta_filter = HybridRetriever.extract_metadata_from_query(query)
            evidence = self.retriever.retrieve(query, k=k, metadata_filter=meta_filter or None)
            result["evidence"] = evidence

        result["hit_at_k"] = len(evidence) > 0

        # ── Step 3a: SR for policy-scenario queries ──────────────────────
        if mode == "m4" and self.sr is not None:
            sr_out = self.sr.reason(query, evidence)
            result["answer"]     = sr_out["answer"]
            result["sr_info"]   = sr_out
            result["is_refusal"] = not bool(sr_out["answer"].strip())

        # ── Step 3b: DDA for all other queries ───────────────────────────
        else:
            if self.dda is not None:
                dda_out = self.dda.arbitrate(query, evidence, reference_answer)
                result["answer"]     = dda_out["answer"]
                result["dda_info"]  = dda_out
                result["is_refusal"] = dda_out["is_refusal"]
            else:
                # Ablation: no DDA → simple retrieval-grounded generation
                result["answer"]     = self._simple_generate(query, evidence)
                result["is_refusal"] = False

        result["latency_s"] = round(time.time() - t_start, 3)
        return result

    def _simple_generate(self, query: str, evidence: List[Dict]) -> str:
        """Fallback when DDA is disabled (ablation V0/V1/V2)."""
        if evidence:
            ev_text = "\n\n".join(
                f"[{i+1}] {e.get('text','')[:400]}" for i, e in enumerate(evidence)
            )
            prompt = f"Answer the question based on the evidence.\n\nQuestion: {query}\n\nEvidence:\n{ev_text}\n\nAnswer:"
        else:
            prompt = f"Answer the following question:\n\nQuestion: {query}\n\nAnswer:"
        try:
            return self.client.generate(prompt, max_tokens=512)
        except Exception:
            return ""
