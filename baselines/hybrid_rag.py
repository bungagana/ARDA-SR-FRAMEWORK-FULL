"""Baseline: Hybrid RAG (dense + BM25 sparse retrieval, no adaptive routing)."""

import time
import logging
from typing import Dict

import numpy as np

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)


class HybridRAGPipeline(BasePipeline):
    """
    Hybrid RAG: combines dense cosine similarity and BM25 lexical matching.
    No adaptive routing or false-refusal mitigation.
    Reference: IEEE 10707868.
    """

    name = "hybrid_rag"

    PROMPT = """\
Answer the question based on the retrieved evidence below.
Prioritize factual accuracy. If the evidence is insufficient, indicate this.

Question: {query}

Evidence:
{evidence}

Answer:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            evidence = self.retriever.retrieve(query, k=k)
            result["evidence"] = evidence

            ev_text = self._format_evidence(evidence)
            prompt  = self.PROMPT.format(query=query, evidence=ev_text)
            answer  = self.client.generate(prompt, max_tokens=512)
            result["answer"]     = answer
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"Hybrid RAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
