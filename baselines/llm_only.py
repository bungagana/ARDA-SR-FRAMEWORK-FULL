"""Baseline: LLM-Only (no retrieval, pure parametric knowledge)."""

import time
import logging
from typing import Dict

from baselines.base import BasePipeline
from config import TOP_K

logger = logging.getLogger(__name__)


class LLMOnlyPipeline(BasePipeline):
    """
    LLM-Only baseline.
    Answers questions using only the LLM's parametric knowledge.
    No retrieval, no external documents.
    Reference: Abdullahi et al. (2026) — LLM hallucination survey.
    """

    name = "llm_only"

    PROMPT = """\
You are a knowledgeable assistant for the Indonesian transmigration domain.
Answer the following question concisely and accurately using your knowledge.
If you are uncertain, provide your best answer but note the uncertainty.

Question: {query}

Answer:"""

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            answer = self.client.generate(self.PROMPT.format(query=query), max_tokens=512)
            result["answer"]     = answer
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"LLM-Only failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
