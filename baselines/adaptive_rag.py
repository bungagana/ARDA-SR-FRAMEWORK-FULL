"""Baseline: Adaptive-RAG (complexity-based strategy routing)."""

import time
import logging
from typing import Dict

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)


class AdaptiveRAGPipeline(BasePipeline):
    """
    Adaptive-RAG: classifies query complexity and routes to different strategies.
    This same-backbone reimplementation uses prompted classification rather than
    the original trained smaller classifier.
    Reference: Jeong et al. 2024, NAACL.
    """

    name = "adaptive_rag"

    COMPLEXITY_PROMPT = """\
Classify the complexity of this question about Indonesian transmigration.
Return ONLY one word: "SIMPLE", "MODERATE", or "COMPLEX".

- SIMPLE: can be answered directly without retrieval
- MODERATE: requires a single retrieval step
- COMPLEX: requires multi-step retrieval, multi-hop inference, or policy analysis

Question: {query}

Classification:"""

    SIMPLE_PROMPT = """\
Answer this simple factual question about Indonesian transmigration.
Question: {query}
Answer:"""

    COMPLEX_PROMPT = """\
Answer this complex question requiring multi-step reasoning about Indonesian transmigration.
Use the retrieved evidence below.
Question: {query}

Evidence:
{evidence}

Detailed Answer:"""

    ITERATIVE_PROMPT = """\
Answer this complex question using the accumulated evidence. If more evidence is needed,
write "RETRIEVE: <specific query>". Otherwise write "ANSWER: <final answer>".

Question: {query}
Evidence so far:
{evidence}

Response:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            # Step 1: Classify complexity
            complexity_raw = self.client.generate(
                self.COMPLEXITY_PROMPT.format(query=query), max_tokens=10
            ).upper().strip()
            if "COMPLEX" in complexity_raw:
                complexity = "COMPLEX"
            elif "MODERATE" in complexity_raw:
                complexity = "MODERATE"
            else:
                complexity = "SIMPLE"
            result["routing"] = {"complexity": complexity}

            if complexity == "COMPLEX":
                evidence = self.retriever.retrieve(query, k=k)
                for _ in range(2):
                    ev_text = self._format_evidence(evidence)
                    step = self.client.generate(
                        self.ITERATIVE_PROMPT.format(query=query, evidence=ev_text[:1800]),
                        max_tokens=512,
                    )
                    if "RETRIEVE:" not in step.upper():
                        answer = step.split("ANSWER:", 1)[-1].strip() if "ANSWER:" in step else step
                        break
                    sub_query = step.split("RETRIEVE:", 1)[-1].splitlines()[0].strip()
                    evidence.extend(self.retriever.retrieve(sub_query or query, k=2))
                else:
                    ev_text = self._format_evidence(evidence)
                    answer = self.client.generate(
                        self.COMPLEX_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                    )
                result["evidence"] = evidence[:k]
            elif complexity == "MODERATE":
                evidence = self.retriever.retrieve(query, k=k)
                result["evidence"] = evidence
                ev_text = self._format_evidence(evidence)
                answer = self.client.generate(
                    self.COMPLEX_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                )
            else:
                answer = self.client.generate(
                    self.SIMPLE_PROMPT.format(query=query), max_tokens=512
                )

            result["answer"]     = answer
            result["is_refusal"] = self._is_refusal(answer)
            # Tool accuracy: was routing decision correct? (estimated by whether evidence helped)
            result["tool_decision"] = complexity
        except Exception as exc:
            logger.error(f"Adaptive-RAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
