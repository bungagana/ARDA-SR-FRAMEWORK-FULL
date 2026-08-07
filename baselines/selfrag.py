"""Baseline: same-backbone Self-RAG reimplementation.

This is an inference-time approximation of Self-RAG behavior under the shared
Gemini backbone, not the original fine-tuned Self-RAG checkpoint.
"""

import time
import logging
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)


class SelfRAGPipeline(BasePipeline):
    """
    Same-backbone Self-RAG reimplementation: asks the shared backbone to decide
    when to retrieve and to emit reflection-like critique tokens.

    This keeps the backbone controlled across baselines, but it should not be
    interpreted as an exact reproduction of the original fine-tuned Self-RAG
    model with learned reflection tokens.
    Reference: Asai et al. 2024, ICLR.
    """

    name = "selfrag"

    RETRIEVE_DECISION_PROMPT = """\
Decide whether retrieving external documents would help answer this question.
Return ONLY JSON: {{"retrieve": true/false, "reason": "<one sentence>"}}

Question: {query}"""

    GENERATION_PROMPT = """\
Answer the question. After your answer, add critique tokens on separate lines:
[IsREL]: Is the evidence relevant? (yes/no)
[IsSUP]: Is the answer supported by evidence? (yes/no)
[IsUSE]: Is the answer useful and complete? (yes/no)

Question: {query}
{evidence_section}

Answer:"""

    REFINE_PROMPT = """\
Your previous answer was flagged as unsupported. Revise it to be better grounded.

Question: {query}
Evidence:
{evidence}
Previous Answer: {prev_answer}

Revised Answer:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            # Step 1: Decide whether to retrieve
            dec = self.client.generate_json(self.RETRIEVE_DECISION_PROMPT.format(query=query))
            should_retrieve = bool(dec.get("retrieve", True))

            if should_retrieve:
                evidence = self.retriever.retrieve(query, k=k)
                result["evidence"] = evidence
                ev_section = "Evidence:\n" + self._format_evidence(evidence)
            else:
                evidence = []
                ev_section = ""

            # Step 2: Generate with critique tokens
            prompt = self.GENERATION_PROMPT.format(
                query=query, evidence_section=ev_section
            )
            raw_answer = self.client.generate(prompt, max_tokens=768)

            # Step 3: Parse critique tokens
            is_sup = "[issup]: yes" in raw_answer.lower()
            clean_answer = raw_answer.split("[IsREL]")[0].strip()

            # Step 4: Refine if unsupported and evidence exists
            if not is_sup and evidence:
                ev_text = self._format_evidence(evidence)
                clean_answer = self.client.generate(
                    self.REFINE_PROMPT.format(
                        query=query, evidence=ev_text[:1500], prev_answer=clean_answer[:400]
                    ), max_tokens=512
                )

            result["answer"]     = clean_answer
            result["is_refusal"] = self._is_refusal(clean_answer)
            result["selfrag_retrieved"] = should_retrieve
            result["selfrag_supported"] = is_sup
        except Exception as exc:
            logger.error(f"Self-RAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
