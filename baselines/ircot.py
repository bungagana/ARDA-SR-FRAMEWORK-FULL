"""
Baseline: IRCoT — Interleaving Retrieval with Chain-of-Thought Reasoning.

Interleaves CoT reasoning steps with retrieval to solve multi-hop questions.
Each reasoning step may trigger a targeted retrieval.
Reference: Trivedi et al. 2023, ACL (Scopus indexed).
"""

import time
import re
import logging
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K, IRCOT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


class IRCoTPipeline(BasePipeline):
    """
    IRCoT: Interleaving Retrieval + CoT for multi-hop question answering.
    At each CoT step, retrieves documents relevant to the current reasoning state.

    Fidelity note: the original IRCoT (Trivedi et al. 2023) retrieves after
    every generated CoT sentence, deterministically. This implementation
    instead lets the LLM itself decide per step whether to emit
    "RETRIEVE: <query>" or "ANSWER: <final answer>" — a minor mechanism
    difference (LLM-decided vs. always-on retrieval cadence) worth noting
    when comparing against the original paper's reported behavior.
    """

    name = "ircot"

    COT_STEP_PROMPT = """\
You are solving a multi-step question about Indonesian transmigration.
Think step by step. If you need more information, write "RETRIEVE: <specific query>".
If you have enough information to answer, write "ANSWER: <final answer>".

Original Question: {query}

Reasoning so far:
{reasoning}

Retrieved Evidence so far:
{evidence_summary}

Next reasoning step:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        reasoning_steps: List[str] = []
        evidence_all: List[Dict]   = []

        try:
            # Initial retrieval for the original question
            initial_evidence = self.retriever.retrieve(query, k=3)
            evidence_all.extend(initial_evidence)

            final_answer = ""
            for iteration in range(IRCOT_MAX_ITERATIONS):
                evidence_summary = self._summarise_evidence(evidence_all[:6])
                reasoning_so_far = "\n".join(reasoning_steps) or "(none yet)"

                step_out = self.client.generate(
                    self.COT_STEP_PROMPT.format(
                        query=query,
                        reasoning=reasoning_so_far,
                        evidence_summary=evidence_summary,
                    ),
                    max_tokens=384,
                )

                reasoning_steps.append(step_out.strip())

                # Check for ANSWER: signal
                answer_match = re.search(r"ANSWER:\s*(.+)", step_out, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    final_answer = answer_match.group(1).strip()
                    break

                # Check for RETRIEVE: signal
                retrieve_match = re.search(r"RETRIEVE:\s*(.+?)(?:\n|$)", step_out, re.IGNORECASE)
                if retrieve_match:
                    sub_query = retrieve_match.group(1).strip()
                    new_evidence = self.retriever.retrieve(sub_query, k=2)
                    evidence_all.extend(new_evidence)

            # If no explicit ANSWER found, extract from last reasoning step
            if not final_answer and reasoning_steps:
                last = reasoning_steps[-1]
                # Remove RETRIEVE lines
                clean = re.sub(r"RETRIEVE:.*?\n?", "", last).strip()
                final_answer = clean or last

            result["answer"]          = final_answer
            result["evidence"]        = evidence_all[:k]
            result["is_refusal"]      = self._is_refusal(final_answer)
            result["ircot_steps"]     = len(reasoning_steps)
            result["reasoning_chain"] = reasoning_steps
        except Exception as exc:
            logger.error(f"IRCoT failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result

    @staticmethod
    def _summarise_evidence(evidence: List[Dict]) -> str:
        if not evidence:
            return "(no evidence retrieved yet)"
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[{i}] {e.get('filename','?')}: {e.get('text','')[:250]}")
        return "\n".join(parts)
