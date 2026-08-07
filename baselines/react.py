"""Baseline: ReAct (Reasoning + Acting with iterative retrieval steps)."""

import time
import logging
import re
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K, REACT_MAX_STEPS

logger = logging.getLogger(__name__)


class ReActPipeline(BasePipeline):
    """
    ReAct: interleaves Thought → Action → Observation steps.
    Actions: Search[query] or Finish[answer].
    Reference: Yao et al. 2023, ICLR.
    """

    name = "react"

    SYSTEM_PROMPT = """\
You are a ReAct agent for answering questions about Indonesian transmigration.
Use Thought/Action/Observation format. Available action: Search[<query>] or Finish[<answer>]

Example:
Thought: I need to find information about X.
Action: Search[transmigration area X statistics]
Observation: [retrieved evidence]
Thought: Based on the evidence, I can answer.
Action: Finish[The answer is ...]

Now answer:
Question: {query}

{history}"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        history = ""
        evidence_all: List[Dict] = []

        try:
            for step in range(REACT_MAX_STEPS):
                prompt = self.SYSTEM_PROMPT.format(query=query, history=history)
                response = self.client.generate(prompt, max_tokens=512)

                # Parse Action
                action_match = re.search(r"Action:\s*(Search|Finish)\[(.+?)\]", response, re.DOTALL)
                if not action_match:
                    # No action found → treat as final answer
                    history += f"\n{response}"
                    break

                action_type = action_match.group(1)
                action_arg  = action_match.group(2).strip()

                if action_type == "Finish":
                    result["answer"] = action_arg
                    break

                # Search action
                evidence = self.retriever.retrieve(action_arg, k=3)
                evidence_all.extend(evidence)
                obs_text = "\n".join(e.get("text", "")[:300] for e in evidence)
                history += f"\n{response}\nObservation: {obs_text[:800]}"

            if not result["answer"]:
                # Extract from last Thought or just return history
                thought_match = re.search(r"Thought:\s*(.+?)(?=\n|$)", history, re.DOTALL)
                result["answer"] = thought_match.group(1).strip() if thought_match else history[-500:]

            result["evidence"]   = evidence_all[:k]
            result["is_refusal"] = self._is_refusal(result["answer"])
            result["react_steps"] = step + 1
        except Exception as exc:
            logger.error(f"ReAct failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
