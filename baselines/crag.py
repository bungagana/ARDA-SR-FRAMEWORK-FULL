"""Baseline: CRAG (Corrective Retrieval-Augmented Generation)."""

import time
import logging
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)


class CRAGPipeline(BasePipeline):
    """
    CRAG: evaluates retrieval quality before generation.
    If retrieval is poor → fallback to parametric knowledge.
    Reference: Yan et al. 2024 (Scopus).

    Fidelity note: this is a same-backbone, prompt-based approximation of
    CRAG's retrieval-evaluation step. The original CRAG uses a small
    trained relevance evaluator (a fine-tuned T5) plus a web-search
    fallback/refinement stage; here, relevance evaluation is delegated to
    the shared LLM backbone via EVAL_PROMPT, and corrective retrieval is
    restricted to the same closed corpus (no live web access is available
    in this setting). Should not be read as a reproduction of the original
    trained evaluator's behavior.
    """

    name = "crag"

    EVAL_PROMPT = """\
Evaluate the relevance of this evidence for answering the question.
Return ONLY JSON: {{"score": <0.0-1.0>, "verdict": "CORRECT"|"AMBIGUOUS"|"INCORRECT"}}

Question: {query}
Evidence:
{evidence}"""

    CORRECT_PROMPT = """\
Answer the question based on the verified evidence.
Question: {query}

Verified Evidence:
{evidence}

Answer:"""

    FALLBACK_PROMPT = """\
The retrieved evidence was insufficient. Answer using your best knowledge.
If uncertain, indicate it clearly.
Question: {query}
Answer:"""

    REFINE_PROMPT = """\
The evidence is partially relevant. Extract only the relevant parts and answer.
Question: {query}

Partial Evidence:
{evidence}

Answer (use only relevant parts):"""

    DECOMPOSE_PROMPT = """\
Decompose the question into 2-3 focused search queries that would help correct
or complete weak retrieval results. Return ONLY JSON:
{{"queries": ["...", "..."]}}

Question: {query}
Current evidence:
{evidence}"""

    RECOMPOSE_PROMPT = """\
Recompose the answer from the filtered evidence. Ignore irrelevant passages and
answer only if the evidence supports the claim.

Question: {query}
Filtered Evidence:
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

            # Step 1: Evaluate retrieval quality
            ev_text  = self._format_evidence(evidence)
            eval_out = self.client.generate_json(
                self.EVAL_PROMPT.format(query=query, evidence=ev_text[:1500])
            )
            verdict = str(eval_out.get("verdict", "AMBIGUOUS")).upper()
            score   = float(eval_out.get("score", 0.5))
            result["crag_verdict"] = verdict
            result["crag_score"]   = score

            # Step 2: Generate based on verdict
            if verdict == "CORRECT" and score >= 0.6:
                answer = self.client.generate(
                    self.CORRECT_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                )
            elif verdict == "INCORRECT" or score < 0.3:
                extra_evidence = self._corrective_retrieve(query, ev_text)
                if extra_evidence:
                    evidence.extend(extra_evidence)
                    corrected_text = self._format_evidence(evidence)
                    answer = self.client.generate(
                        self.RECOMPOSE_PROMPT.format(query=query, evidence=corrected_text),
                        max_tokens=512,
                    )
                else:
                    answer = self.client.generate(
                        self.FALLBACK_PROMPT.format(query=query), max_tokens=512
                    )
            else:
                extra_evidence = self._corrective_retrieve(query, ev_text)
                if extra_evidence:
                    evidence.extend(extra_evidence)
                    ev_text = self._format_evidence(evidence)
                answer = self.client.generate(
                    self.REFINE_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                )

            result["answer"]     = answer
            result["evidence"]   = evidence[:k]
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"CRAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result

    def _corrective_retrieve(self, query: str, evidence_text: str) -> List[Dict]:
        """Approximate CRAG's corrective retrieval using the same corpus."""
        try:
            out = self.client.generate_json(
                self.DECOMPOSE_PROMPT.format(query=query, evidence=evidence_text[:1000])
            )
            sub_queries = out.get("queries", []) if isinstance(out, dict) else []
        except Exception:
            sub_queries = []

        collected: List[Dict] = []
        seen = set()
        for sub_query in sub_queries[:3]:
            for chunk in self.retriever.retrieve(str(sub_query), k=2):
                key = (chunk.get("filename"), chunk.get("text", "")[:80])
                if key in seen:
                    continue
                seen.add(key)
                collected.append(chunk)
        return collected
