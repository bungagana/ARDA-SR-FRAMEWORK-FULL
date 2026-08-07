"""
DDA: Dual-Draft Arbitrator
Generates two parallel answer drafts and selects/combines via utility-based scoring.

u(a) = β1·Rel(a) + β2·Faith(a) + β3·Cov(a) − β4·Risk(a)
Default β values are fixed from design reasoning. Optional grid search must use
development data that are separate from the final test set.
"""

import itertools
import logging
import json
from typing import Dict, List, Optional, Tuple

from utils.llm_client import GeminiClient
from config import DDA_BETA_SEARCH, DDA_DECISION_MARGIN

logger = logging.getLogger(__name__)

DEFAULT_BETAS = {"b1": 0.30, "b2": 0.25, "b3": 0.20, "b4": 0.25}

DIRECT_PROMPT = """\
You are a knowledgeable assistant for the Indonesian transmigration domain.
Answer the following question using your general knowledge. Be concise and accurate.

Question: {query}

Answer:"""

RETRIEVAL_PROMPT = """\
You are a precise assistant for the Indonesian transmigration domain.
Answer the question strictly based on the provided evidence. Do not add information not in the evidence.
If the evidence is insufficient, say so briefly.

Question: {query}

Evidence:
{evidence}

Answer:"""

UTILITY_PROMPT = """\
Evaluate the quality of this answer to the given question.
Return ONLY JSON with scores in [0.0, 1.0]:

Question: {query}
Reference Answer: {reference}
Evidence Available to the Answer (may be empty if this was a parametric-only draft):
{evidence}
Candidate Answer: {answer}

{{
  "relevance": <0.0–1.0>,     // how relevant is the answer to the question
  "faithfulness": <0.0–1.0>,  // how well grounded in the Evidence above / factually accurate.
                               // If Evidence is empty, judge only against general factual plausibility.
  "coverage": <0.0–1.0>,      // completeness of information provided
  "risk": <0.0–1.0>           // risk of hallucination or incorrect info (LOWER is better answer)
}}"""

# Minimum acceptable weighted-utility score for a lone parametric draft
# (Algorithm 1, line 31: "minimum-quality verification"). Reuses the same
# 0.3 bar DDA already applies elsewhere so the threshold is consistent
# across the module rather than a second, undocumented magic number.
MIN_QUALITY_THRESHOLD = 0.3

COMBINE_PROMPT = """\
You have two candidate answers. Combine the best elements of both into one comprehensive answer.
Prioritize factual accuracy over length.

Question: {query}
Answer A (direct knowledge): {a_dir}
Answer B (retrieval-grounded): {a_ret}

Combined Answer:"""


class DDA:
    """
    Dual-Draft Arbitrator.
    Maintains answerability by always generating a parametric fallback draft.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        betas: Dict[str, float] | None = None,
    ):
        self.client = client or GeminiClient()
        self.betas = betas or DEFAULT_BETAS

    def arbitrate(
        self,
        query: str,
        evidence: List[Dict],
        reference: str = "",
    ) -> Dict:
        """
        Generate two drafts, score with utility function, return best (or combined).

        Returns dict with: answer, draft_dir, draft_ret, utility_dir, utility_ret,
                           selected, betas, evidence_used
        """
        # ── Draft A: Parametric (direct) ──────────────────────────────────
        draft_dir = self._generate_direct(query)

        # ── Draft B: Retrieval-grounded ────────────────────────────────────
        evidence_text = self._format_evidence(evidence)
        draft_ret = self._generate_retrieval(query, evidence_text) if evidence else ""

        # ── Score both drafts (Faith is judged against the SAME evidence
        #    text passed to the retrieval draft, per paper Section 2.3.4:
        #    "evaluates each draft against the query and the available
        #    contextual evidence") ─────────────────────────────────────────
        u_dir = self._score_utility(query, draft_dir, reference, evidence_text="")
        u_ret = (
            self._score_utility(query, draft_ret, reference, evidence_text=evidence_text)
            if draft_ret else {"relevance": 0, "faithfulness": 0, "coverage": 0, "risk": 1}
        )

        score_dir = self._weighted_utility(u_dir)
        score_ret = self._weighted_utility(u_ret) if draft_ret else -1.0

        # ── Arbitration — literal three-way rule, Eq. (17) ─────────────────
        min_quality_passed = True
        if not draft_ret:
            # Algorithm 1, line 31: "Set a* <- a(dir) after minimum-quality
            # verification". Score the lone parametric draft against the
            # same utility function; flag (but still return, to preserve
            # DDA's answerability-maintenance goal) if it fails the bar.
            selected = "direct"
            answer = draft_dir
            min_quality_passed = score_dir >= MIN_QUALITY_THRESHOLD
        elif score_dir - score_ret > DDA_DECISION_MARGIN:
            selected = "direct"
            answer = draft_dir
        elif score_ret - score_dir > DDA_DECISION_MARGIN:
            selected = "retrieval"
            answer = draft_ret
        else:
            selected = "combined"
            answer = self._combine(query, draft_dir, draft_ret)

        # Detect refusal in the chosen answer
        is_refusal = self._is_refusal(answer) or not min_quality_passed

        return {
            "answer":             answer,
            "draft_dir":          draft_dir,
            "draft_ret":          draft_ret,
            "utility_dir":        u_dir,
            "utility_ret":        u_ret,
            "score_dir":          round(score_dir, 4),
            "score_ret":          round(score_ret, 4),
            "selected":           selected,
            "min_quality_passed": min_quality_passed,
            "is_refusal":         is_refusal,
            "betas":              self.betas,
        }

    def _generate_direct(self, query: str) -> str:
        prompt = DIRECT_PROMPT.format(query=query)
        try:
            return self.client.generate(prompt, max_tokens=512)
        except Exception as exc:
            logger.warning(f"Direct draft failed: {exc}")
            return ""

    def _generate_retrieval(self, query: str, evidence_text: str) -> str:
        prompt = RETRIEVAL_PROMPT.format(query=query, evidence=evidence_text)
        try:
            return self.client.generate(prompt, max_tokens=768)
        except Exception as exc:
            logger.warning(f"Retrieval draft failed: {exc}")
            return ""

    def _score_utility(self, query: str, answer: str, reference: str, evidence_text: str = "") -> Dict:
        if not answer.strip():
            return {"relevance": 0.0, "faithfulness": 0.0, "coverage": 0.0, "risk": 1.0}
        prompt = UTILITY_PROMPT.format(
            query=query, reference=reference or answer, answer=answer,
            evidence=evidence_text or "(none — parametric-only draft)",
        )
        try:
            scores = self.client.generate_json(prompt)
            return {
                "relevance":    float(scores.get("relevance", 0.5)),
                "faithfulness": float(scores.get("faithfulness", 0.5)),
                "coverage":     float(scores.get("coverage", 0.5)),
                "risk":         float(scores.get("risk", 0.5)),
            }
        except Exception:
            return {"relevance": 0.5, "faithfulness": 0.5, "coverage": 0.5, "risk": 0.5}

    def _weighted_utility(self, scores: Dict) -> float:
        b = self.betas
        return (b["b1"] * scores["relevance"]
                + b["b2"] * scores["faithfulness"]
                + b["b3"] * scores["coverage"]
                - b["b4"] * scores["risk"])

    def _combine(self, query: str, a_dir: str, a_ret: str) -> str:
        prompt = COMBINE_PROMPT.format(query=query, a_dir=a_dir, a_ret=a_ret)
        try:
            return self.client.generate(prompt, max_tokens=768)
        except Exception:
            return a_ret or a_dir

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[Evidence {i}] (source: {e.get('filename','?')})\n{e.get('text','')[:600]}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        refusal_phrases = [
            "i don't have", "i do not have", "tidak memiliki informasi",
            "tidak dapat menjawab", "saya tidak tahu", "tidak ada informasi",
            "please refer to", "silakan merujuk", "cannot answer",
            "unable to answer", "no information available",
        ]
        t = text.lower()
        return any(p in t for p in refusal_phrases)


# ── Beta Grid Search ───────────────────────────────────────────────────────

def grid_search_betas(
    validation_data: List[Dict],
    client: GeminiClient,
    beta_values: List[float] = DDA_BETA_SEARCH,
) -> Dict[str, float]:
    """
    Grid search β1–β4 over validation_data (list of {query, evidence, reference_answer}).
    Constraint: β1+β2+β3+β4 = 1.0, all > 0.
    Returns best beta dict.
    """
    logger.info("Running DDA β grid search...")
    best_betas = DEFAULT_BETAS
    best_score = -1.0

    dda_tmp = DDA(client=client)
    candidates = [
        (b1, b2, b3, b4)
        for b1 in beta_values
        for b2 in beta_values
        for b3 in beta_values
        for b4 in beta_values
        if abs(b1 + b2 + b3 + b4 - 1.0) < 1e-6
    ]
    logger.info(f"  {len(candidates)} beta combinations to evaluate on {len(validation_data)} samples")

    for b1, b2, b3, b4 in candidates[:50]:  # limit to first 50 for speed
        betas = {"b1": b1, "b2": b2, "b3": b3, "b4": b4}
        dda_tmp.betas = betas
        total = 0.0
        for item in validation_data[:10]:    # sample 10 items
            result = dda_tmp.arbitrate(
                item["query"], item.get("evidence", []), item.get("reference_answer", "")
            )
            total += result["score_dir"] if result["selected"] == "direct" else result["score_ret"]
        avg = total / min(10, len(validation_data))
        if avg > best_score:
            best_score = avg
            best_betas = betas

    logger.info(f"  Best β: {best_betas} (score={best_score:.4f})")
    return best_betas
