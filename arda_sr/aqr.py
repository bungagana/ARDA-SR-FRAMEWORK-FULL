"""
AQR: Adaptive Query Router
Implements uncertainty-based query routing via conditional entropy.

φ(q)  = feature vector (5 components)
P(m|q) = mode probability distribution over M = {m1, m2, m3, m4}
H(M|φ(q)) = -Σ P(m|q) log P(m|q)  → routing decision
"""

import math
import logging
import json
from typing import Dict, Tuple

from utils.llm_client import GeminiClient
from config import ENTROPY_THRESHOLD

logger = logging.getLogger(__name__)

MODES = ["m1", "m2", "m3", "m4"]

MODE_DESC = {
    "m1": "Direct conceptual answering (no retrieval needed)",
    "m2": "Retrieval-grounded answering (single-document factual)",
    "m3": "Hybrid answering under routing uncertainty (multi-hop or ambiguous)",
    "m4": "Policy-scenario reasoning (multi-alternative decision analysis)",
}

AQR_PROMPT = """\
You are an AQR (Adaptive Query Router) for a government transmigration QA system.

Query: "{query}"

Classify this query into one of 4 response modes and provide feature scores.
Modes: m1=Direct Knowledge (no retrieval), m2=Factual Retrieval, m3=Hybrid/Ambiguous, m4=Policy Scenario
Probabilities must sum to 1.0. All scores are floats between 0.0 and 1.0.

Respond with ONLY valid JSON (no markdown, no explanation):
{{"features":{{"entity_signal":0.5,"domain_specificity":0.5,"temporal_ref":0.0,"multihop_signal":0.2,"context_dep":0.1}},"mode_probs":{{"m1":0.1,"m2":0.6,"m3":0.2,"m4":0.1}},"reasoning":"brief reason"}}

Now classify:
Query: "{query}"
JSON:"""


class AQR:
    """
    Adaptive Query Router.

    Route a query to one of four response modes using entropy-based uncertainty:
    - H ≤ τ_H → confident routing → assign dominant mode
    - H  > τ_H → activate hybrid path (m3)
    """

    def __init__(self, client: GeminiClient | None = None, tau_h: float = ENTROPY_THRESHOLD):
        self.client = client or GeminiClient()
        self.tau_h = tau_h

    def classify(self, query: str) -> Dict:
        """
        Returns:
          mode          : str  — assigned mode (m1/m2/m3/m4)
          mode_probs    : dict — P(m|q) for all modes
          entropy       : float — H(M|φ(q))
          features      : dict — φ(q) feature vector
          hybrid_path   : bool — True if entropy exceeds threshold
          reasoning     : str
        """
        prompt = AQR_PROMPT.format(query=query)
        try:
            result = self.client.generate_json(prompt)
        except Exception as exc:
            logger.warning(f"AQR classification failed: {exc}. Defaulting to m3.")
            return self._default_result()

        features   = result.get("features", {})
        mode_probs = result.get("mode_probs", {"m1": 0.25, "m2": 0.25, "m3": 0.25, "m4": 0.25})
        reasoning  = result.get("reasoning", "")

        # Normalise probabilities
        total = sum(mode_probs.get(m, 0.0) for m in MODES)
        if total < 1e-9:
            total = 1.0
        mode_probs = {m: mode_probs.get(m, 0.0) / total for m in MODES}

        entropy    = self._entropy(mode_probs)
        hybrid     = entropy > self.tau_h
        dominant   = max(mode_probs, key=mode_probs.get)
        mode       = "m3" if hybrid else dominant

        return {
            "mode":       mode,
            "mode_probs": mode_probs,
            "entropy":    round(entropy, 4),
            "features":   features,
            "hybrid_path": hybrid,
            "reasoning":  reasoning,
        }

    @staticmethod
    def _entropy(probs: Dict[str, float]) -> float:
        """H(M|q) = -Σ p·log2(p), in bits.

        Base-2 log is required so H_max for |M|=4 modes equals log2(4) = 2.0
        bits (paper Section 2.3.2), making τ_H = 1.05 the intended 52.5% of
        H_max. Using natural log here would make τ_H a different (and
        undocumented) fraction of the actual max entropy (ln(4) ≈ 1.386 nats).
        """
        h = 0.0
        for p in probs.values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    @staticmethod
    def _default_result() -> Dict:
        uniform = {m: 0.25 for m in MODES}
        return {
            "mode": "m3",
            "mode_probs": uniform,
            "entropy": round(AQR._entropy(uniform), 4),
            "features": {},
            "hybrid_path": True,
            "reasoning": "Default fallback due to classifier error.",
        }
