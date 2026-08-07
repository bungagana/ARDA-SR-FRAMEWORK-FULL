"""
SR: Scenario Reasoning
For policy-scenario queries (mode m4), generates structured scenarios and
selects the optimal one via Expected Utility maximization.

EU(s) = p_success(s) · Utility(s) − λ_SR · Risk(s) · Loss(s)
s* = argmax_{s ∈ S(q)} EU(s)
"""

import logging
import json
from typing import Dict, List

from utils.llm_client import GeminiClient
from config import SR_LAMBDA, SR_NUM_SCENARIOS

logger = logging.getLogger(__name__)

SCENARIO_GENERATION_PROMPT = """\
You are a policy analyst for the Indonesian transmigration sector.
Generate exactly {n_scenarios} alternative intervention/policy scenarios for the following query.

Query: "{query}"

Relevant Evidence:
{evidence}

For each scenario, provide a structured analysis. Return ONLY this JSON array:
[
  {{
    "name": "<short scenario name>",
    "description": "<2-3 sentence description>",
    "p_success": <0.0–1.0>,
    "utility": <0.0–1.0>,
    "risk": <0.0–1.0>,
    "loss": <0.0–1.0>,
    "rationale": "<why this scenario, its key trade-offs>",
    "assumptions": "<key assumptions>",
    "timeline": "<estimated timeline>"
  }},
  ...
]

Scoring guidelines:
- p_success: probability of successful implementation given constraints
- utility: alignment with policy objectives and stakeholder needs
- risk: probability of adverse outcomes
- loss: magnitude of worst-case impact (0=negligible, 1=catastrophic)"""

POLICY_ANSWER_PROMPT = """\
You are a government policy advisor for the Indonesian transmigration sector.
Provide a structured policy recommendation based on the scenario analysis below.

Query: "{query}"

Optimal Scenario Selected: {optimal_name}
EU Score: {eu_score:.3f}

All Scenarios Evaluated:
{scenarios_text}

Evidence Used:
{evidence}

Provide a comprehensive policy answer that:
1. States the recommended intervention/scenario and why
2. Compares it explicitly against the alternatives
3. Identifies key risks and mitigation strategies
4. Gives actionable implementation steps
5. Notes any assumptions or limitations

Policy Recommendation:"""


class SR:
    """
    Scenario Reasoning module.
    Activated only for policy-scenario queries (mode m4).
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        lam: float = SR_LAMBDA,
        n_scenarios: int = SR_NUM_SCENARIOS,
    ):
        self.client = client or GeminiClient()
        self.lam = lam
        self.n_scenarios = n_scenarios

    def reason(self, query: str, evidence: List[Dict]) -> Dict:
        """
        Generate scenarios, compute EU, select optimal, produce policy answer.

        Returns:
          answer, scenarios, optimal_scenario, eu_scores, sr_compliant
        """
        evidence_text = self._format_evidence(evidence)

        # ── 1. Generate scenario set S(q) ──────────────────────────────────
        scenarios = self._generate_scenarios(query, evidence_text)
        if not scenarios:
            return self._fallback(query)

        # ── 2. Compute EU for each scenario ───────────────────────────────
        eu_scores = {}
        for s in scenarios:
            eu = self._expected_utility(s)
            eu_scores[s["name"]] = round(eu, 4)
            s["eu"] = round(eu, 4)

        # ── 3. Select optimal scenario ────────────────────────────────────
        optimal = max(scenarios, key=lambda s: s["eu"])

        # ── 4. Generate policy-structured answer ──────────────────────────
        scenarios_text = self._format_scenarios(scenarios)
        answer = self._generate_answer(query, optimal, scenarios_text, evidence_text)

        # SR compliance: answer must contain structured elements
        compliant = self._check_compliance(answer)

        return {
            "answer":           answer,
            "scenarios":        scenarios,
            "optimal_scenario": optimal,
            "eu_scores":        eu_scores,
            "sr_compliant":    compliant,
        }

    def _generate_scenarios(self, query: str, evidence_text: str) -> List[Dict]:
        prompt = SCENARIO_GENERATION_PROMPT.format(
            n_scenarios=self.n_scenarios,
            query=query,
            evidence=evidence_text[:2000],
        )
        try:
            raw = self.client.generate_json(prompt)
            if isinstance(raw, list):
                return [self._validate_scenario(s) for s in raw]
            return []
        except Exception as exc:
            logger.warning(f"SR scenario generation failed: {exc}")
            return []

    def _expected_utility(self, s: Dict) -> float:
        """EU(s) = p_success * utility − λ * risk * loss"""
        p = float(s.get("p_success", 0.5))
        u = float(s.get("utility",   0.5))
        r = float(s.get("risk",      0.5))
        lo = float(s.get("loss",     0.5))
        return p * u - self.lam * r * lo

    def _generate_answer(
        self,
        query: str,
        optimal: Dict,
        scenarios_text: str,
        evidence_text: str,
    ) -> str:
        prompt = POLICY_ANSWER_PROMPT.format(
            query=query,
            optimal_name=optimal.get("name", ""),
            eu_score=optimal.get("eu", 0.0),
            scenarios_text=scenarios_text,
            evidence=evidence_text[:1500],
        )
        try:
            return self.client.generate(prompt, max_tokens=1024)
        except Exception as exc:
            logger.warning(f"SR answer generation failed: {exc}")
            return f"Recommended scenario: {optimal.get('name','')}. EU={optimal.get('eu',0):.3f}."

    @staticmethod
    def _validate_scenario(s: dict) -> dict:
        defaults = {
            "name": "Scenario", "description": "", "p_success": 0.5,
            "utility": 0.5, "risk": 0.5, "loss": 0.5,
            "rationale": "", "assumptions": "", "timeline": "",
        }
        for k, v in defaults.items():
            if k not in s:
                s[k] = v
        for num_key in ["p_success", "utility", "risk", "loss"]:
            try:
                s[num_key] = max(0.0, min(1.0, float(s[num_key])))
            except (ValueError, TypeError):
                s[num_key] = 0.5
        return s

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[{i}] {e.get('filename','?')}: {e.get('text','')[:400]}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_scenarios(scenarios: List[Dict]) -> str:
        lines = []
        for s in scenarios:
            lines.append(
                f"• {s['name']} — p_success={s['p_success']:.2f}, "
                f"utility={s['utility']:.2f}, risk={s['risk']:.2f}, "
                f"EU={s.get('eu',0):.3f}\n  {s.get('rationale','')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_compliance(answer: str) -> bool:
        required = ["scenario", "risk", "recommend"]
        t = answer.lower()
        return sum(1 for r in required if r in t) >= 2

    @staticmethod
    def _fallback(query: str) -> Dict:
        return {
            "answer": f"Unable to generate policy scenarios for: {query}",
            "scenarios": [],
            "optimal_scenario": {},
            "eu_scores": {},
            "sr_compliant": False,
        }
