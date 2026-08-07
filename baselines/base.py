"""Abstract base class for all QA pipeline implementations."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from utils.llm_client import GeminiClient
from utils.kb_builder import KnowledgeBase
from config import TOP_K

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Common interface for all baseline and proposed methods."""

    name: str = "base"

    def __init__(self, kb: KnowledgeBase, client: GeminiClient | None = None):
        self.kb     = kb
        self.client = client or GeminiClient()

    @abstractmethod
    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        """
        Run the pipeline for a single query.

        Returns dict with keys:
          query, reference, method, answer, evidence,
          is_refusal, latency_s
          (subclasses may add more keys)
        """

    def _base_result(self, query: str, reference: str) -> Dict:
        return {
            "query":      query,
            "reference":  reference,
            "method":     self.name,
            "answer":     "",
            "evidence":   [],
            "is_refusal": False,
            "latency_s":  0.0,
        }

    @staticmethod
    def _is_refusal(text: str) -> bool:
        phrases = [
            "i don't have", "i do not have", "tidak memiliki informasi",
            "tidak dapat menjawab", "saya tidak tahu", "tidak ada informasi",
            "please refer to", "silakan merujuk", "cannot answer",
            "unable to answer", "no information available", "maaf", "I cannot",
        ]
        t = text.lower()
        return any(p in t for p in phrases)

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[Evidence {i}] (source: {e.get('filename','?')})\n{e.get('text','')[:600]}")
        return "\n\n".join(parts)
