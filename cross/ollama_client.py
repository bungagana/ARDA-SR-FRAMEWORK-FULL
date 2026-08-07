"""Ollama client adapter with the same interface as GeminiClient.

This lets the existing ARDA-SR pipeline run on local, free LLM backbones.
The only required methods are generate() and generate_json().
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import MAX_OUTPUT_TOKENS, MAX_RETRIES, REQUEST_DELAY_S, TEMPERATURE

logger = logging.getLogger(__name__)


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = TEMPERATURE
    request_delay_s: float = REQUEST_DELAY_S

    def __post_init__(self) -> None:
        self.model_name = self.model
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.request_delay_s:
            time.sleep(self.request_delay_s - elapsed)
        self._last_call = time.time()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
        self._throttle()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        for attempt in range(MAX_RETRIES):
            try:
                raw = self._post_json("/api/generate", payload)
                return str(raw.get("response", "")).strip()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
            except Exception as exc:
                wait = 2**attempt
                logger.warning(
                    "Ollama error for %s (attempt %s/%s): %s",
                    self.model,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(wait)
        return ""

    def generate_json(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict | list:
        raw = self.generate(prompt, max_tokens=max_tokens)
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            logger.error("Failed to parse JSON from Ollama response: %s", raw[:300])
            raise
