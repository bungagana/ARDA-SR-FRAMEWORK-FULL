"""Hugging Face Inference API client adapter for ARDA-SR.

This adapter exposes generate() and generate_json(), matching GeminiClient.
It uses hosted Hugging Face inference instead of local model execution.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from config import MAX_OUTPUT_TOKENS, MAX_RETRIES, REQUEST_DELAY_S, TEMPERATURE

logger = logging.getLogger(__name__)


@dataclass
class HFInferenceClient:
    model: str
    provider: str = "auto"
    temperature: float = TEMPERATURE
    request_delay_s: float = REQUEST_DELAY_S
    timeout: float = 300.0

    def __post_init__(self) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.model_name = self.model
        self._last_call = 0.0
        self._client = InferenceClient(
            model=self.model,
            provider=self.provider,
            token=token,
            timeout=self.timeout,
        )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.request_delay_s:
            time.sleep(self.request_delay_s - elapsed)
        self._last_call = time.time()

    def generate(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
        self._throttle()
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_chat(prompt, max_tokens=max_tokens)
            except Exception as exc:
                try:
                    return self._generate_text(prompt, max_tokens=max_tokens)
                except Exception as text_exc:
                    wait = 2**attempt
                    logger.warning(
                        "HF inference error for %s (attempt %s/%s): chat=%s | text=%s",
                        self.model,
                        attempt + 1,
                        MAX_RETRIES,
                        exc,
                        text_exc,
                    )
                    if attempt == MAX_RETRIES - 1:
                        raise text_exc
                    time.sleep(wait)
        return ""

    def _generate_chat(self, prompt: str, max_tokens: int) -> str:
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        choices = getattr(response, "choices", None) or response.get("choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None) or choices[0].get("message", {})
        content = getattr(message, "content", None) or message.get("content", "")
        return str(content).strip()

    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        return str(
            self._client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
        ).strip()

    def generate_json(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict | list:
        raw = self.generate(prompt, max_tokens=max_tokens)
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            logger.error("Failed to parse JSON from HF response: %s", raw[:300])
            raise
