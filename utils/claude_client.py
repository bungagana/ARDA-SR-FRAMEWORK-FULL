"""
Claude API wrapper (official `anthropic` SDK) for QA generation.

Used for QA dataset generation, per Section 2.4.1 ("The QAs were generated
using claude-haiku-4-5"). All other components (ARDA-SR generation, judging,
baselines) remain on Gemini per config.py — this client is deliberately
scoped to QA generation only.

Non-streaming: QA generation is short, structured JSON output (batches of
~15 QA pairs, well under the ~16000-token non-streaming safety threshold), so
there is no need for the streaming/get_final_message() pattern used for long
generations.

JSON output: via prompting (the existing DK_PROMPT/FR_PROMPT/etc. templates
already instruct "Return ONLY a JSON array"), not output_config.format
structured outputs — this keeps parity with GeminiClient.generate_json()'s
regex-fallback parsing so QAGenerator (02_generate_qa.py) works unmodified
against either client.
"""

import time
import json
import re
import logging
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_QA_MODEL, REQUEST_DELAY_S, MAX_RETRIES

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Thin wrapper around anthropic.Anthropic with the same generate()/
    generate_json() surface as utils.llm_client.GeminiClient, so it's a
    drop-in replacement for QA generation call sites.
    """

    def __init__(self, model: str = CLAUDE_QA_MODEL):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set. Check your .env file.")
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S - elapsed)
        self._last_call = time.time()

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate text from a prompt (non-streaming). Returns raw text string."""
        self._throttle()
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.stop_reason == "refusal":
                    raise RuntimeError(
                        f"Claude declined the request (stop_reason=refusal): "
                        f"{getattr(response, 'stop_details', None)}"
                    )
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return text.strip()
            except anthropic.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Claude rate limited (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                    f"Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Claude server error (attempt {attempt+1}/{MAX_RETRIES}): "
                        f"{exc}. Waiting {wait}s"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                        continue
                raise
            except anthropic.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Claude connection error (attempt {attempt+1}/{MAX_RETRIES}): "
                    f"{exc}. Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise

    def generate_json(self, prompt: str, max_tokens: int = 2048) -> dict | list:
        """Generate and parse JSON response. Strips markdown fences if present."""
        raw = self.generate(prompt, max_tokens)
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            logger.error(f"Failed to parse JSON from: {raw[:200]}")
            raise


_client: Optional[ClaudeClient] = None


def get_client() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
