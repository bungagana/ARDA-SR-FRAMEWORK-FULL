"""
OpenAI API wrapper (official `openai` SDK) for LLM-as-judge evaluation.

Used ONLY as an alternative judge model (drop-in for evaluation.llm_judge.LLMJudge),
so that judging is not done by the same model family (Gemini) that also generates
the answers being judged — this addresses the "judge is not independent of the
generator" critique. Generation/retrieval/DDA/SR/AQR all stay on Gemini; QA
dataset generation stays on Claude Haiku (utils/claude_client.py). This client
is deliberately scoped to judging only.

Uses the Responses API (`client.responses.create`), the current recommended
endpoint for GPT-5.x models, with `reasoning: {"effort": ...}` — judging is a
short structured-JSON scoring task, so a low reasoning effort is used by
default (see config.GPT_JUDGE_REASONING_EFFORT).

JSON output: via prompting (same convention as GeminiClient.generate_json() /
ClaudeClient.generate_json()) so LLMJudge and QAValidator work unmodified
against this client too — no SDK-specific structured-output schema is used,
to keep behavior identical (and therefore comparable) across judge backends.
"""

import time
import json
import re
import logging
from typing import Optional

import openai

from config import OPENAI_API_KEY, GPT_JUDGE_MODEL, GPT_JUDGE_REASONING_EFFORT, REQUEST_DELAY_S, MAX_RETRIES

logger = logging.getLogger(__name__)


class GPTJudgeClient:
    """
    Thin wrapper around openai.OpenAI with the same generate()/generate_json()
    surface as utils.llm_client.GeminiClient and utils.claude_client.ClaudeClient,
    so it's a drop-in replacement for evaluation.llm_judge.LLMJudge(client=...).
    """

    def __init__(self, model: str = GPT_JUDGE_MODEL, reasoning_effort: str = GPT_JUDGE_REASONING_EFFORT):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model
        self.reasoning_effort = reasoning_effort
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
                response = self._client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": self.reasoning_effort},
                )
                text = getattr(response, "output_text", None)
                if text is None:
                    # Fallback: walk the output items for the first text part.
                    text = ""
                    for item in getattr(response, "output", []) or []:
                        for part in getattr(item, "content", []) or []:
                            if getattr(part, "type", "") == "output_text":
                                text = part.text
                                break
                        if text:
                            break
                return (text or "").strip()
            except openai.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"GPT judge rate limited (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                    f"Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        f"GPT judge server error (attempt {attempt+1}/{MAX_RETRIES}): "
                        f"{exc}. Waiting {wait}s"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                        continue
                raise
            except openai.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"GPT judge connection error (attempt {attempt+1}/{MAX_RETRIES}): "
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


_client: Optional[GPTJudgeClient] = None


def get_client() -> GPTJudgeClient:
    global _client
    if _client is None:
        _client = GPTJudgeClient()
    return _client
