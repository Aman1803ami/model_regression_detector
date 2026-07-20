"""
LLM-powered customer support email classifier.

Wraps the Google Gemini API to classify customer emails into categories
and generate summaries. Handles retries, timeouts, JSON parsing, and
tracks latency/token usage per call.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from google import genai

from src.config import (
    ClassificationResult,
    EmailCategory,
    FewShotExample,
    PromptConfig,
)


def _get_client() -> genai.Client:
    """Create a Gemini client using the API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com/"
        )
    return genai.Client(api_key=api_key)


def _build_prompt(email_text: str, config: PromptConfig) -> str:
    """
    Build the full prompt from the system prompt, few-shot examples,
    and the target email.
    """
    parts = [config.system_prompt.strip()]

    # Add few-shot examples if any
    if config.few_shot_examples:
        parts.append("\nHere are some examples:\n")
        for i, ex in enumerate(config.few_shot_examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Email: {ex.input.strip()}")
            output = json.dumps({
                "category": ex.output_category.value,
                "summary": ex.output_summary,
            })
            parts.append(f"Response: {output}\n")

    # Add the target email
    parts.append("Now classify this email:")
    parts.append(f"Email: {email_text.strip()}")
    parts.append("Response:")

    return "\n".join(parts)


def _parse_llm_response(raw_response: str) -> tuple[str, str]:
    """
    Parse the LLM response to extract category and summary.

    Handles common LLM output variations:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with trailing text
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try to find JSON object in the response
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            category = data.get("category", "general").lower().strip()
            summary = data.get("summary", "").strip()
            return category, summary
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole thing
    try:
        data = json.loads(text)
        category = data.get("category", "general").lower().strip()
        summary = data.get("summary", "").strip()
        return category, summary
    except json.JSONDecodeError:
        pass

    # Last resort: return general with the raw text as summary
    return "general", text[:200]


def classify_email(
    email_text: str,
    prompt_config: PromptConfig,
    client: Optional[genai.Client] = None,
    max_retries: int = 3,
) -> ClassificationResult:
    """
    Classify a customer support email using the LLM.

    Args:
        email_text: The customer email text to classify.
        prompt_config: Versioned prompt configuration.
        client: Optional pre-configured Gemini client.
        max_retries: Number of retry attempts for transient failures.

    Returns:
        ClassificationResult with category, summary, and metrics.
    """
    if client is None:
        client = _get_client()

    prompt = _build_prompt(email_text, prompt_config)

    last_error = None
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()

            response = client.models.generate_content(
                model=prompt_config.model,
                contents=prompt,
                config={
                    "temperature": prompt_config.temperature,
                    "max_output_tokens": prompt_config.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            raw_text = response.text or ""

            # Extract token usage from response metadata
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ) or 0
                completion_tokens = getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ) or 0

            # Parse the response
            category_str, summary = _parse_llm_response(raw_text)

            # Validate category
            try:
                category = EmailCategory(category_str)
            except ValueError:
                category = EmailCategory.GENERAL

            return ClassificationResult(
                category=category,
                summary=summary,
                raw_response=raw_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait = 2**attempt
                time.sleep(wait)
                continue
            break

    # All retries failed — return error result
    return ClassificationResult(
        category=EmailCategory.GENERAL,
        summary=f"Classification failed: {str(last_error)}",
        raw_response=str(last_error),
        latency_ms=0.0,
    )


async def classify_email_async(
    email_text: str,
    prompt_config: PromptConfig,
    client: Optional[genai.Client] = None,
    max_retries: int = 3,
) -> ClassificationResult:
    """
    Async wrapper for classify_email.

    The google-genai SDK is synchronous, so this runs the sync version
    in an executor to avoid blocking the event loop during batch processing.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        classify_email,
        email_text,
        prompt_config,
        client,
        max_retries,
    )
