"""
Multi-dimensional scoring engine.

Scores LLM classifier outputs across multiple dimensions:
- Category match (binary exact match)
- Summary relevance (LLM-as-judge, 1-5 scale)
- Latency (flag outliers)
- Token efficiency

This module converts raw ClassificationResults into scored TestCaseResults
and aggregates them into per-category and overall metrics.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Optional

from google import genai

from src.config import (
    CategoryMetrics,
    ClassificationResult,
    EmailCategory,
    TestCase,
    TestCaseResult,
)


# ---------------------------------------------------------------------------
# LLM-as-Judge prompt for summary relevance scoring
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a customer support email summary.

Given:
- The original customer email
- The expected ideal summary
- The generated summary to evaluate

Rate the generated summary on a scale of 1 to 5:
- 5: Captures all key information, matches the intent and tone of the expected summary perfectly
- 4: Captures most key information with minor omissions or phrasing differences
- 3: Captures the general idea but misses important details or has inaccurate elements
- 2: Partially relevant but misses the main point or includes significant inaccuracies
- 1: Irrelevant, incorrect, or completely misses the customer's issue

Respond with ONLY a JSON object: {"score": <integer 1-5>, "reason": "<brief explanation>"}

Original email:
{email}

Expected summary:
{expected}

Generated summary:
{generated}

Your evaluation:"""


def score_category_match(
    predicted: Optional[EmailCategory],
    expected: EmailCategory,
) -> bool:
    """Binary exact match between predicted and expected category."""
    if predicted is None:
        return False
    return predicted == expected


def score_summary_relevance(
    email_text: str,
    expected_summary: str,
    generated_summary: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.1-flash-lite",
) -> float:
    """
    Use LLM-as-judge to score summary relevance on a 1-5 scale.

    Falls back to a simple heuristic if the judge call fails.
    """
    if not generated_summary or generated_summary.startswith("Classification failed"):
        return 1.0

    # Try LLM-as-judge
    try:
        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                return _heuristic_summary_score(expected_summary, generated_summary)

        prompt = _JUDGE_PROMPT.format(
            email=email_text[:500],  # Truncate to save tokens
            expected=expected_summary,
            generated=generated_summary,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 100,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text or ""

        # Parse score from response
        import re
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 3))
            return max(1.0, min(5.0, score))

    except Exception:
        pass

    # Fallback to heuristic
    return _heuristic_summary_score(expected_summary, generated_summary)


def _heuristic_summary_score(expected: str, generated: str) -> float:
    """
    Simple keyword-overlap heuristic for summary scoring.
    Used as fallback when LLM-as-judge is unavailable.
    """
    if not generated:
        return 1.0

    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())

    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                  "to", "for", "of", "in", "on", "at", "by", "with", "that",
                  "this", "their", "they", "from"}
    expected_words -= stop_words
    generated_words -= stop_words

    if not expected_words:
        return 3.0

    overlap = expected_words & generated_words
    overlap_ratio = len(overlap) / len(expected_words)

    # Map overlap ratio to 1-5 scale
    if overlap_ratio >= 0.7:
        return 5.0
    elif overlap_ratio >= 0.5:
        return 4.0
    elif overlap_ratio >= 0.3:
        return 3.0
    elif overlap_ratio >= 0.15:
        return 2.0
    else:
        return 1.0


def score_test_case(
    test_case: TestCase,
    result: ClassificationResult,
    client: Optional[genai.Client] = None,
    use_llm_judge: bool = True,
) -> TestCaseResult:
    """
    Score a single test case result across all dimensions.

    Args:
        test_case: The golden test case with expected outputs.
        result: The raw classification result from the LLM.
        client: Optional Gemini client for LLM-as-judge scoring.
        use_llm_judge: Whether to use LLM-as-judge for summary scoring.

    Returns:
        Fully scored TestCaseResult.
    """
    cat_match = score_category_match(result.category, test_case.expected_category)

    if use_llm_judge:
        summary_score = score_summary_relevance(
            test_case.input_email,
            test_case.expected_summary,
            result.summary,
            client=client,
        )
    else:
        summary_score = _heuristic_summary_score(
            test_case.expected_summary, result.summary
        )

    return TestCaseResult(
        test_case_id=test_case.id,
        input_email=test_case.input_email,
        expected_category=test_case.expected_category,
        expected_summary=test_case.expected_summary,
        predicted_category=result.category,
        predicted_summary=result.summary,
        category_match=cat_match,
        summary_relevance_score=summary_score,
        latency_ms=result.latency_ms,
        tokens_used=result.total_tokens,
        difficulty=test_case.difficulty,
    )


def aggregate_metrics(
    results: list[TestCaseResult],
) -> tuple[float, float, float, int, list[CategoryMetrics]]:
    """
    Aggregate scored results into overall and per-category metrics.

    Returns:
        Tuple of (overall_accuracy, avg_summary_score, avg_latency_ms,
                  total_tokens, per_category_metrics)
    """
    if not results:
        return 0.0, 0.0, 0.0, 0, []

    # Overall metrics
    total = len(results)
    passed = sum(1 for r in results if r.category_match)
    overall_accuracy = (passed / total) * 100

    summary_scores = [r.summary_relevance_score for r in results]
    avg_summary = statistics.mean(summary_scores) if summary_scores else 0.0

    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    total_tokens = sum(r.tokens_used for r in results)

    # Per-category metrics
    by_category: dict[EmailCategory, list[TestCaseResult]] = {}
    for r in results:
        by_category.setdefault(r.expected_category, []).append(r)

    category_metrics = []
    for cat, cat_results in sorted(by_category.items(), key=lambda x: x[0].value):
        cat_total = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r.category_match)
        cat_accuracy = (cat_correct / cat_total) * 100 if cat_total > 0 else 0.0
        cat_summary_scores = [r.summary_relevance_score for r in cat_results]
        cat_avg_summary = statistics.mean(cat_summary_scores) if cat_summary_scores else 0.0
        cat_latencies = [r.latency_ms for r in cat_results if r.latency_ms > 0]
        cat_avg_latency = statistics.mean(cat_latencies) if cat_latencies else 0.0

        category_metrics.append(CategoryMetrics(
            category=cat,
            total=cat_total,
            correct=cat_correct,
            accuracy=cat_accuracy,
            avg_summary_score=cat_avg_summary,
            avg_latency_ms=cat_avg_latency,
        ))

    return overall_accuracy, avg_summary, avg_latency, total_tokens, category_metrics
