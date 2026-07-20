"""
Evaluation test runner.

Orchestrates the end-to-end evaluation: loads the golden dataset,
runs every test case through the classifier, scores results, and
produces an EvalRunSummary. Uses async batching with rate limiting
to respect Gemini free-tier limits.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai

from src.classifier import classify_email, classify_email_async
from src.config import (
    EvalRunSummary,
    GoldenDataset,
    PromptConfig,
    TestCase,
    TestCaseResult,
)
from src.scoring import aggregate_metrics, score_test_case


# Default path to the golden dataset
_DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "golden_dataset",
    "dataset_v1.json",
)


def load_dataset(path: str = _DEFAULT_DATASET_PATH) -> GoldenDataset:
    """
    Load and validate the golden dataset from a JSON file.

    Args:
        path: Absolute or relative path to the dataset JSON file.

    Returns:
        Validated GoldenDataset instance.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        ValidationError: If the JSON doesn't match the expected schema.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return GoldenDataset(**data)


def run_evaluation_sync(
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    use_llm_judge: bool = True,
    rate_limit_rpm: int = 15,
    verbose: bool = True,
) -> EvalRunSummary:
    """
    Run evaluation synchronously with rate limiting.

    This is the simpler, more reliable approach that respects Gemini
    free-tier rate limits (15 RPM default). For each test case:
    1. Classify the email
    2. Score the result (optionally with LLM-as-judge)
    3. Wait to stay within rate limits

    Args:
        prompt_config: The prompt version to evaluate.
        dataset: Golden dataset with test cases.
        use_llm_judge: Whether to use LLM-as-judge for summary scoring.
        rate_limit_rpm: Max requests per minute (Gemini free tier = 15).
        verbose: Print progress updates.

    Returns:
        Complete EvalRunSummary with all results and metrics.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    # Calculate delay between requests to stay within rate limit
    # Account for LLM-as-judge calls (2 API calls per test case if enabled)
    calls_per_case = 2 if use_llm_judge else 1
    delay_seconds = (60.0 / rate_limit_rpm) * calls_per_case

    results: list[TestCaseResult] = []
    total = dataset.size
    errors = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluation Run: {prompt_config.prompt_fingerprint()}")
        print(f"  Dataset: {dataset.version} ({total} cases)")
        print(f"  Rate limit: {rate_limit_rpm} RPM")
        print(f"  LLM-as-judge: {'enabled' if use_llm_judge else 'disabled'}")
        print(f"{'='*60}\n")

    for i, test_case in enumerate(dataset.test_cases, 1):
        if verbose:
            print(f"  [{i:3d}/{total}] {test_case.id} ({test_case.difficulty.value}) ... ", end="", flush=True)

        try:
            # Classify
            classification = classify_email(
                email_text=test_case.input_email,
                prompt_config=prompt_config,
                client=client,
            )

            # Score
            scored = score_test_case(
                test_case=test_case,
                result=classification,
                client=client if use_llm_judge else None,
                use_llm_judge=use_llm_judge,
            )

            results.append(scored)

            if verbose:
                status = "✓" if scored.category_match else "✗"
                print(
                    f"{status} {scored.predicted_category.value:10s} "
                    f"(expected: {scored.expected_category.value:10s}) "
                    f"summary: {scored.summary_relevance_score:.0f}/5 "
                    f"latency: {scored.latency_ms:.0f}ms"
                )

        except Exception as e:
            errors += 1
            error_result = TestCaseResult(
                test_case_id=test_case.id,
                input_email=test_case.input_email,
                expected_category=test_case.expected_category,
                expected_summary=test_case.expected_summary,
                error=str(e),
                difficulty=test_case.difficulty,
            )
            results.append(error_result)

            if verbose:
                print(f"ERROR: {e}")

        # Rate limiting delay
        if i < total:
            time.sleep(delay_seconds)

    # Aggregate metrics
    overall_accuracy, avg_summary, avg_latency, total_tokens, per_category = (
        aggregate_metrics(results)
    )

    passed = sum(1 for r in results if r.category_match)
    failed = sum(1 for r in results if not r.category_match and r.error is None)

    summary = EvalRunSummary(
        prompt_version=prompt_config.version,
        model=prompt_config.model,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        error_cases=errors,
        overall_accuracy=overall_accuracy,
        avg_summary_score=avg_summary,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        per_category=per_category,
        results=results,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} passed ({overall_accuracy:.1f}%)")
        print(f"  Avg summary score: {avg_summary:.2f}/5")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  Total tokens: {total_tokens}")
        if errors > 0:
            print(f"  Errors: {errors}")
        print(f"{'='*60}\n")

    return summary


async def run_evaluation_async(
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    use_llm_judge: bool = True,
    concurrency: int = 5,
    rate_limit_rpm: int = 15,
    verbose: bool = True,
) -> EvalRunSummary:
    """
    Run evaluation with async batching and rate limiting.

    Uses a semaphore for concurrency control and a token bucket
    for rate limiting. Better throughput than sync for large datasets,
    but more complex.

    Args:
        prompt_config: The prompt version to evaluate.
        dataset: Golden dataset with test cases.
        use_llm_judge: Whether to use LLM-as-judge for summary scoring.
        concurrency: Max concurrent requests.
        rate_limit_rpm: Max requests per minute.
        verbose: Print progress updates.

    Returns:
        Complete EvalRunSummary with all results and metrics.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    semaphore = asyncio.Semaphore(concurrency)

    # Simple rate limiter
    calls_per_case = 2 if use_llm_judge else 1
    delay_seconds = (60.0 / rate_limit_rpm) * calls_per_case
    rate_lock = asyncio.Lock()

    total = dataset.size
    completed = 0
    results: list[TestCaseResult] = []

    if verbose:
        print(f"\n  Running async eval: {total} cases, concurrency={concurrency}")

    async def process_case(test_case: TestCase) -> TestCaseResult:
        nonlocal completed
        async with semaphore:
            # Rate limiting
            async with rate_lock:
                await asyncio.sleep(delay_seconds)

            try:
                classification = await classify_email_async(
                    email_text=test_case.input_email,
                    prompt_config=prompt_config,
                    client=client,
                )

                scored = score_test_case(
                    test_case=test_case,
                    result=classification,
                    client=client if use_llm_judge else None,
                    use_llm_judge=use_llm_judge,
                )

                completed += 1
                if verbose:
                    status = "✓" if scored.category_match else "✗"
                    print(f"  [{completed}/{total}] {test_case.id} {status}")

                return scored

            except Exception as e:
                completed += 1
                if verbose:
                    print(f"  [{completed}/{total}] {test_case.id} ERROR: {e}")

                return TestCaseResult(
                    test_case_id=test_case.id,
                    input_email=test_case.input_email,
                    expected_category=test_case.expected_category,
                    expected_summary=test_case.expected_summary,
                    error=str(e),
                    difficulty=test_case.difficulty,
                )

    # Run all cases
    tasks = [process_case(tc) for tc in dataset.test_cases]
    results = await asyncio.gather(*tasks)
    results = list(results)

    # Aggregate
    overall_accuracy, avg_summary, avg_latency, total_tokens, per_category = (
        aggregate_metrics(results)
    )

    passed = sum(1 for r in results if r.category_match)
    failed = sum(1 for r in results if not r.category_match and r.error is None)
    errors = sum(1 for r in results if r.error is not None)

    return EvalRunSummary(
        prompt_version=prompt_config.version,
        model=prompt_config.model,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        error_cases=errors,
        overall_accuracy=overall_accuracy,
        avg_summary_score=avg_summary,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        per_category=per_category,
        results=results,
    )
