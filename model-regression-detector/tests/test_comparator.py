"""
Tests for the regression comparator.

Tests regression detection, improvement detection, threshold
classification, and noise suppression logic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.comparator import compare_runs
from src.config import (
    CategoryMetrics,
    EmailCategory,
    EvalRunSummary,
    EvalStatus,
    TestCaseResult,
    ThresholdConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tc_id: str,
    category: EmailCategory,
    match: bool,
    summary: str = "test",
) -> TestCaseResult:
    return TestCaseResult(
        test_case_id=tc_id,
        input_email=f"Test email for {tc_id}",
        expected_category=category,
        expected_summary="Expected summary",
        predicted_category=category if match else EmailCategory.GENERAL,
        predicted_summary=summary,
        category_match=match,
        summary_relevance_score=4.0 if match else 2.0,
    )


def _make_run(
    run_id: str,
    results: list[TestCaseResult],
    accuracy: float | None = None,
) -> EvalRunSummary:
    passed = sum(1 for r in results if r.category_match)
    total = len(results)
    calc_accuracy = (passed / total) * 100 if total > 0 else 0.0

    return EvalRunSummary(
        run_id=run_id,
        prompt_version="test",
        model="test-model",
        timestamp=datetime.now(timezone.utc),
        total_cases=total,
        passed_cases=passed,
        failed_cases=total - passed,
        overall_accuracy=accuracy if accuracy is not None else calc_accuracy,
        avg_summary_score=3.5,
        results=results,
        per_category=[
            CategoryMetrics(
                category=EmailCategory.BILLING,
                total=total,
                correct=passed,
                accuracy=calc_accuracy,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_no_changes(self):
        """Identical runs should produce PASS with no regressions."""
        results = [
            _make_result("TC-001", EmailCategory.BILLING, True),
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
            _make_result("TC-003", EmailCategory.ACCOUNT, False),
        ]
        baseline = _make_run("baseline", results)
        current = _make_run("current", results)

        comparison = compare_runs(current, baseline)

        assert comparison.status == EvalStatus.PASS
        assert len(comparison.regressions) == 0
        assert len(comparison.improvements) == 0

    def test_regression_detected(self):
        """A case flipping from pass to fail should be detected."""
        baseline_results = [
            _make_result("TC-001", EmailCategory.BILLING, True),
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
        ]
        current_results = [
            _make_result("TC-001", EmailCategory.BILLING, True),
            _make_result("TC-002", EmailCategory.TECHNICAL, False),
        ]
        baseline = _make_run("baseline", baseline_results, accuracy=100.0)
        current = _make_run("current", current_results, accuracy=50.0)

        thresholds = ThresholdConfig(min_flips_for_signal=1)
        comparison = compare_runs(current, baseline, thresholds)

        assert len(comparison.regressions) == 1
        assert comparison.regressions[0].test_case_id == "TC-002"

    def test_improvement_detected(self):
        """A case flipping from fail to pass should be detected."""
        baseline_results = [
            _make_result("TC-001", EmailCategory.BILLING, False),
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
        ]
        current_results = [
            _make_result("TC-001", EmailCategory.BILLING, True),
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
        ]
        baseline = _make_run("baseline", baseline_results, accuracy=50.0)
        current = _make_run("current", current_results, accuracy=100.0)

        comparison = compare_runs(current, baseline)

        assert len(comparison.improvements) == 1
        assert comparison.improvements[0].test_case_id == "TC-001"

    def test_critical_threshold(self):
        """Large accuracy drops should trigger CRITICAL status."""
        baseline_results = [_make_result(f"TC-{i:03d}", EmailCategory.BILLING, True) for i in range(10)]
        current_results = [_make_result(f"TC-{i:03d}", EmailCategory.BILLING, i < 8) for i in range(10)]

        baseline = _make_run("baseline", baseline_results, accuracy=100.0)
        current = _make_run("current", current_results, accuracy=80.0)

        thresholds = ThresholdConfig(critical_delta_pct=8.0)
        comparison = compare_runs(current, baseline, thresholds)

        assert comparison.status == EvalStatus.CRITICAL
        assert comparison.accuracy_delta == -20.0

    def test_warning_threshold(self):
        """Moderate accuracy drops should trigger WARNING status."""
        baseline_results = [_make_result(f"TC-{i:03d}", EmailCategory.BILLING, True) for i in range(20)]
        current_results = [_make_result(f"TC-{i:03d}", EmailCategory.BILLING, i < 19) for i in range(20)]

        baseline = _make_run("baseline", baseline_results, accuracy=100.0)
        current = _make_run("current", current_results, accuracy=95.0)

        thresholds = ThresholdConfig(warning_delta_pct=3.0, critical_delta_pct=8.0)
        comparison = compare_runs(current, baseline, thresholds)

        assert comparison.status == EvalStatus.WARNING

    def test_noise_suppression(self):
        """Small number of flips below threshold should not change status."""
        baseline_results = [
            _make_result("TC-001", EmailCategory.BILLING, True),
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
            *[_make_result(f"TC-{i:03d}", EmailCategory.GENERAL, True) for i in range(3, 83)],
        ]
        current_results = [
            _make_result("TC-001", EmailCategory.BILLING, False),  # 1 regression
            _make_result("TC-002", EmailCategory.TECHNICAL, True),
            *[_make_result(f"TC-{i:03d}", EmailCategory.GENERAL, True) for i in range(3, 83)],
        ]
        baseline = _make_run("baseline", baseline_results, accuracy=100.0)
        current = _make_run("current", current_results, accuracy=98.75)

        # With min_flips_for_signal=2, one regression is noise
        thresholds = ThresholdConfig(
            warning_delta_pct=3.0,
            min_flips_for_signal=2,
        )
        comparison = compare_runs(current, baseline, thresholds)

        # 1 regression exists but delta is only ~1.25%, below warning
        assert comparison.status == EvalStatus.PASS
        assert len(comparison.regressions) == 1  # Still detected, just not alarming

    def test_accuracy_delta_calculation(self):
        """Verify accuracy delta is computed correctly."""
        baseline = _make_run("baseline", [], accuracy=90.0)
        current = _make_run("current", [], accuracy=85.0)

        comparison = compare_runs(current, baseline)
        assert comparison.accuracy_delta == -5.0

    def test_per_category_deltas(self):
        """Per-category deltas should be computed."""
        baseline = _make_run("baseline", [
            _make_result("TC-001", EmailCategory.BILLING, True),
        ], accuracy=100.0)
        baseline.per_category = [
            CategoryMetrics(category=EmailCategory.BILLING, total=1, correct=1, accuracy=100.0),
        ]

        current = _make_run("current", [
            _make_result("TC-001", EmailCategory.BILLING, False),
        ], accuracy=0.0)
        current.per_category = [
            CategoryMetrics(category=EmailCategory.BILLING, total=1, correct=0, accuracy=0.0),
        ]

        comparison = compare_runs(current, baseline)
        assert "billing" in comparison.per_category_deltas
        assert comparison.per_category_deltas["billing"] == -100.0
