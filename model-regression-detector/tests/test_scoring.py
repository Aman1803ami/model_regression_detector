"""
Tests for the scoring module.

Tests category matching, heuristic summary scoring, and
metric aggregation logic.
"""

from __future__ import annotations

import pytest

from src.config import (
    CategoryMetrics,
    ClassificationResult,
    Difficulty,
    EmailCategory,
    TestCase,
    TestCaseResult,
)
from src.scoring import (
    _heuristic_summary_score,
    aggregate_metrics,
    score_category_match,
    score_test_case,
)


# ---------------------------------------------------------------------------
# Tests for score_category_match
# ---------------------------------------------------------------------------

class TestCategoryMatch:
    def test_exact_match(self):
        assert score_category_match(EmailCategory.BILLING, EmailCategory.BILLING) is True

    def test_mismatch(self):
        assert score_category_match(EmailCategory.BILLING, EmailCategory.TECHNICAL) is False

    def test_none_predicted(self):
        assert score_category_match(None, EmailCategory.BILLING) is False

    def test_all_categories(self):
        for cat in EmailCategory:
            assert score_category_match(cat, cat) is True


# ---------------------------------------------------------------------------
# Tests for _heuristic_summary_score
# ---------------------------------------------------------------------------

class TestHeuristicSummaryScore:
    def test_identical_summaries(self):
        score = _heuristic_summary_score(
            "Customer reports being double charged for subscription",
            "Customer reports being double charged for subscription",
        )
        assert score >= 4.0

    def test_high_overlap(self):
        score = _heuristic_summary_score(
            "Customer reports double charge and requests refund",
            "Customer reports being charged twice and wants a refund",
        )
        assert score >= 3.0

    def test_no_overlap(self):
        score = _heuristic_summary_score(
            "Customer reports billing issue with subscription",
            "Application crashes when opening dashboard",
        )
        assert score <= 2.0

    def test_empty_generated(self):
        score = _heuristic_summary_score(
            "Customer reports an issue",
            "",
        )
        assert score == 1.0

    def test_empty_expected(self):
        score = _heuristic_summary_score(
            "",
            "Some generated summary",
        )
        assert score == 3.0  # Fallback for empty expected


# ---------------------------------------------------------------------------
# Tests for score_test_case
# ---------------------------------------------------------------------------

class TestScoreTestCase:
    def test_correct_classification(self):
        test_case = TestCase(
            id="TC-001",
            input_email="I was charged twice",
            expected_category=EmailCategory.BILLING,
            expected_summary="Customer reports double charge",
        )
        result = ClassificationResult(
            category=EmailCategory.BILLING,
            summary="Customer reports being charged twice",
            latency_ms=150.0,
            total_tokens=100,
        )

        scored = score_test_case(test_case, result, use_llm_judge=False)

        assert scored.category_match is True
        assert scored.summary_relevance_score > 0
        assert scored.latency_ms == 150.0
        assert scored.tokens_used == 100

    def test_incorrect_classification(self):
        test_case = TestCase(
            id="TC-002",
            input_email="I was charged twice",
            expected_category=EmailCategory.BILLING,
            expected_summary="Customer reports double charge",
        )
        result = ClassificationResult(
            category=EmailCategory.TECHNICAL,
            summary="Technical issue reported",
            latency_ms=200.0,
        )

        scored = score_test_case(test_case, result, use_llm_judge=False)
        assert scored.category_match is False


# ---------------------------------------------------------------------------
# Tests for aggregate_metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def _make_result(
        self,
        tc_id: str,
        category: EmailCategory,
        match: bool,
        score: float = 3.0,
        latency: float = 100.0,
        tokens: int = 50,
    ) -> TestCaseResult:
        return TestCaseResult(
            test_case_id=tc_id,
            input_email="test",
            expected_category=category,
            expected_summary="test",
            predicted_category=category if match else EmailCategory.GENERAL,
            category_match=match,
            summary_relevance_score=score,
            latency_ms=latency,
            tokens_used=tokens,
        )

    def test_perfect_accuracy(self):
        results = [
            self._make_result("TC-001", EmailCategory.BILLING, True),
            self._make_result("TC-002", EmailCategory.TECHNICAL, True),
        ]
        accuracy, _, _, _, _ = aggregate_metrics(results)
        assert accuracy == 100.0

    def test_zero_accuracy(self):
        results = [
            self._make_result("TC-001", EmailCategory.BILLING, False),
            self._make_result("TC-002", EmailCategory.TECHNICAL, False),
        ]
        accuracy, _, _, _, _ = aggregate_metrics(results)
        assert accuracy == 0.0

    def test_mixed_accuracy(self):
        results = [
            self._make_result("TC-001", EmailCategory.BILLING, True),
            self._make_result("TC-002", EmailCategory.BILLING, False),
            self._make_result("TC-003", EmailCategory.TECHNICAL, True),
            self._make_result("TC-004", EmailCategory.TECHNICAL, True),
        ]
        accuracy, _, _, _, _ = aggregate_metrics(results)
        assert accuracy == 75.0

    def test_per_category_metrics(self):
        results = [
            self._make_result("TC-001", EmailCategory.BILLING, True, score=5.0),
            self._make_result("TC-002", EmailCategory.BILLING, True, score=3.0),
            self._make_result("TC-003", EmailCategory.TECHNICAL, False, score=1.0),
        ]
        _, _, _, _, per_cat = aggregate_metrics(results)

        assert len(per_cat) == 2

        billing = next(c for c in per_cat if c.category == EmailCategory.BILLING)
        assert billing.total == 2
        assert billing.correct == 2
        assert billing.accuracy == 100.0
        assert billing.avg_summary_score == 4.0

        tech = next(c for c in per_cat if c.category == EmailCategory.TECHNICAL)
        assert tech.total == 1
        assert tech.correct == 0
        assert tech.accuracy == 0.0

    def test_empty_results(self):
        accuracy, avg_summary, avg_latency, tokens, per_cat = aggregate_metrics([])
        assert accuracy == 0.0
        assert avg_summary == 0.0
        assert avg_latency == 0.0
        assert tokens == 0
        assert per_cat == []

    def test_total_tokens(self):
        results = [
            self._make_result("TC-001", EmailCategory.BILLING, True, tokens=100),
            self._make_result("TC-002", EmailCategory.TECHNICAL, True, tokens=200),
        ]
        _, _, _, total_tokens, _ = aggregate_metrics(results)
        assert total_tokens == 300
