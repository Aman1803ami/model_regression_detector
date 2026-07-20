"""
Regression comparator.

Compares two evaluation runs to detect regressions (pass → fail),
improvements (fail → pass), and overall quality deltas. Classifies
the comparison as PASS, WARNING, or CRITICAL based on configurable
thresholds.
"""

from __future__ import annotations

from src.config import (
    ComparisonResult,
    EvalRunSummary,
    EvalStatus,
    ImprovementCase,
    RegressionCase,
    ThresholdConfig,
)


def compare_runs(
    current: EvalRunSummary,
    baseline: EvalRunSummary,
    thresholds: ThresholdConfig | None = None,
) -> ComparisonResult:
    """
    Compare the current eval run against a baseline to detect regressions.

    For each test case, checks if it flipped from pass to fail (regression)
    or fail to pass (improvement). Then computes overall deltas and
    classifies the result based on threshold configuration.

    Args:
        current: The new evaluation run to assess.
        baseline: The baseline/previous run to compare against.
        thresholds: Threshold config for WARNING/CRITICAL classification.

    Returns:
        ComparisonResult with regressions, improvements, deltas, and status.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()

    # Build lookup maps: test_case_id -> result
    baseline_map = {r.test_case_id: r for r in baseline.results}
    current_map = {r.test_case_id: r for r in current.results}

    # Find all test case IDs present in both runs
    common_ids = set(baseline_map.keys()) & set(current_map.keys())

    regressions: list[RegressionCase] = []
    improvements: list[ImprovementCase] = []

    for tc_id in sorted(common_ids):
        old = baseline_map[tc_id]
        new = current_map[tc_id]

        # Regression: was correct, now wrong
        if old.category_match and not new.category_match:
            regressions.append(RegressionCase(
                test_case_id=tc_id,
                input_email=old.input_email[:200],  # Truncate for readability
                expected_category=old.expected_category,
                expected_summary=old.expected_summary[:200],
                old_predicted_category=old.predicted_category,
                old_predicted_summary=old.predicted_summary[:200],
                new_predicted_category=new.predicted_category,
                new_predicted_summary=new.predicted_summary[:200],
                old_summary_score=old.summary_relevance_score,
                new_summary_score=new.summary_relevance_score,
            ))

        # Improvement: was wrong, now correct
        elif not old.category_match and new.category_match:
            improvements.append(ImprovementCase(
                test_case_id=tc_id,
                input_email=old.input_email[:200],
                expected_category=old.expected_category,
                old_predicted_category=old.predicted_category,
                new_predicted_category=new.predicted_category,
            ))

    # Calculate deltas
    accuracy_delta = current.overall_accuracy - baseline.overall_accuracy
    summary_delta = current.avg_summary_score - baseline.avg_summary_score

    # Per-category deltas
    baseline_cat_map = {cm.category.value: cm.accuracy for cm in baseline.per_category}
    current_cat_map = {cm.category.value: cm.accuracy for cm in current.per_category}
    all_categories = set(baseline_cat_map.keys()) | set(current_cat_map.keys())
    per_category_deltas = {}
    for cat in sorted(all_categories):
        old_acc = baseline_cat_map.get(cat, 0.0)
        new_acc = current_cat_map.get(cat, 0.0)
        per_category_deltas[cat] = new_acc - old_acc

    # Determine status based on thresholds
    status = EvalStatus.PASS
    messages: list[str] = []

    # Check accuracy delta
    if accuracy_delta <= -thresholds.critical_delta_pct:
        status = EvalStatus.CRITICAL
        messages.append(
            f"Accuracy dropped {abs(accuracy_delta):.1f}% "
            f"(>{thresholds.critical_delta_pct}% critical threshold)"
        )
    elif accuracy_delta <= -thresholds.warning_delta_pct:
        status = EvalStatus.WARNING
        messages.append(
            f"Accuracy dropped {abs(accuracy_delta):.1f}% "
            f"(>{thresholds.warning_delta_pct}% warning threshold)"
        )

    # Check regression count (with noise suppression)
    num_regressions = len(regressions)
    if num_regressions >= thresholds.min_flips_for_signal:
        if status == EvalStatus.PASS and num_regressions >= 3:
            status = EvalStatus.WARNING
        messages.append(f"{num_regressions} test case(s) regressed")
    elif num_regressions > 0:
        messages.append(
            f"{num_regressions} regression(s) detected but below noise threshold "
            f"({thresholds.min_flips_for_signal})"
        )

    # Note improvements
    if improvements:
        messages.append(f"{len(improvements)} test case(s) improved")

    # Build summary message
    if not messages:
        messages.append("No significant changes detected")

    message = "; ".join(messages)

    return ComparisonResult(
        status=status,
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_accuracy=baseline.overall_accuracy,
        current_accuracy=current.overall_accuracy,
        accuracy_delta=accuracy_delta,
        baseline_summary_score=baseline.avg_summary_score,
        current_summary_score=current.avg_summary_score,
        summary_score_delta=summary_delta,
        regressions=regressions,
        improvements=improvements,
        per_category_deltas=per_category_deltas,
        message=message,
    )


def format_comparison_summary(comparison: ComparisonResult) -> str:
    """
    Format a human-readable summary of the comparison result.

    Useful for CLI output and logging.
    """
    status_emoji = {
        EvalStatus.PASS: "🟢",
        EvalStatus.WARNING: "🟡",
        EvalStatus.CRITICAL: "🔴",
    }

    lines = [
        f"\n{'='*60}",
        f"  {status_emoji[comparison.status]} Comparison: {comparison.status.value.upper()}",
        f"{'='*60}",
        f"  Baseline run:  {comparison.baseline_run_id}",
        f"  Current run:   {comparison.current_run_id}",
        f"  Accuracy:      {comparison.baseline_accuracy:.1f}% → {comparison.current_accuracy:.1f}% "
        f"({comparison.accuracy_delta:+.1f}%)",
        f"  Summary score: {comparison.baseline_summary_score:.2f} → {comparison.current_summary_score:.2f} "
        f"({comparison.summary_score_delta:+.2f})",
        f"  Regressions:   {len(comparison.regressions)}",
        f"  Improvements:  {len(comparison.improvements)}",
    ]

    if comparison.per_category_deltas:
        lines.append(f"\n  Per-category accuracy deltas:")
        for cat, delta in comparison.per_category_deltas.items():
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            lines.append(f"    {cat:12s}: {delta:+.1f}% {arrow}")

    if comparison.regressions:
        lines.append(f"\n  Regressed test cases:")
        for reg in comparison.regressions:
            lines.append(
                f"    {reg.test_case_id}: "
                f"{reg.old_predicted_category.value if reg.old_predicted_category else '?'} → "
                f"{reg.new_predicted_category.value if reg.new_predicted_category else '?'} "
                f"(expected: {reg.expected_category.value})"
            )

    lines.append(f"\n  Message: {comparison.message}")
    lines.append(f"{'='*60}\n")

    return "\n".join(lines)
