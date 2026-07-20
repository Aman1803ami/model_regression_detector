"""
Drift detector.

Monitors rolling averages of evaluation metrics over time to catch
slow degradation that individual per-run comparisons might miss.
A single run might pass all thresholds, but if scores are trending
downward over many runs, this module raises a "slow drift" warning.
"""

from __future__ import annotations

import statistics

from src.config import DriftAlert, EvalRunSummary, ThresholdConfig


def detect_drift(
    run_history: list[EvalRunSummary],
    thresholds: ThresholdConfig | None = None,
) -> list[DriftAlert]:
    """
    Analyze the history of evaluation runs for slow quality drift.

    Calculates a rolling average of key metrics over the last N runs
    (configured by thresholds.drift_window) and checks if the average
    has dropped below configured floors.

    This catches gradual degradation patterns like:
    - Scores slowly declining 1-2% per run, never enough to trigger
      a per-run alert, but eventually dropping from 95% to 80%
    - Model provider silently changing behavior over time

    Args:
        run_history: List of past EvalRunSummaries, ordered oldest to newest.
        thresholds: Threshold configuration for drift detection.

    Returns:
        List of DriftAlerts for any metrics that are drifting below floors.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()

    alerts: list[DriftAlert] = []

    if len(run_history) < thresholds.drift_window:
        # Not enough data to detect drift
        return alerts

    # Use the last N runs for the rolling window
    window = run_history[-thresholds.drift_window:]

    # Check accuracy drift
    accuracy_values = [run.overall_accuracy for run in window]
    accuracy_avg = statistics.mean(accuracy_values)

    if accuracy_avg < thresholds.drift_accuracy_floor:
        alerts.append(DriftAlert(
            triggered=True,
            metric="overall_accuracy",
            current_moving_avg=round(accuracy_avg, 2),
            threshold=thresholds.drift_accuracy_floor,
            window_size=thresholds.drift_window,
            message=(
                f"Accuracy drift detected: {thresholds.drift_window}-run moving average "
                f"is {accuracy_avg:.1f}%, below floor of {thresholds.drift_accuracy_floor}%. "
                f"Recent values: {[round(v, 1) for v in accuracy_values]}"
            ),
        ))

    # Check summary score drift
    summary_values = [run.avg_summary_score for run in window]
    summary_avg = statistics.mean(summary_values)

    if summary_avg < thresholds.drift_summary_floor:
        alerts.append(DriftAlert(
            triggered=True,
            metric="avg_summary_score",
            current_moving_avg=round(summary_avg, 2),
            threshold=thresholds.drift_summary_floor,
            window_size=thresholds.drift_window,
            message=(
                f"Summary quality drift detected: {thresholds.drift_window}-run moving average "
                f"is {summary_avg:.2f}/5, below floor of {thresholds.drift_summary_floor}/5. "
                f"Recent values: {[round(v, 2) for v in summary_values]}"
            ),
        ))

    # Check for downward trend even if above floor
    # (early warning before hitting the floor)
    if len(accuracy_values) >= 3:
        # Simple linear trend check: are the last 3 values each lower than the previous?
        recent_3 = accuracy_values[-3:]
        if recent_3[0] > recent_3[1] > recent_3[2]:
            trend_drop = recent_3[0] - recent_3[2]
            if trend_drop >= thresholds.warning_delta_pct:
                alerts.append(DriftAlert(
                    triggered=True,
                    metric="accuracy_trend",
                    current_moving_avg=round(accuracy_avg, 2),
                    threshold=thresholds.drift_accuracy_floor,
                    window_size=3,
                    message=(
                        f"Downward trend warning: accuracy dropped {trend_drop:.1f}% "
                        f"over last 3 runs ({recent_3[0]:.1f}% → {recent_3[1]:.1f}% → "
                        f"{recent_3[2]:.1f}%). May breach floor if trend continues."
                    ),
                ))

    return alerts


def format_drift_alerts(alerts: list[DriftAlert]) -> str:
    """Format drift alerts for CLI output or logging."""
    if not alerts:
        return "  No drift detected."

    lines = [
        "\n  ⚠️  DRIFT ALERTS:",
        "  " + "-" * 50,
    ]

    for alert in alerts:
        lines.append(f"  • {alert.message}")

    lines.append("  " + "-" * 50)
    return "\n".join(lines)
