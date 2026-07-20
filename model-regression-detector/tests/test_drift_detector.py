"""
Tests for the drift detector module.

Tests rolling average calculation, floor breach detection,
downward trend detection, and edge cases.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.config import EvalRunSummary, ThresholdConfig
from src.drift_detector import detect_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(accuracy: float, summary_score: float = 4.0, days_ago: int = 0) -> EvalRunSummary:
    return EvalRunSummary(
        prompt_version="test",
        model="test-model",
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        overall_accuracy=accuracy,
        avg_summary_score=summary_score,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectDrift:
    def test_no_drift_healthy_scores(self):
        """Healthy scores should produce no alerts."""
        history = [_make_run(95.0, 4.5, days_ago=i) for i in range(7, 0, -1)]
        alerts = detect_drift(history, ThresholdConfig(drift_window=7))
        # No triggered alerts
        triggered = [a for a in alerts if a.triggered]
        assert len(triggered) == 0

    def test_accuracy_floor_breach(self):
        """Scores below the accuracy floor should trigger an alert."""
        history = [_make_run(80.0, 4.0, days_ago=i) for i in range(7, 0, -1)]
        thresholds = ThresholdConfig(drift_window=7, drift_accuracy_floor=85.0)
        alerts = detect_drift(history, thresholds)

        accuracy_alerts = [a for a in alerts if a.metric == "overall_accuracy"]
        assert len(accuracy_alerts) == 1
        assert accuracy_alerts[0].triggered is True
        assert accuracy_alerts[0].current_moving_avg == 80.0

    def test_summary_floor_breach(self):
        """Summary scores below floor should trigger an alert."""
        history = [_make_run(90.0, 2.5, days_ago=i) for i in range(7, 0, -1)]
        thresholds = ThresholdConfig(drift_window=7, drift_summary_floor=3.0)
        alerts = detect_drift(history, thresholds)

        summary_alerts = [a for a in alerts if a.metric == "avg_summary_score"]
        assert len(summary_alerts) == 1
        assert summary_alerts[0].triggered is True

    def test_insufficient_history(self):
        """Not enough runs for the window should return no alerts."""
        history = [_make_run(50.0, 1.0, days_ago=i) for i in range(3, 0, -1)]
        thresholds = ThresholdConfig(drift_window=7)
        alerts = detect_drift(history, thresholds)
        assert len(alerts) == 0

    def test_downward_trend_detection(self):
        """Three consecutive declining scores should trigger trend alert."""
        history = [
            _make_run(95.0, 4.5, days_ago=7),
            _make_run(94.0, 4.4, days_ago=6),
            _make_run(93.0, 4.3, days_ago=5),
            _make_run(92.0, 4.2, days_ago=4),
            _make_run(91.0, 4.1, days_ago=3),
            _make_run(89.0, 4.0, days_ago=2),
            _make_run(86.0, 3.9, days_ago=1),
        ]
        thresholds = ThresholdConfig(
            drift_window=7,
            drift_accuracy_floor=80.0,  # Above floor so floor alert won't fire
            warning_delta_pct=3.0,
        )
        alerts = detect_drift(history, thresholds)

        trend_alerts = [a for a in alerts if a.metric == "accuracy_trend"]
        assert len(trend_alerts) == 1
        assert "Downward trend" in trend_alerts[0].message

    def test_no_trend_when_stable(self):
        """Stable scores should not trigger trend alert."""
        history = [
            _make_run(90.0, 4.0, days_ago=7),
            _make_run(91.0, 4.0, days_ago=6),
            _make_run(90.0, 4.0, days_ago=5),
            _make_run(91.0, 4.0, days_ago=4),
            _make_run(90.0, 4.0, days_ago=3),
            _make_run(91.0, 4.0, days_ago=2),
            _make_run(90.0, 4.0, days_ago=1),
        ]
        thresholds = ThresholdConfig(drift_window=7, drift_accuracy_floor=80.0)
        alerts = detect_drift(history, thresholds)

        trend_alerts = [a for a in alerts if a.metric == "accuracy_trend"]
        assert len(trend_alerts) == 0

    def test_multiple_alerts_simultaneously(self):
        """Both accuracy and summary floors breached should produce two alerts."""
        history = [_make_run(70.0, 2.0, days_ago=i) for i in range(7, 0, -1)]
        thresholds = ThresholdConfig(
            drift_window=7,
            drift_accuracy_floor=85.0,
            drift_summary_floor=3.0,
        )
        alerts = detect_drift(history, thresholds)

        triggered = [a for a in alerts if a.triggered]
        metrics = {a.metric for a in triggered}
        assert "overall_accuracy" in metrics
        assert "avg_summary_score" in metrics

    def test_empty_history(self):
        """Empty history should return no alerts."""
        alerts = detect_drift([], ThresholdConfig(drift_window=7))
        assert len(alerts) == 0
