"""
Slack alerting integration.

Sends structured alert messages to Slack via incoming webhooks
when evaluation results warrant attention. Uses Block Kit for
rich formatting with status indicators and key metrics.

Gracefully degrades to console logging when no webhook URL
is configured.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

from src.config import (
    ComparisonResult,
    DriftAlert,
    EvalRunSummary,
    EvalStatus,
)


logger = logging.getLogger(__name__)


def send_slack_alert(
    current_run: EvalRunSummary,
    comparison: ComparisonResult | None = None,
    drift_alerts: list[DriftAlert] | None = None,
    report_path: str | None = None,
    webhook_url: str | None = None,
) -> bool:
    """
    Send an evaluation alert to Slack.

    Args:
        current_run: The current evaluation run summary.
        comparison: Optional comparison result with baseline.
        drift_alerts: Optional list of drift detection alerts.
        report_path: Optional path/URL to the full HTML report.
        webhook_url: Slack webhook URL. Falls back to SLACK_WEBHOOK_URL env var.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    if not url:
        logger.info("No Slack webhook URL configured. Logging alert to console.")
        _log_to_console(current_run, comparison, drift_alerts)
        return False

    payload = _build_payload(current_run, comparison, drift_alerts, report_path)

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            logger.info("Slack alert sent successfully.")
            return True
        else:
            logger.error(
                f"Slack webhook returned {response.status_code}: {response.text}"
            )
            return False

    except requests.RequestException as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def _build_payload(
    current_run: EvalRunSummary,
    comparison: ComparisonResult | None,
    drift_alerts: list[DriftAlert] | None,
    report_path: str | None,
) -> dict:
    """Build the Slack Block Kit message payload."""
    status = comparison.status if comparison else EvalStatus.PASS
    status_config = {
        EvalStatus.PASS: {"emoji": "✅", "color": "#22c55e", "label": "PASS"},
        EvalStatus.WARNING: {"emoji": "⚠️", "color": "#eab308", "label": "WARNING"},
        EvalStatus.CRITICAL: {"emoji": "🔴", "color": "#ef4444", "label": "CRITICAL"},
    }[status]

    blocks = []

    # Header
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{status_config['emoji']} Model Eval: {status_config['label']}",
        },
    })

    # Summary section
    accuracy_str = f"{current_run.overall_accuracy:.1f}%"
    delta_str = ""
    if comparison:
        delta = comparison.accuracy_delta
        delta_str = f" ({'+' if delta > 0 else ''}{delta:.1f}%)"

    summary_text = (
        f"*Prompt:* `{current_run.prompt_version}` → *Model:* `{current_run.model}`\n"
        f"*Accuracy:* {accuracy_str}{delta_str} | "
        f"*Summary Score:* {current_run.avg_summary_score:.2f}/5 | "
        f"*Latency:* {current_run.avg_latency_ms:.0f}ms\n"
        f"*Cases:* {current_run.passed_cases}/{current_run.total_cases} passed"
    )

    if current_run.error_cases > 0:
        summary_text += f" | *Errors:* {current_run.error_cases}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary_text},
    })

    # Regression/Improvement details
    if comparison:
        details = []
        if comparison.regressions:
            reg_ids = ", ".join(r.test_case_id for r in comparison.regressions[:5])
            details.append(
                f"🔻 *{len(comparison.regressions)} regression(s):* {reg_ids}"
            )
        if comparison.improvements:
            imp_ids = ", ".join(r.test_case_id for r in comparison.improvements[:5])
            details.append(
                f"🔺 *{len(comparison.improvements)} improvement(s):* {imp_ids}"
            )
        if comparison.message:
            details.append(f"💬 {comparison.message}")

        if details:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(details)},
            })

    # Per-category breakdown
    if current_run.per_category:
        cat_lines = []
        for cm in current_run.per_category:
            bar = "█" * int(cm.accuracy / 10) + "░" * (10 - int(cm.accuracy / 10))
            delta_cat = ""
            if comparison and cm.category.value in comparison.per_category_deltas:
                d = comparison.per_category_deltas[cm.category.value]
                delta_cat = f" ({'+' if d > 0 else ''}{d:.0f}%)"
            cat_lines.append(
                f"`{cm.category.value:10s}` {bar} {cm.accuracy:.0f}%{delta_cat}"
            )

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Per-Category Accuracy:*\n" + "\n".join(cat_lines),
            },
        })

    # Drift alerts
    if drift_alerts:
        active_alerts = [a for a in drift_alerts if a.triggered]
        if active_alerts:
            blocks.append({"type": "divider"})
            drift_text = "*📉 Drift Alerts:*\n"
            for alert in active_alerts:
                drift_text += f"• {alert.message}\n"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": drift_text},
            })

    # Report link
    if report_path:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"📄 <{report_path}|View Full Report>",
            },
        })

    # Context footer
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    f"Run ID: `{current_run.run_id}` | "
                    f"{current_run.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
                ),
            }
        ],
    })

    return {
        "text": (
            f"Model Eval {status_config['label']}: "
            f"{current_run.prompt_version} — "
            f"{current_run.overall_accuracy:.1f}% accuracy"
        ),
        "blocks": blocks,
    }


def _log_to_console(
    current_run: EvalRunSummary,
    comparison: ComparisonResult | None,
    drift_alerts: list[DriftAlert] | None,
) -> None:
    """Log the alert content to console when Slack is not configured."""
    status = comparison.status.value.upper() if comparison else "PASS"
    status_emoji = {"PASS": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(status, "⚪")

    print(f"\n{'='*60}")
    print(f"  {status_emoji} ALERT: {status}")
    print(f"{'='*60}")
    print(f"  Prompt: {current_run.prompt_version}")
    print(f"  Model:  {current_run.model}")
    print(f"  Accuracy: {current_run.overall_accuracy:.1f}%")
    print(f"  Summary Score: {current_run.avg_summary_score:.2f}/5")
    print(f"  Cases: {current_run.passed_cases}/{current_run.total_cases}")

    if comparison:
        print(f"  Delta: {comparison.accuracy_delta:+.1f}%")
        if comparison.regressions:
            print(f"  Regressions: {len(comparison.regressions)}")
        if comparison.improvements:
            print(f"  Improvements: {len(comparison.improvements)}")
        print(f"  Message: {comparison.message}")

    if drift_alerts:
        for alert in drift_alerts:
            if alert.triggered:
                print(f"  ⚠️  DRIFT: {alert.message}")

    print(f"{'='*60}\n")
