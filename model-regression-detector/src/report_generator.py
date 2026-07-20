"""
HTML diff report generator.

Creates rich, self-contained HTML reports showing:
- Run metadata and summary scorecard
- Side-by-side comparison of current vs baseline metrics
- Regression and improvement tables with old vs new outputs
- Trend charts showing scores over the last N runs
- Per-category breakdown with visual bars
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    ComparisonResult,
    DriftAlert,
    EvalRunSummary,
    EvalStatus,
)


def generate_report(
    current_run: EvalRunSummary,
    comparison: ComparisonResult | None = None,
    drift_alerts: list[DriftAlert] | None = None,
    run_history: list[EvalRunSummary] | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Generate a self-contained HTML diff report.

    Args:
        current_run: The current evaluation run.
        comparison: Optional comparison with baseline.
        drift_alerts: Optional drift detection alerts.
        run_history: Optional list of past runs for trend charts.
        output_dir: Directory to save the report. Defaults to ./reports/

    Returns:
        Absolute path to the generated HTML report.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "reports"
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = current_run.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{current_run.prompt_version}_{timestamp_str}.html"
    filepath = os.path.join(output_dir, filename)

    html = _render_html(current_run, comparison, drift_alerts, run_history)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def _render_html(
    current: EvalRunSummary,
    comparison: ComparisonResult | None,
    drift_alerts: list[DriftAlert] | None,
    run_history: list[EvalRunSummary] | None,
) -> str:
    """Render the full HTML report."""
    status = comparison.status if comparison else EvalStatus.PASS
    status_color = {
        EvalStatus.PASS: "#22c55e",
        EvalStatus.WARNING: "#eab308",
        EvalStatus.CRITICAL: "#ef4444",
    }[status]
    status_emoji = {
        EvalStatus.PASS: "✅",
        EvalStatus.WARNING: "⚠️",
        EvalStatus.CRITICAL: "🔴",
    }[status]
    status_label = status.value.upper()

    # Build sections
    header_section = _render_header(current, status_color, status_emoji, status_label)
    scorecard_section = _render_scorecard(current, comparison)
    category_section = _render_category_breakdown(current)
    regression_section = _render_regressions(comparison) if comparison else ""
    improvement_section = _render_improvements(comparison) if comparison else ""
    drift_section = _render_drift_alerts(drift_alerts) if drift_alerts else ""
    trend_section = _render_trend_chart(run_history) if run_history else ""
    results_section = _render_all_results(current)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Report: {current.prompt_version} | {status_label}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid {status_color};
        }}
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .header .meta {{
            color: #94a3b8;
            font-size: 0.875rem;
        }}
        .status-badge {{
            display: inline-block;
            background: {status_color}22;
            color: {status_color};
            border: 1px solid {status_color};
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-weight: 600;
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h2 {{
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #f1f5f9;
        }}
        .scorecard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric-card {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #f1f5f9;
        }}
        .metric-label {{
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}
        .metric-delta {{
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}
        .delta-positive {{ color: #22c55e; }}
        .delta-negative {{ color: #ef4444; }}
        .delta-neutral {{ color: #94a3b8; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        th {{
            text-align: left;
            padding: 0.75rem;
            background: #0f172a;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #334155;
        }}
        td {{
            padding: 0.75rem;
            border-bottom: 1px solid #1e293b;
            vertical-align: top;
        }}
        tr:hover td {{
            background: #334155;
        }}
        .category-bar {{
            height: 8px;
            border-radius: 4px;
            background: #334155;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .category-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .pass {{ color: #22c55e; }}
        .fail {{ color: #ef4444; }}
        .warn {{ color: #eab308; }}
        .diff-old {{ background: #7f1d1d33; color: #fca5a5; }}
        .diff-new {{ background: #14532d33; color: #86efac; }}
        .alert-box {{
            background: #7f1d1d22;
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .trend-chart {{
            width: 100%;
            height: 200px;
            position: relative;
        }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible::after {{
            content: ' ▸';
            font-size: 0.875rem;
        }}
        .collapsible.open::after {{
            content: ' ▾';
        }}
        .collapsible-content {{
            display: none;
        }}
        .collapsible-content.show {{
            display: block;
        }}
        .email-preview {{
            max-height: 100px;
            overflow-y: auto;
            font-size: 0.8rem;
            color: #94a3b8;
            padding: 0.5rem;
            background: #0f172a;
            border-radius: 4px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        {header_section}
        {scorecard_section}
        {drift_section}
        {category_section}
        {trend_section}
        {regression_section}
        {improvement_section}
        {results_section}
    </div>
    <script>
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', () => {{
                el.classList.toggle('open');
                const content = el.nextElementSibling;
                content.classList.toggle('show');
            }});
        }});
    </script>
</body>
</html>"""


def _render_header(
    run: EvalRunSummary,
    status_color: str,
    status_emoji: str,
    status_label: str,
) -> str:
    ts = run.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""
    <div class="header">
        <div class="status-badge">{status_emoji} {status_label}</div>
        <h1>Model Regression Report — {run.prompt_version}</h1>
        <div class="meta">
            Run ID: {run.run_id} &nbsp;|&nbsp;
            Model: {run.model} &nbsp;|&nbsp;
            Prompt: {run.prompt_version} &nbsp;|&nbsp;
            Timestamp: {ts} &nbsp;|&nbsp;
            Cases: {run.total_cases}
        </div>
    </div>"""


def _render_scorecard(
    current: EvalRunSummary,
    comparison: ComparisonResult | None,
) -> str:
    def delta_html(delta: float, unit: str = "%", reverse: bool = False) -> str:
        if comparison is None:
            return ""
        cls = "delta-neutral"
        if delta > 0.1:
            cls = "delta-negative" if reverse else "delta-positive"
        elif delta < -0.1:
            cls = "delta-positive" if reverse else "delta-negative"
        sign = "+" if delta > 0 else ""
        return f'<div class="metric-delta {cls}">{sign}{delta:.1f}{unit}</div>'

    acc_delta = comparison.accuracy_delta if comparison else 0
    sum_delta = comparison.summary_score_delta if comparison else 0
    lat_delta = (
        current.avg_latency_ms - (current.avg_latency_ms - 0)  # Need baseline for real delta
    )

    return f"""
    <div class="card">
        <h2>📊 Scorecard</h2>
        <div class="scorecard-grid">
            <div class="metric-card">
                <div class="metric-value">{current.overall_accuracy:.1f}%</div>
                <div class="metric-label">Accuracy</div>
                {delta_html(acc_delta)}
            </div>
            <div class="metric-card">
                <div class="metric-value">{current.avg_summary_score:.2f}</div>
                <div class="metric-label">Avg Summary Score (1-5)</div>
                {delta_html(sum_delta, unit="")}
            </div>
            <div class="metric-card">
                <div class="metric-value">{current.avg_latency_ms:.0f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{current.total_tokens:,}</div>
                <div class="metric-label">Total Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{current.passed_cases}/{current.total_cases}</div>
                <div class="metric-label">Cases Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{current.error_cases}</div>
                <div class="metric-label">Errors</div>
            </div>
        </div>
    </div>"""


def _render_category_breakdown(run: EvalRunSummary) -> str:
    rows = ""
    for cm in run.per_category:
        bar_color = "#22c55e" if cm.accuracy >= 80 else "#eab308" if cm.accuracy >= 60 else "#ef4444"
        rows += f"""
        <tr>
            <td><strong>{cm.category.value}</strong></td>
            <td>{cm.correct}/{cm.total}</td>
            <td>
                {cm.accuracy:.1f}%
                <div class="category-bar">
                    <div class="category-bar-fill" style="width: {cm.accuracy}%; background: {bar_color};"></div>
                </div>
            </td>
            <td>{cm.avg_summary_score:.2f}</td>
            <td>{cm.avg_latency_ms:.0f}ms</td>
        </tr>"""

    return f"""
    <div class="card">
        <h2>📂 Per-Category Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Correct/Total</th>
                    <th>Accuracy</th>
                    <th>Avg Summary Score</th>
                    <th>Avg Latency</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""


def _render_regressions(comparison: ComparisonResult) -> str:
    if not comparison.regressions:
        return """
    <div class="card">
        <h2>⬇️ Regressions</h2>
        <p style="color: #22c55e;">No regressions detected.</p>
    </div>"""

    rows = ""
    for reg in comparison.regressions:
        old_cat = reg.old_predicted_category.value if reg.old_predicted_category else "—"
        new_cat = reg.new_predicted_category.value if reg.new_predicted_category else "—"
        rows += f"""
        <tr>
            <td><strong>{reg.test_case_id}</strong></td>
            <td>{reg.expected_category.value}</td>
            <td class="diff-old">{old_cat}</td>
            <td class="diff-new">{new_cat}</td>
            <td>
                <div class="email-preview">{_escape_html(reg.input_email)}</div>
            </td>
            <td>
                <div class="diff-old" style="padding: 4px; border-radius: 4px; margin-bottom: 4px; font-size: 0.8rem;">
                    OLD: {_escape_html(reg.old_predicted_summary)}
                </div>
                <div class="diff-new" style="padding: 4px; border-radius: 4px; font-size: 0.8rem;">
                    NEW: {_escape_html(reg.new_predicted_summary)}
                </div>
            </td>
        </tr>"""

    return f"""
    <div class="card">
        <h2>⬇️ Regressions ({len(comparison.regressions)})</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Case</th>
                    <th>Expected</th>
                    <th>Old Prediction</th>
                    <th>New Prediction</th>
                    <th>Email</th>
                    <th>Summary Comparison</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""


def _render_improvements(comparison: ComparisonResult) -> str:
    if not comparison.improvements:
        return ""

    rows = ""
    for imp in comparison.improvements:
        old_cat = imp.old_predicted_category.value if imp.old_predicted_category else "—"
        new_cat = imp.new_predicted_category.value if imp.new_predicted_category else "—"
        rows += f"""
        <tr>
            <td><strong>{imp.test_case_id}</strong></td>
            <td>{imp.expected_category.value}</td>
            <td class="diff-old">{old_cat}</td>
            <td class="diff-new">{new_cat}</td>
        </tr>"""

    return f"""
    <div class="card">
        <h2>⬆️ Improvements ({len(comparison.improvements)})</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Case</th>
                    <th>Expected</th>
                    <th>Old Prediction</th>
                    <th>New Prediction</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""


def _render_drift_alerts(alerts: list[DriftAlert]) -> str:
    if not alerts or not any(a.triggered for a in alerts):
        return ""

    alert_boxes = ""
    for alert in alerts:
        if alert.triggered:
            alert_boxes += f"""
            <div class="alert-box">
                <strong>⚠️ {alert.metric}</strong>
                <p>{_escape_html(alert.message)}</p>
                <p style="color: #94a3b8; font-size: 0.8rem;">
                    Moving avg: {alert.current_moving_avg} | Threshold: {alert.threshold} | Window: {alert.window_size} runs
                </p>
            </div>"""

    return f"""
    <div class="card">
        <h2>📉 Drift Alerts</h2>
        {alert_boxes}
    </div>"""


def _render_trend_chart(run_history: list[EvalRunSummary]) -> str:
    if not run_history or len(run_history) < 2:
        return ""

    # Build SVG inline chart
    width = 800
    height = 200
    padding = 40
    chart_w = width - 2 * padding
    chart_h = height - 2 * padding

    n = len(run_history)
    accuracies = [r.overall_accuracy for r in run_history]
    min_acc = max(0, min(accuracies) - 5)
    max_acc = min(100, max(accuracies) + 5)
    acc_range = max_acc - min_acc if max_acc > min_acc else 1

    # Generate points
    points = []
    labels = []
    for i, run in enumerate(run_history):
        x = padding + (i / max(n - 1, 1)) * chart_w
        y = padding + chart_h - ((run.overall_accuracy - min_acc) / acc_range) * chart_h
        points.append(f"{x:.0f},{y:.0f}")
        labels.append((x, run.prompt_version, run.overall_accuracy))

    polyline = " ".join(points)

    # Build SVG
    dots = ""
    for i, (x, y_str) in enumerate(zip(points, labels)):
        px, py = x.split(",") if isinstance(x, str) else (str(x), "0")
        dots += f'<circle cx="{labels[i][0]:.0f}" cy="{padding + chart_h - ((labels[i][2] - min_acc) / acc_range) * chart_h:.0f}" r="4" fill="#3b82f6"/>'

    svg = f"""
    <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto;">
        <!-- Grid lines -->
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{padding + chart_h}" stroke="#334155" stroke-width="1"/>
        <line x1="{padding}" y1="{padding + chart_h}" x2="{padding + chart_w}" y2="{padding + chart_h}" stroke="#334155" stroke-width="1"/>

        <!-- Y axis labels -->
        <text x="{padding - 5}" y="{padding + 4}" text-anchor="end" fill="#94a3b8" font-size="11">{max_acc:.0f}%</text>
        <text x="{padding - 5}" y="{padding + chart_h + 4}" text-anchor="end" fill="#94a3b8" font-size="11">{min_acc:.0f}%</text>

        <!-- Accuracy line -->
        <polyline points="{polyline}" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round"/>

        <!-- Data points -->
        {dots}
    </svg>"""

    return f"""
    <div class="card">
        <h2>📈 Accuracy Trend (Last {n} Runs)</h2>
        {svg}
    </div>"""


def _render_all_results(run: EvalRunSummary) -> str:
    rows = ""
    for r in run.results:
        status_cls = "pass" if r.category_match else "fail"
        status_icon = "✓" if r.category_match else "✗"
        pred_cat = r.predicted_category.value if r.predicted_category else "—"
        error_text = f'<span class="fail">{_escape_html(r.error or "")}</span>' if r.error else ""

        rows += f"""
        <tr>
            <td>{r.test_case_id}</td>
            <td><span class="{status_cls}">{status_icon}</span></td>
            <td>{r.expected_category.value}</td>
            <td class="{status_cls}">{pred_cat}</td>
            <td>{r.summary_relevance_score:.1f}</td>
            <td>{r.latency_ms:.0f}ms</td>
            <td>{r.difficulty.value}</td>
            <td>{error_text}</td>
        </tr>"""

    return f"""
    <div class="card">
        <h2 class="collapsible">📋 All Results ({run.total_cases} cases)</h2>
        <div class="collapsible-content">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Status</th>
                        <th>Expected</th>
                        <th>Predicted</th>
                        <th>Summary Score</th>
                        <th>Latency</th>
                        <th>Difficulty</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>"""


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
