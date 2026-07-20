"""
Main CLI entrypoint for the Model Regression Detection System.

Orchestrates the full evaluation pipeline:
1. Load prompt config
2. Load golden dataset
3. Run evaluation
4. Score results
5. Compare against baseline
6. Check for drift
7. Generate HTML report
8. Send Slack alert (if enabled)
9. Persist results to SQLite
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.alerting import send_slack_alert
from src.comparator import compare_runs, format_comparison_summary
from src.config import ThresholdConfig
from src.db import get_baseline, get_run_history, save_run, set_baseline
from src.drift_detector import detect_drift, format_drift_alerts
from src.eval_runner import load_dataset, run_evaluation_sync
from src.prompt_loader import load_prompt, list_versions
from src.report_generator import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Regression Detection System — Evaluate LLM prompts against golden datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run eval with latest prompt (no Slack)
  python scripts/run_eval.py

  # Run eval with specific prompt version
  python scripts/run_eval.py --prompt-version v1

  # Run with Slack alerts and save as baseline
  python scripts/run_eval.py --alert --set-baseline

  # Use LLM-as-judge for higher accuracy summary scoring (2x API cost)
  python scripts/run_eval.py --llm-judge

  # List all available prompt versions
  python scripts/run_eval.py --list-versions
        """,
    )

    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Prompt version to evaluate (e.g., 'v1'). Default: latest.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to golden dataset JSON. Default: golden_dataset/dataset_v1.json",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Output directory for HTML reports. Default: reports/",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database. Default: data/eval_results.db",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        default=False,
        help="Send Slack alert with results.",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        default=False,
        help="Mark this run as the new baseline.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-judge for summary scoring (more accurate but 2x API cost). Default: heuristic scoring.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=15,
        help="API rate limit in requests per minute. Default: 15 (Gemini free tier).",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=3.0,
        help="Accuracy drop %% to trigger WARNING. Default: 3.0",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=8.0,
        help="Accuracy drop %% to trigger CRITICAL. Default: 8.0",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        default=False,
        help="List available prompt versions and exit.",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point. Returns 0 for PASS, 1 for WARNING, 2 for CRITICAL."""
    load_dotenv()
    args = parse_args()

    # List versions mode
    if args.list_versions:
        versions = list_versions()
        print(f"\nAvailable prompt versions: {versions}")
        return 0

    # Validate API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Get a free key at https://aistudio.google.com/")
        return 2

    # Build threshold config
    thresholds = ThresholdConfig(
        warning_delta_pct=args.warning_threshold,
        critical_delta_pct=args.critical_threshold,
    )

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = args.db_path or os.path.join(project_root, "data", "eval_results.db")
    report_dir = args.report_dir or os.path.join(project_root, "reports")

    try:
        # Step 1: Load prompt
        print("\n📋 Loading prompt configuration...")
        prompt_config = load_prompt(version=args.prompt_version)
        print(f"   Loaded: {prompt_config.prompt_fingerprint()}")

        # Step 2: Load dataset
        print("\n📦 Loading golden dataset...")
        dataset_path = args.dataset or os.path.join(
            project_root, "golden_dataset", "dataset_v1.json"
        )
        dataset = load_dataset(dataset_path)
        print(f"   Loaded: {dataset.version} ({dataset.size} cases)")

        # Step 3: Run evaluation
        print("\n🔬 Running evaluation...")
        summary = run_evaluation_sync(
            prompt_config=prompt_config,
            dataset=dataset,
            use_llm_judge=args.llm_judge,
            rate_limit_rpm=args.rate_limit,
            verbose=not args.quiet,
        )

        # Step 4: Save to database
        print("\n💾 Saving results to database...")
        save_run(summary, db_path=db_path, is_baseline=args.set_baseline)
        if args.set_baseline:
            set_baseline(summary.run_id, db_path=db_path)
            print(f"   Marked run {summary.run_id} as baseline.")
        print(f"   Saved run {summary.run_id}")

        # Step 5: Compare against baseline
        comparison = None
        baseline = get_baseline(db_path=db_path)
        if baseline and baseline.run_id != summary.run_id:
            print("\n🔍 Comparing against baseline...")
            comparison = compare_runs(summary, baseline, thresholds)
            print(format_comparison_summary(comparison))
        else:
            print("\n   No baseline found for comparison (this may be the first run).")

        # Step 6: Check for drift
        drift_alerts = []
        history = get_run_history(n=thresholds.drift_window, db_path=db_path)
        if len(history) >= thresholds.drift_window:
            print("\n📉 Checking for drift...")
            drift_alerts = detect_drift(history, thresholds)
            print(format_drift_alerts(drift_alerts))

        # Step 7: Generate report
        print("\n📄 Generating HTML report...")
        report_path = generate_report(
            current_run=summary,
            comparison=comparison,
            drift_alerts=drift_alerts,
            run_history=history if len(history) >= 2 else None,
            output_dir=report_dir,
        )
        print(f"   Report: {report_path}")

        # Step 8: Send Slack alert (if enabled)
        if args.alert:
            print("\n📬 Sending Slack alert...")
            sent = send_slack_alert(
                current_run=summary,
                comparison=comparison,
                drift_alerts=drift_alerts,
                report_path=report_path,
            )
            if sent:
                print("   Slack alert sent successfully.")
            else:
                print("   Alert logged to console (Slack not configured or failed).")

        # Return exit code based on comparison status
        if comparison:
            from src.config import EvalStatus
            if comparison.status == EvalStatus.CRITICAL:
                print("\n🔴 EXIT: CRITICAL — regressions detected!")
                return 2
            elif comparison.status == EvalStatus.WARNING:
                print("\n🟡 EXIT: WARNING — potential regressions detected.")
                return 1

        print("\n🟢 EXIT: PASS — evaluation completed successfully.")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
