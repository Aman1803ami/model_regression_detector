"""
Seed the baseline evaluation run.

Runs the evaluation with the current prompt (default: v1) and marks
the result as the baseline for future comparisons. This should be run
once when setting up the system for the first time.
"""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.db import save_run, set_baseline
from src.eval_runner import load_dataset, run_evaluation_sync
from src.prompt_loader import load_prompt
from src.report_generator import generate_report


def main() -> None:
    load_dotenv()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Get a free key at https://aistudio.google.com/")
        sys.exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "data", "eval_results.db")

    print("🌱 Seeding baseline evaluation run...\n")

    # Load v1 prompt (the stable baseline)
    prompt = load_prompt(version="v1")
    print(f"   Prompt: {prompt.prompt_fingerprint()}")

    # Load dataset
    dataset = load_dataset()
    print(f"   Dataset: {dataset.version} ({dataset.size} cases)")

    # Run evaluation
    print("\n   Running full evaluation (this may take a few minutes)...\n")
    summary = run_evaluation_sync(
        prompt_config=prompt,
        dataset=dataset,
        use_llm_judge=False,  # Heuristic default; pass True for LLM-as-judge
        rate_limit_rpm=15,
    )

    # Save as baseline
    run_id = save_run(summary, db_path=db_path, is_baseline=True)
    set_baseline(run_id, db_path=db_path)

    # Generate report
    report_path = generate_report(
        current_run=summary,
        output_dir=os.path.join(project_root, "reports"),
    )

    print(f"\n✅ Baseline seeded!")
    print(f"   Run ID:   {run_id}")
    print(f"   Accuracy: {summary.overall_accuracy:.1f}%")
    print(f"   Summary:  {summary.avg_summary_score:.2f}/5")
    print(f"   Report:   {report_path}")


if __name__ == "__main__":
    main()
