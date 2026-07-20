# Model Regression Detection System

A CI/CD pipeline that continuously evaluates LLM-powered features against a golden dataset, detects quality regressions when prompts or models change, and alerts your team via Slack before degraded outputs reach production.

## Why This Exists

Every AI team ships prompt changes blind. There's no equivalent of unit tests for LLM behavior — you change a prompt, deploy it, and hope nothing breaks. This system brings the rigor of traditional CI/CD to LLM operations:

- **Every prompt change gets tested** against 80 hand-curated ground-truth cases before it can merge
- **Multi-dimensional scoring** catches both hard failures (wrong category) and soft degradation (worse summaries)
- **Drift detection** catches slow quality erosion that no single run would flag
- **Blocking CI gates** prevent regressions from reaching production

## 📊 Live Execution & Test Results

Here is an actual pipeline execution comparing a degraded prompt (`v2.yaml`) against the baseline (`v1.yaml`):

```text
============================================================
  🔴 Comparison: CRITICAL
============================================================
  Baseline run:  a6bf5a1aab4f (Prompt: v1)
  Current run:   5f3b7d3d1033 (Prompt: v2)
  Accuracy:      92.5% → 82.5% (-10.0%)
  Summary score: 3.41 → 3.00 (-0.41)
  Regressions:   10 test cases regressed

  Per-category accuracy deltas:
    account     : -14.3% ↓
    billing     : -5.0% ↓
    general     : +0.0% →
    technical   : -18.2% ↓

  Regressed test cases:
    TC-017: billing   → general   (expected: billing)
    TC-027: technical → general   (expected: technical)
    TC-028: technical → general   (expected: technical)
    TC-038: technical → general   (expected: technical)
    TC-039: technical → general   (expected: technical)
    TC-049: account   → general   (expected: account)
    TC-057: account   → technical (expected: account)
    TC-058: account   → technical (expected: account)
    TC-059: account   → general   (expected: account)
    TC-060: account   → general   (expected: account)

  Message: Accuracy dropped 10.0% (>8.0% critical threshold); 10 test case(s) regressed

📄 Report: assets/sample_report.html
📬 Slack Alert: Delivered via Webhook
🔴 EXIT: 2 (CRITICAL — CI build blocked)
```

### Slack Alert Example
When a critical regression is detected, the pipeline formats a Block Kit payload and posts to Slack:

```text
🔴 Model Evaluation: CRITICAL
Prompt: v2 → Model: gemini-3.1-flash-lite
Accuracy: 82.5% (-10.0%) | Summary Score: 3.00/5 | Latency: 813ms
Cases: 66/80 passed | Errors: 0

🔻 10 regression(s): TC-017, TC-027, TC-028, TC-038, TC-039

Per-Category Accuracy:
`billing   ` █████████░ 85% (-5%)
`technical ` ████████░░ 75% (-18%)
`account   ` ███████░░░ 70% (-14%)
`general   ` █████████░ 95% (+0%)

📄 View Full Report | Run ID: 5f3b7d3d1033
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Actions CI                     │
│                                                         │
│  PR modifies /prompts/* ──→ Triggers eval pipeline      │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Evaluation Pipeline (run_eval.py)           │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Prompt   │  │  Golden  │  │ Gemini   │              │
│  │  Loader   │──│ Dataset  │──│ API      │              │
│  │ (YAML)    │  │ (80 cases)│  │(classify)│              │
│  └──────────┘  └──────────┘  └────┬─────┘              │
│                                    │                     │
│  ┌──────────┐  ┌──────────┐  ┌────▼─────┐              │
│  │  Scoring  │──│Comparator│──│   Drift  │              │
│  │ (multi-   │  │(baseline │  │ Detector │              │
│  │  dim)     │  │  diff)   │  │(rolling) │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│        │              │              │                   │
│        ▼              ▼              ▼                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  SQLite   │  │   HTML   │  │  Slack   │              │
│  │   DB      │  │  Report  │  │  Alert   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

## The LLM Feature Under Test

A customer support email classifier that:
- **Input**: Raw customer email text
- **Output**: `{ "category": "billing|technical|account|general", "summary": "one-sentence summary" }`
- **Prompt**: Versioned YAML in `/prompts/` with system prompt and few-shot examples

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/model-regression-detector.git
cd model-regression-detector
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key at https://aistudio.google.com/
```

### 3. Seed the baseline

```bash
python scripts/seed_baseline.py
```

This runs the v1 prompt against all 80 test cases and marks it as the comparison baseline. Takes ~10 minutes on free tier.

### 4. Run an evaluation

```bash
# Evaluate latest prompt against baseline
python scripts/run_eval.py

# Evaluate a specific prompt version
python scripts/run_eval.py --prompt-version v2

# With Slack alerts
python scripts/run_eval.py --alert

# Enable LLM-as-judge for higher-accuracy summary scoring (2x API cost)
python scripts/run_eval.py --llm-judge
```

### 5. Run with Docker

```bash
docker build -t model-regression-detector .
docker run --env-file .env -v $(pwd)/reports:/app/reports model-regression-detector
```

## How to Add New Test Cases

1. Open `golden_dataset/dataset_v1.json`
2. Add a new object to the `test_cases` array:

```json
{
  "id": "TC-081",
  "input_email": "Your actual customer email text here...",
  "expected_category": "billing",
  "expected_summary": "One-sentence summary of what this email is about.",
  "difficulty": "medium",
  "notes": "Why this test case matters — what edge case does it cover?"
}
```

3. Run `python scripts/run_eval.py` to verify the new case works
4. Commit the dataset change — it will be evaluated in CI on your next PR

**Important**: Write test cases by hand. Do NOT generate them with an LLM. The entire point is human-verified ground truth.

## How to Create a New Prompt Version

1. Create a new YAML file in `/prompts/` (e.g., `v3.yaml`):

```yaml
version: "v3"
timestamp: "2026-07-25T00:00:00Z"
model: "gemini-3.5-flash"
temperature: 0.0
system_prompt: |
  Your updated prompt here...
few_shot_examples:
  - input: "Example email..."
    output:
      category: "billing"
      summary: "Example summary..."
```

2. Run the eval locally first: `python scripts/run_eval.py --prompt-version v3`
3. Check the HTML report in `/reports/`
4. If it passes, commit and open a PR — CI will run the full eval

## How to Adjust Thresholds

Thresholds are configurable via CLI args or `ThresholdConfig` in `src/config.py`:

| Threshold | Default | What it does |
|-----------|---------|-------------|
| `warning_delta_pct` | 3.0% | Accuracy drop to trigger WARNING |
| `critical_delta_pct` | 8.0% | Accuracy drop to trigger CRITICAL (blocks merge) |
| `min_flips_for_signal` | 2 | Minimum regressions before alerting (noise filter) |
| `drift_window` | 7 | Number of runs for rolling average |
| `drift_accuracy_floor` | 85.0% | Moving average below this triggers drift alert |
| `drift_summary_floor` | 3.0/5 | Summary score below this triggers drift alert |

```bash
python scripts/run_eval.py --warning-threshold 5.0 --critical-threshold 10.0
```

## Scoring Dimensions

| Dimension | Method | Purpose |
|-----------|--------|---------|
| Category Match | Exact string match | Did the LLM pick the right category? |
| Summary Relevance | Keyword overlap heuristic (default) or LLM-as-judge via `--llm-judge` (1-5 scale) | Is the summary accurate and complete? |
| Latency | Wall-clock timing per request | Detect performance regressions |
| Token Usage | Extracted from API response metadata | Track cost changes |

## Design Decisions & Rationale

### Why hand-curated test cases, not synthetic?
LLM-generated test cases have the same distribution biases as LLM outputs. You can't evaluate a model using data that looks like model output. Human-curated cases capture real-world patterns (sarcasm, typos, ambiguity) that models struggle with.

### Why slow drift detection separate from per-run diffs?
A model provider can silently change behavior over time. Each individual run might pass within thresholds, but over 7 runs you've dropped from 95% to 82%. The rolling average catches this before it compounds.

### Why SQLite instead of PostgreSQL?
Zero infrastructure. The database is a single file that can be committed to git (or not). No Docker containers to manage for storage. When you need scale, the interface is identical — just swap the connection string.

### Why not use an existing eval framework?
Building the eval engine from scratch demonstrates understanding of evaluation mechanics, not just API usage. The custom implementation also gives complete control over scoring dimensions and threshold logic.

### Why heuristic scoring as default instead of LLM-as-judge?
LLM-as-judge doubles API costs and latency. For CI on every PR, the heuristic (keyword overlap) is fast, free, and deterministic. LLM-as-judge is available via `--llm-judge` for deeper evaluations when you need higher accuracy on summary quality.

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_comparator.py -v
```

## Project Structure

```
├── README.md                     # You are here
├── Dockerfile                    # Containerized eval runner
├── docker-compose.yml            # Local dev setup
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .github/workflows/
│   └── eval-pipeline.yml         # CI/CD workflow
├── prompts/
│   ├── v1.yaml                   # Baseline prompt (stable)
│   └── v2.yaml                   # Test prompt (intentionally weaker)
├── golden_dataset/
│   ├── dataset_v1.json           # 80 hand-curated test cases
│   └── schema.json               # JSON schema for validation
├── src/
│   ├── config.py                 # Pydantic data models
│   ├── classifier.py             # LLM email classifier
│   ├── prompt_loader.py          # YAML prompt versioning
│   ├── eval_runner.py            # Test runner (sync + async)
│   ├── scoring.py                # Multi-dimensional scoring
│   ├── comparator.py             # Regression diff engine
│   ├── drift_detector.py         # Rolling average drift detection
│   ├── alerting.py               # Slack webhook integration
│   ├── report_generator.py       # HTML diff report builder
│   └── db.py                     # SQLite persistence layer
├── scripts/
│   ├── run_eval.py               # CLI entrypoint
│   └── seed_baseline.py          # Baseline seeder
├── tests/                        # Unit tests (pytest)
├── reports/                      # Generated HTML reports
└── data/                         # SQLite database
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.11+ | Industry standard for ML tooling |
| LLM | Google Gemini (free tier) | Fast, free, widely recognized |
| Data Models | Pydantic v2 | Type-safe interface contracts |
| Storage | SQLite | Zero infra, portable, git-friendly |
| Alerting | Slack Webhooks | What real teams actually use |
| CI/CD | GitHub Actions | Runs on every PR, free tier |
| Reports | Self-contained HTML | No server needed, works offline |
| Container | Docker | Production readiness signal |

## License

MIT
