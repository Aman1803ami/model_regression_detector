# Model Regression Detection System — Implementation Plan

A CI/CD-style pipeline that continuously tests an LLM-powered customer support email classifier against a golden dataset, detects quality regressions, and alerts via Slack before bad outputs reach users.

## Project Structure

```
resume project one/
├── README.md                          # Production-style onboarding docs
├── Dockerfile                         # Containerized eval runner
├── docker-compose.yml                 # Local dev environment
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
├── .env.example                       # Template for env vars
├── .github/
│   └── workflows/
│       └── eval-pipeline.yml          # GitHub Actions CI workflow
├── prompts/
│   ├── v1.yaml                        # Baseline prompt
│   └── v2.yaml                        # Modified prompt (for testing)
├── golden_dataset/
│   ├── dataset_v1.json                # 80+ hand-curated test cases
│   └── schema.json                    # JSON schema for validation
├── src/
│   ├── __init__.py
│   ├── config.py                      # Pydantic config models
│   ├── classifier.py                  # LLM email classifier feature
│   ├── prompt_loader.py               # YAML prompt versioning
│   ├── eval_runner.py                 # Test runner (async batched)
│   ├── scoring.py                     # Multi-dimensional scoring
│   ├── comparator.py                  # Diff/regression detection
│   ├── drift_detector.py             # Rolling average drift detection
│   ├── alerting.py                    # Slack webhook integration
│   ├── report_generator.py           # HTML diff report builder
│   └── db.py                          # SQLite persistence layer
├── reports/                           # Generated HTML reports
├── data/
│   └── eval_results.db                # SQLite database for runs
├── tests/
│   ├── test_classifier.py
│   ├── test_scoring.py
│   ├── test_comparator.py
│   └── test_drift_detector.py
└── scripts/
    ├── run_eval.py                    # CLI entrypoint
    └── seed_baseline.py               # Seed first baseline run
```

---

## Tech Stack

| Component | Tool | Rationale |
|---|---|---|
| Language | Python 3.11+ | Industry standard for ML tooling |
| LLM Provider | Google Gemini (`gemini-3.5-flash`) via `google-genai` SDK | Free tier, fast, widely recognized |
| Data Models | Pydantic v2 | Typed interface contracts |
| Data Storage | SQLite + JSON | Zero infra, portable, git-friendly |
| Alerting | Slack Webhooks (via `requests`) | What real teams use |
| CI/CD | GitHub Actions | Free tier, runs on every PR |
| Reporting | Jinja2 HTML templates | Rich diff reports, no server needed |
| Containerization | Docker | Production readiness signal |
| Testing | pytest + pytest-asyncio | Standard Python testing |
| Async | asyncio + aiohttp | Batch LLM calls efficiently |

---

## Proposed Changes

### Phase 1 — Core Data Models & Prompt Versioning

#### [NEW] [config.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/config.py)
- `PromptConfig` dataclass (Pydantic): `version_id`, `timestamp`, `system_prompt`, `few_shot_examples`, `model_name`
- `TestCase` model: `id`, `input_email`, `expected_category`, `expected_summary`, `difficulty`, `notes`
- `EvalResult` model: `test_case_id`, `predicted_category`, `predicted_summary`, `category_match` (bool), `summary_relevance_score` (1-5), `latency_ms`, `tokens_used`
- `EvalRunSummary` model: `run_id`, `prompt_version`, `model`, `timestamp`, `overall_pass_rate`, `per_category_accuracy`, `avg_latency`, `total_tokens`
- `ThresholdConfig`: `warning_delta_pct` (default 3%), `critical_delta_pct` (default 8%), `drift_window` (default 7), `drift_threshold`

#### [NEW] [prompt_loader.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/prompt_loader.py)
- Load YAML prompt files from `/prompts` directory
- Parse into `PromptConfig` objects
- Support listing all versions, loading specific version, loading latest

#### [NEW] [v1.yaml](file:///c:/Users/hp/Desktop/resume%20project%20one/prompts/v1.yaml)
```yaml
version: "v1"
timestamp: "2026-07-20T00:00:00Z"
model: "gemini-3.5-flash"
system_prompt: |
  You are a customer support email classifier...
few_shot_examples:
  - input: "I was charged twice for my subscription..."
    output: { category: "billing", summary: "..." }
```

---

### Phase 2 — Golden Dataset (80+ Hand-Curated Cases)

#### [NEW] [dataset_v1.json](file:///c:/Users/hp/Desktop/resume%20project%20one/golden_dataset/dataset_v1.json)
- **80 test cases** across 4 categories: `billing` (20), `technical` (20), `account` (20), `general` (20)
- Each case: `id`, `input_email`, `expected_category`, `expected_summary`, `difficulty` (easy/medium/hard), `notes`
- **Edge cases included**: ambiguous category emails, extremely short emails, typo-heavy emails, sarcastic tone, mixed language fragments, multi-issue emails
- All written by hand — NOT LLM-generated

#### [NEW] [schema.json](file:///c:/Users/hp/Desktop/resume%20project%20one/golden_dataset/schema.json)
- JSON Schema for validating dataset format

---

### Phase 3 — LLM Classifier Feature

#### [NEW] [classifier.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/classifier.py)
- `classify_email(email_text: str, prompt_config: PromptConfig) -> ClassificationResult`
- Uses `google-genai` SDK (`gemini-3.5-flash`)
- Returns structured JSON: `{ "category": str, "summary": str }`
- Handles retries, timeouts, and malformed LLM responses
- Tracks latency and token usage per call

---

### Phase 4 — Evaluation Engine

#### [NEW] [eval_runner.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/eval_runner.py)
- `run_evaluation(prompt_config: PromptConfig, dataset: list[TestCase]) -> EvalRunSummary`
- Async batched execution (configurable concurrency, default 5)
- Rate limiting to stay within Gemini free tier (15 RPM)
- Collects raw outputs + all scoring dimensions per test case
- Persists results to SQLite via `db.py`

#### [NEW] [scoring.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/scoring.py)
- **Category match**: Binary exact match
- **Summary relevance**: LLM-as-judge scoring (1-5 scale) — uses a separate Gemini call with a judge prompt to rate summary quality against the expected summary
- **Latency scoring**: Flag if > 2x median latency
- **Token efficiency**: Track tokens per request
- Aggregate scores into per-category and overall metrics

#### [NEW] [comparator.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/comparator.py)
- `compare_runs(current: EvalRunSummary, baseline: EvalRunSummary) -> ComparisonResult`
- Calculates: overall pass rate delta, per-category accuracy delta
- Identifies: regressions (pass → fail), improvements (fail → pass)
- Classifies result as `PASS` / `WARNING` / `CRITICAL` based on `ThresholdConfig`
- Statistical significance check: if total flips < 2 on dataset of 80, suppress noise

#### [NEW] [drift_detector.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/drift_detector.py)
- Tracks rolling average of scores over last N runs (configurable, default 7)
- Detects slow degradation even when no single run triggers an alert
- Fires "slow drift" warning when moving average drops below threshold
- Stores historical metrics in SQLite

---

### Phase 5 — Persistence Layer

#### [NEW] [db.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/db.py)
- SQLite database with tables:
  - `eval_runs`: run metadata (id, prompt_version, model, timestamp, overall_score)
  - `eval_results`: per-test-case results for each run
  - `score_history`: time series of aggregate scores for drift detection
- Functions: `save_run()`, `get_latest_run()`, `get_baseline()`, `get_run_history(n)`

---

### Phase 6 — Reporting & Alerting

#### [NEW] [report_generator.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/report_generator.py)
- Generates a rich HTML diff report using Jinja2 templates:
  - **Header**: Run metadata (prompt version, model, timestamp)
  - **Summary scorecard**: Side-by-side current vs baseline metrics
  - **Regression table**: Each regressed case with old output vs new output
  - **Improvement table**: Cases that improved
  - **Trend chart**: Inline SVG/Chart.js showing scores over last N runs
  - **Per-category breakdown**: Accuracy per category with bar charts
- Saves to `/reports/` directory with timestamped filename

#### [NEW] [alerting.py](file:///c:/Users/hp/Desktop/resume%20project%20one/src/alerting.py)
- `send_slack_alert(comparison: ComparisonResult, report_url: str)`
- Uses `requests` to POST to Slack incoming webhook
- Structured Block Kit message with:
  - 🟢 PASS / 🟡 WARNING / 🔴 CRITICAL status emoji
  - Headline numbers: "3 regressions detected, accuracy dropped from 94% to 89%"
  - Link to full HTML diff report
  - Per-category breakdown in a compact format
- Graceful handling when webhook URL is not configured (logs instead)

---

### Phase 7 — CLI Entrypoint & Scripts

#### [NEW] [run_eval.py](file:///c:/Users/hp/Desktop/resume%20project%20one/scripts/run_eval.py)
- CLI interface using `argparse`:
  - `--prompt-version`: Which prompt to eval (default: latest)
  - `--baseline`: Which run to compare against (default: latest stored)
  - `--dataset`: Path to golden dataset
  - `--alert`: Enable/disable Slack alerts
  - `--report-dir`: Output directory for reports
- Orchestrates: load prompt → load dataset → run eval → score → compare → report → alert

#### [NEW] [seed_baseline.py](file:///c:/Users/hp/Desktop/resume%20project%20one/scripts/seed_baseline.py)
- Seeds the first baseline eval run so subsequent runs have something to compare against

---

### Phase 8 — CI/CD Integration

#### [NEW] [eval-pipeline.yml](file:///c:/Users/hp/Desktop/resume%20project%20one/.github/workflows/eval-pipeline.yml)
```yaml
name: LLM Eval Pipeline
on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'src/**'
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scripts/run_eval.py --alert --report-dir reports/
      - uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: reports/
      - name: Comment on PR
        uses: actions/github-script@v7
        with:
          script: |
            // Read summary, post as PR comment
            // Block merge if CRITICAL
```

---

### Phase 9 — Docker & Production Readiness

#### [NEW] [Dockerfile](file:///c:/Users/hp/Desktop/resume%20project%20one/Dockerfile)
- Multi-stage build: slim Python base
- Installs dependencies, copies source
- Accepts env vars: `GEMINI_API_KEY`, `SLACK_WEBHOOK_URL`, `THRESHOLDS`
- Entrypoint: `python scripts/run_eval.py`

#### [NEW] [docker-compose.yml](file:///c:/Users/hp/Desktop/resume%20project%20one/docker-compose.yml)
- Service for eval runner with env var passthrough
- Volume mount for reports output

#### [NEW] [.env.example](file:///c:/Users/hp/Desktop/resume%20project%20one/.env.example)
- Template with all configurable environment variables

---

### Phase 10 — Testing

#### [NEW] [test_classifier.py](file:///c:/Users/hp/Desktop/resume%20project%20one/tests/test_classifier.py)
- Tests for JSON parsing, error handling, retry logic (with mocked LLM)

#### [NEW] [test_scoring.py](file:///c:/Users/hp/Desktop/resume%20project%20one/tests/test_scoring.py)
- Tests for each scoring dimension, edge cases in aggregation

#### [NEW] [test_comparator.py](file:///c:/Users/hp/Desktop/resume%20project%20one/tests/test_comparator.py)
- Tests for regression detection, threshold logic, noise suppression

#### [NEW] [test_drift_detector.py](file:///c:/Users/hp/Desktop/resume%20project%20one/tests/test_drift_detector.py)
- Tests for rolling average calculation, drift alerting

---

### Phase 11 — README & Documentation

#### [NEW] [README.md](file:///c:/Users/hp/Desktop/resume%20project%20one/README.md)
- Written as **internal team documentation**, not a tutorial
- Sections: What This Does (1 paragraph), Architecture Overview (with diagram), Setup & Configuration, How to Add Test Cases, How to Adjust Thresholds, Design Decisions & Rationale
- Includes Mermaid architecture diagram

---

## User Review Required

> [!IMPORTANT]
> **Gemini API Key**: You'll need a free Google AI Studio API key. Go to [aistudio.google.com](https://aistudio.google.com/) to create one. The project uses `gemini-3.5-flash` on the free tier.

> [!IMPORTANT]  
> **Slack Webhook**: To test Slack alerting, you'll need to create a Slack app with an incoming webhook. The system gracefully degrades (logs to console) when no webhook is configured, so this is optional for development.

## Open Questions

> [!IMPORTANT]
> **LLM-as-Judge for Summary Scoring**: The plan uses a second Gemini call to rate summary quality (1-5). This doubles API usage. Should I implement this, or use a simpler heuristic (e.g., keyword overlap / ROUGE-like scoring) as the default with LLM-as-judge as an optional flag?

> [!NOTE]
> **Dataset Size**: The guide suggests 50-100 cases. I'm planning **80 test cases** (20 per category) with deliberate edge cases. This balances thoroughness with free-tier API limits. Sound good?

> [!NOTE]
> **HTML Report Hosting**: The GitHub Actions workflow generates an HTML report and uploads it as a build artifact. For the Slack link, it would point to the GitHub Actions artifact URL. An alternative is generating a self-contained single-file HTML report that can be viewed locally. Which do you prefer?

---

## Verification Plan

### Automated Tests
```bash
# Unit tests (mocked LLM calls)
pytest tests/ -v

# Integration test (requires GEMINI_API_KEY)
python scripts/run_eval.py --prompt-version v1 --no-alert

# Docker build verification
docker build -t model-regression-detector .
docker run --env-file .env model-regression-detector
```

### Manual Verification
1. Run eval with `v1` prompt → seed baseline
2. Modify prompt to `v2` (intentionally degrade) → run eval → verify regressions detected
3. Check Slack alert fires with correct status and numbers
4. Open HTML diff report → verify side-by-side regression display
5. Run multiple evals → verify drift detection triggers on gradual degradation
6. Push to GitHub → verify Actions workflow triggers on `/prompts` change
