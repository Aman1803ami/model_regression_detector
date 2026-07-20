"""
Core configuration and data models for the Model Regression Detection System.

All data contracts are defined here using Pydantic v2 for type safety
and validation. These models form the interface between every component
in the pipeline: prompt loading, evaluation, scoring, and reporting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    """Valid classification categories for customer support emails."""
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Difficulty(str, Enum):
    """Expected difficulty for a test case."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalStatus(str, Enum):
    """Overall evaluation result status."""
    PASS = "pass"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Prompt Configuration
# ---------------------------------------------------------------------------

class FewShotExample(BaseModel):
    """A single few-shot example included in the prompt."""
    input: str
    output_category: EmailCategory
    output_summary: str


class PromptConfig(BaseModel):
    """
    Versioned prompt configuration loaded from YAML files.
    This is the 'code' that the CI pipeline tests against.
    """
    version: str
    timestamp: datetime
    model: str = "gemini-3.1-flash-lite"
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 256

    def prompt_fingerprint(self) -> str:
        """Return a short identifier for logging."""
        return f"{self.version}@{self.model}"


# ---------------------------------------------------------------------------
# Golden Dataset Models
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    """A single test case from the golden dataset."""
    id: str
    input_email: str
    expected_category: EmailCategory
    expected_summary: str
    difficulty: Difficulty = Difficulty.MEDIUM
    notes: str = ""


class GoldenDataset(BaseModel):
    """The full versioned golden dataset."""
    version: str
    created_at: datetime
    description: str = ""
    test_cases: list[TestCase]

    @property
    def size(self) -> int:
        return len(self.test_cases)

    def cases_by_category(self) -> dict[EmailCategory, list[TestCase]]:
        """Group test cases by expected category."""
        groups: dict[EmailCategory, list[TestCase]] = {}
        for case in self.test_cases:
            groups.setdefault(case.expected_category, []).append(case)
        return groups


# ---------------------------------------------------------------------------
# Classification Result
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Raw output from the LLM classifier for a single email."""
    category: EmailCategory
    summary: str
    raw_response: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------------------------

class TestCaseResult(BaseModel):
    """Evaluation result for a single test case."""
    test_case_id: str
    input_email: str
    expected_category: EmailCategory
    expected_summary: str
    predicted_category: Optional[EmailCategory] = None
    predicted_summary: str = ""
    category_match: bool = False
    summary_relevance_score: float = 0.0  # 1-5 scale from LLM-as-judge
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    difficulty: Difficulty = Difficulty.MEDIUM


class CategoryMetrics(BaseModel):
    """Accuracy metrics for a single category."""
    category: EmailCategory
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_summary_score: float = 0.0
    avg_latency_ms: float = 0.0


class EvalRunSummary(BaseModel):
    """Aggregate summary of a complete evaluation run."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt_version: str
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    overall_accuracy: float = 0.0
    avg_summary_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    per_category: list[CategoryMetrics] = Field(default_factory=list)
    results: list[TestCaseResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison / Regression Models
# ---------------------------------------------------------------------------

class RegressionCase(BaseModel):
    """A test case that flipped from pass to fail between runs."""
    test_case_id: str
    input_email: str
    expected_category: EmailCategory
    expected_summary: str
    old_predicted_category: Optional[EmailCategory] = None
    old_predicted_summary: str = ""
    new_predicted_category: Optional[EmailCategory] = None
    new_predicted_summary: str = ""
    old_summary_score: float = 0.0
    new_summary_score: float = 0.0


class ImprovementCase(BaseModel):
    """A test case that flipped from fail to pass between runs."""
    test_case_id: str
    input_email: str
    expected_category: EmailCategory
    old_predicted_category: Optional[EmailCategory] = None
    new_predicted_category: Optional[EmailCategory] = None


class ComparisonResult(BaseModel):
    """Result of comparing two evaluation runs."""
    status: EvalStatus = EvalStatus.PASS
    baseline_run_id: str
    current_run_id: str
    baseline_accuracy: float = 0.0
    current_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    baseline_summary_score: float = 0.0
    current_summary_score: float = 0.0
    summary_score_delta: float = 0.0
    regressions: list[RegressionCase] = Field(default_factory=list)
    improvements: list[ImprovementCase] = Field(default_factory=list)
    per_category_deltas: dict[str, float] = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------

class DriftAlert(BaseModel):
    """Alert generated when slow drift is detected."""
    triggered: bool = False
    metric: str = ""
    current_moving_avg: float = 0.0
    threshold: float = 0.0
    window_size: int = 7
    message: str = ""


# ---------------------------------------------------------------------------
# Threshold Configuration
# ---------------------------------------------------------------------------

class ThresholdConfig(BaseModel):
    """Configurable thresholds for regression and drift detection."""
    warning_delta_pct: float = 3.0     # Flag as WARNING if accuracy drops > 3%
    critical_delta_pct: float = 8.0    # Flag as CRITICAL if accuracy drops > 8%
    min_flips_for_signal: int = 2      # Suppress noise if fewer flips than this
    drift_window: int = 7              # Number of runs for rolling average
    drift_accuracy_floor: float = 85.0 # Alert if moving avg drops below this
    drift_summary_floor: float = 3.0   # Alert if avg summary score drops below this
    latency_multiplier: float = 2.0    # Flag if latency > 2x median
