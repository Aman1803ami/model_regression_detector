"""
Tests for the email classifier module.

Tests JSON parsing, error handling, prompt building, and response
extraction — all with mocked LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.classifier import _build_prompt, _parse_llm_response, classify_email
from src.config import EmailCategory, FewShotExample, PromptConfig
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prompt_config() -> PromptConfig:
    return PromptConfig(
        version="test-v1",
        timestamp=datetime.now(timezone.utc),
        model="gemini-3.5-flash",
        system_prompt="You are a test classifier. Return JSON: {\"category\": \"<cat>\", \"summary\": \"<sum>\"}",
        few_shot_examples=[],
        temperature=0.0,
    )


@pytest.fixture
def prompt_config_with_examples() -> PromptConfig:
    return PromptConfig(
        version="test-v2",
        timestamp=datetime.now(timezone.utc),
        model="gemini-3.5-flash",
        system_prompt="Classify emails. Return JSON.",
        few_shot_examples=[
            FewShotExample(
                input="I was charged twice",
                output_category=EmailCategory.BILLING,
                output_summary="Customer reports double charge.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests for _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_clean_json(self):
        response = '{"category": "billing", "summary": "Customer needs refund"}'
        cat, summary = _parse_llm_response(response)
        assert cat == "billing"
        assert summary == "Customer needs refund"

    def test_json_in_markdown_fences(self):
        response = '```json\n{"category": "technical", "summary": "App crashes"}\n```'
        cat, summary = _parse_llm_response(response)
        assert cat == "technical"
        assert summary == "App crashes"

    def test_json_with_trailing_text(self):
        response = 'Here is the result: {"category": "account", "summary": "Password reset"} Hope this helps!'
        cat, summary = _parse_llm_response(response)
        assert cat == "account"
        assert summary == "Password reset"

    def test_invalid_json_returns_general(self):
        response = "I cannot classify this email properly."
        cat, summary = _parse_llm_response(response)
        assert cat == "general"

    def test_empty_response(self):
        cat, summary = _parse_llm_response("")
        assert cat == "general"

    def test_json_with_extra_whitespace(self):
        response = '  \n  {"category": "billing", "summary": "Refund request"}  \n  '
        cat, summary = _parse_llm_response(response)
        assert cat == "billing"

    def test_uppercase_category_normalized(self):
        response = '{"category": "BILLING", "summary": "test"}'
        cat, summary = _parse_llm_response(response)
        assert cat == "billing"

    def test_category_with_spaces(self):
        response = '{"category": "  technical  ", "summary": "test"}'
        cat, summary = _parse_llm_response(response)
        assert cat == "technical"


# ---------------------------------------------------------------------------
# Tests for _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_basic_prompt(self, prompt_config):
        prompt = _build_prompt("Test email content", prompt_config)
        assert "Test email content" in prompt
        assert "test classifier" in prompt

    def test_prompt_with_few_shot(self, prompt_config_with_examples):
        prompt = _build_prompt("Test email", prompt_config_with_examples)
        assert "Example 1:" in prompt
        assert "I was charged twice" in prompt
        assert "billing" in prompt

    def test_prompt_ends_with_response(self, prompt_config):
        prompt = _build_prompt("Test email", prompt_config)
        assert prompt.strip().endswith("Response:")


# ---------------------------------------------------------------------------
# Tests for classify_email
# ---------------------------------------------------------------------------

class TestClassifyEmail:
    @patch("src.classifier._get_client")
    def test_successful_classification(self, mock_get_client, prompt_config):
        # Mock the Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "billing", "summary": "Customer wants refund"}'
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 20
        mock_client.models.generate_content.return_value = mock_response

        result = classify_email(
            "I need a refund please",
            prompt_config,
            client=mock_client,
        )

        assert result.category == EmailCategory.BILLING
        assert result.summary == "Customer wants refund"
        assert result.total_tokens == 120

    @patch("src.classifier._get_client")
    def test_retry_on_failure(self, mock_get_client, prompt_config):
        mock_client = MagicMock()

        # First call fails, second succeeds
        mock_response = MagicMock()
        mock_response.text = '{"category": "technical", "summary": "Bug report"}'
        mock_response.usage_metadata = None

        mock_client.models.generate_content.side_effect = [
            Exception("API Error"),
            mock_response,
        ]

        result = classify_email(
            "The app crashes",
            prompt_config,
            client=mock_client,
            max_retries=2,
        )

        assert result.category == EmailCategory.TECHNICAL
        assert mock_client.models.generate_content.call_count == 2

    @patch("src.classifier._get_client")
    def test_all_retries_fail(self, mock_get_client, prompt_config):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Persistent error")

        result = classify_email(
            "Test email",
            prompt_config,
            client=mock_client,
            max_retries=2,
        )

        # Should return a result with error info instead of raising
        assert result.category == EmailCategory.GENERAL
        assert "failed" in result.summary.lower() or "error" in result.summary.lower()

    @patch("src.classifier._get_client")
    def test_invalid_category_defaults_to_general(self, mock_get_client, prompt_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "unknown_category", "summary": "test"}'
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        result = classify_email("Test", prompt_config, client=mock_client)
        assert result.category == EmailCategory.GENERAL

    @patch("src.classifier._get_client")
    def test_latency_tracked(self, mock_get_client, prompt_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "general", "summary": "test"}'
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        result = classify_email("Test", prompt_config, client=mock_client)
        assert result.latency_ms > 0  # Should have measured some time
