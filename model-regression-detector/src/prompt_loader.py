"""
Prompt version loader.

Reads versioned YAML prompt files from the /prompts directory and
parses them into PromptConfig objects. Supports listing all versions,
loading a specific version, or loading the latest by timestamp.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import FewShotExample, PromptConfig


# Default prompts directory relative to project root
_DEFAULT_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _parse_prompt_file(filepath: str) -> PromptConfig:
    """Parse a single YAML prompt file into a PromptConfig."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse few-shot examples
    few_shot = []
    for ex in data.get("few_shot_examples", []):
        output = ex.get("output", {})
        few_shot.append(FewShotExample(
            input=ex["input"],
            output_category=output.get("category", "general"),
            output_summary=output.get("summary", ""),
        ))

    # Parse timestamp — accept string or datetime
    ts = data.get("timestamp")
    if isinstance(ts, str):
        # Try ISO format first, fall back to basic parsing
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = datetime.now(timezone.utc)
    elif not isinstance(ts, datetime):
        ts = datetime.now(timezone.utc)

    # Ensure timezone-aware
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return PromptConfig(
        version=data["version"],
        timestamp=ts,
        model=data.get("model", "gemini-3.5-flash"),
        system_prompt=data["system_prompt"],
        few_shot_examples=few_shot,
        temperature=data.get("temperature", 0.0),
        max_output_tokens=data.get("max_output_tokens", 256),
    )


def list_versions(prompts_dir: str = _DEFAULT_PROMPTS_DIR) -> list[str]:
    """List all available prompt version IDs, sorted by filename."""
    pattern = os.path.join(prompts_dir, "*.yaml")
    files = sorted(glob.glob(pattern))
    versions = []
    for f in files:
        try:
            cfg = _parse_prompt_file(f)
            versions.append(cfg.version)
        except Exception:
            continue
    return versions


def load_prompt(
    version: str | None = None,
    prompts_dir: str = _DEFAULT_PROMPTS_DIR,
) -> PromptConfig:
    """
    Load a specific prompt version, or the latest if version is None.

    Args:
        version: Version ID to load (e.g., "v1"). If None, loads latest.
        prompts_dir: Directory containing YAML prompt files.

    Returns:
        PromptConfig for the requested version.

    Raises:
        FileNotFoundError: If no prompt files exist or version not found.
    """
    pattern = os.path.join(prompts_dir, "*.yaml")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No prompt files found in {prompts_dir}")

    # Parse all configs
    configs: list[tuple[str, PromptConfig]] = []
    for f in files:
        try:
            cfg = _parse_prompt_file(f)
            configs.append((f, cfg))
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")
            continue

    if not configs:
        raise FileNotFoundError("No valid prompt files found")

    # If specific version requested, find it
    if version is not None:
        for filepath, cfg in configs:
            if cfg.version == version:
                return cfg
        available = [c.version for _, c in configs]
        raise FileNotFoundError(
            f"Prompt version '{version}' not found. Available: {available}"
        )

    # Return latest by timestamp
    configs.sort(key=lambda x: x[1].timestamp)
    return configs[-1][1]


def load_all_prompts(
    prompts_dir: str = _DEFAULT_PROMPTS_DIR,
) -> list[PromptConfig]:
    """Load all prompt configs, sorted by version."""
    pattern = os.path.join(prompts_dir, "*.yaml")
    files = sorted(glob.glob(pattern))
    configs = []
    for f in files:
        try:
            configs.append(_parse_prompt_file(f))
        except Exception:
            continue
    return configs
