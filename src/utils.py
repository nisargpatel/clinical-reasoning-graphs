from __future__ import annotations
"""Shared utilities for the clinical reasoning graphs pipeline."""

import json
import os
from pathlib import Path
from typing import Any

import jsonlines
from dotenv import load_dotenv

load_dotenv()


def get_openai_client():
    """Get OpenAI client — works with both direct OpenAI and OpenRouter."""
    from openai import OpenAI

    base_url = os.getenv("OPENAI_BASE_URL", None)
    api_key = os.getenv("OPENAI_API_KEY")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def load_traces(path: str | Path) -> list[dict]:
    """Load reasoning traces from JSONL or JSON file."""
    path = Path(path)
    traces = []

    if path.suffix == ".jsonl":
        with jsonlines.open(path) as reader:
            for obj in reader:
                traces.append(obj)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                traces = data
            else:
                traces = [data]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return traces


def load_cases(path: str | Path) -> dict[str, dict]:
    """Load case definitions, keyed by case_id."""
    with open(path) as f:
        cases = json.load(f)
    # Key by case_id; ground_truth field holds correct diagnosis
    return {c["case_id"]: c for c in cases}


def get_trace_text(trace: dict) -> str:
    """Extract the reasoning text from a trace record."""
    if "response" in trace and trace["response"]:
        return trace["response"]
    raise ValueError(f"No reasoning text found in trace. Keys: {list(trace.keys())}")


def get_case_presentation(case: dict) -> str:
    """Build the full case presentation string from case fields.
    
    Matches the cases.json format: presentation + labs + additional.
    """
    parts = []
    if "presentation" in case:
        parts.append(case["presentation"])
    if "labs" in case:
        parts.append(f"\nLaboratory and Imaging:\n{case['labs']}")
    if "additional" in case and case["additional"]:
        parts.append(f"\nAdditional:\n{case['additional']}")
    return "\n".join(parts)


def save_json(data: Any, path: str | Path, indent: int = 2):
    """Save data as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def parse_json_response(text: str) -> dict:
    """Parse a JSON response from an LLM, handling common formatting issues."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())
