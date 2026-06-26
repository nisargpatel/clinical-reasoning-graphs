from __future__ import annotations
import os
"""
Extract clinical reasoning graphs from LLM diagnostic reasoning traces.

Usage:
    python -m src.extract --input data/raw/results.jsonl --cases data/raw/cases.json --output data/extracted/
    python -m src.extract --input data/raw/results.jsonl --cases data/raw/cases.json --output data/extracted/ --sample 10
    python -m src.extract --input data/raw/results.jsonl --cases data/raw/cases.json --output data/extracted_no_gold/ --no-gold
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from tqdm import tqdm

from prompts.extraction import build_extraction_messages
from src.utils import (
    get_openai_client,
    get_case_presentation,
    get_trace_text,
    load_cases,
    load_traces,
    parse_json_response,
    save_json,
)


def extract_single_graph(
    client,
    trace: dict,
    case: dict,
    model_name: str = "gpt-5.4",
    max_retries: int = 3,
    include_gold: bool = True,
) -> dict | None:
    """Extract a reasoning graph from a single trace using GPT-5.4."""
    reasoning_text = get_trace_text(trace)
    case_presentation = get_case_presentation(case)
    correct_diagnosis = case.get("correct_diagnosis", case.get("diagnosis", "Unknown"))
    condition = trace.get("condition", "unknown")
    source_model = trace.get("model", "unknown")

    messages = build_extraction_messages(
        case_presentation=case_presentation,
        correct_diagnosis=correct_diagnosis,
        model_name=source_model,
        condition=condition,
        reasoning_trace=reasoning_text,
        include_gold=include_gold,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32768,
            )
            raw = response.choices[0].message.content
            graph = parse_json_response(raw)

            # Attach metadata from both extraction and original trace
            graph["_metadata"] = {
                "case_id": trace.get("case_id"),
                "source_model": source_model,
                "condition": condition,
                "difficulty": trace.get("difficulty"),
                "category": trace.get("category"),
                "ground_truth": trace.get("ground_truth"),
                "provider": trace.get("provider"),
                "tier": trace.get("tier"),
                "family": trace.get("family"),
                "extractor_model": model_name,
                "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "source_response_tokens": trace.get("output_tokens"),
                "extraction_variant": "no_gold" if not include_gold else "with_gold",
            }
            return graph

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"_error": str(e), "_raw": raw, "_metadata": {
                    "case_id": trace.get("case_id"),
                    "source_model": source_model,
                    "condition": condition,
                }}
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"_error": str(e), "_metadata": {
                    "case_id": trace.get("case_id"),
                    "source_model": source_model,
                    "condition": condition,
                }}

    return None


def run_extraction(
    traces: list[dict],
    cases: dict[str, dict],
    output_dir: Path,
    sample_n: int | None = None,
    extractor_model: str = "gpt-5.4",
    include_gold: bool = True,
):
    """Run graph extraction on all traces."""
    client = get_openai_client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_n:
        import random
        traces = random.sample(traces, min(sample_n, len(traces)))

    # Track incremental results
    results_path = output_dir / "graphs_incremental.jsonl"
    completed_keys = set()

    # Resume from previous run if exists
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                obj = json.loads(line)
                meta = obj.get("_metadata", {})
                key = f"{meta.get('case_id')}_{meta.get('source_model')}_{meta.get('condition')}"
                completed_keys.add(key)
        print(f"Resuming: {len(completed_keys)} traces already extracted.")

    remaining = []
    for trace in traces:
        key = f"{trace.get('case_id')}_{trace.get('model')}_{trace.get('condition')}"
        if key not in completed_keys:
            remaining.append(trace)

    variant_label = "WITHOUT gold diagnosis" if not include_gold else "WITH gold diagnosis"
    print(f"Extracting {len(remaining)} graphs ({len(completed_keys)} already done)")
    print(f"Extractor model: {extractor_model}")
    print(f"Extraction variant: {variant_label}")

    errors = 0
    for trace in tqdm(remaining, desc="Extracting graphs"):
        case_id = trace.get("case_id")
        case = cases.get(case_id)

        if case is None:
            print(f"  WARNING: No case found for {case_id}, skipping")
            continue

        graph = extract_single_graph(
            client=client,
            trace=trace,
            case=case,
            model_name=extractor_model,
            include_gold=include_gold,
        )

        if graph is None:
            errors += 1
            continue

        if "_error" in graph:
            errors += 1

        # Save incrementally
        with open(results_path, "a") as f:
            f.write(json.dumps(graph) + "\n")

        # Rate limiting
        time.sleep(0.5)

    # Compile final output
    all_graphs = []
    with open(results_path) as f:
        for line in f:
            all_graphs.append(json.loads(line))

    save_json(all_graphs, output_dir / "all_graphs.json")

    # Summary stats
    summary = {
        "total_traces": len(traces),
        "extracted": len(all_graphs),
        "errors": errors,
        "extractor_model": extractor_model,
        "extraction_variant": "no_gold" if not include_gold else "with_gold",
        "unique_models": list(set(
            g["_metadata"]["source_model"]
            for g in all_graphs
            if "_metadata" in g and "source_model" in g["_metadata"]
        )),
        "conditions": list(set(
            g["_metadata"]["condition"]
            for g in all_graphs
            if "_metadata" in g and "condition" in g["_metadata"]
        )),
    }
    save_json(summary, output_dir / "extraction_summary.json")
    print(f"\nDone. {len(all_graphs)} graphs extracted, {errors} errors.")
    print(f"Results: {output_dir / 'all_graphs.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reasoning graphs from traces")
    parser.add_argument("--input", required=True, help="Path to traces JSONL or JSON")
    parser.add_argument("--cases", required=True, help="Path to cases.json")
    parser.add_argument("--output", default="data/extracted/", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Extract only N traces (for testing)")
    parser.add_argument("--model", default=None, help="Extractor model (default: from EXTRACTOR_MODEL env var or openai/gpt-5.4)")
    parser.add_argument("--no-gold", action="store_true",
                        help="Omit the reference diagnosis from the extraction prompt")
    args = parser.parse_args()

    extractor_model = args.model or os.getenv("EXTRACTOR_MODEL", "openai/gpt-5.4")

    traces = load_traces(args.input)
    cases = load_cases(args.cases)

    run_extraction(
        traces=traces,
        cases=cases,
        output_dir=Path(args.output),
        sample_n=args.sample,
        extractor_model=extractor_model,
        include_gold=not args.no_gold,
    )
