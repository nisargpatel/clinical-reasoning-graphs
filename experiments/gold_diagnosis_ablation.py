"""
Concern 5 ablation: Does providing the correct diagnosis to the extractor
bias the extracted graphs?

Re-extracts 20 traces with the gold diagnosis removed from the prompt,
then compares the resulting graphs to the original (gold-included) extractions.

Usage:
    # Step 1: Prepare traces
    python3 -m experiments.gold_diagnosis_ablation --step prepare

    # Step 2: Extract (costs ~$15-20)
    python3 -m experiments.gold_diagnosis_ablation --step extract

    # Step 3: Compare
    python3 -m experiments.gold_diagnosis_ablation --step compare
"""

from __future__ import annotations
import argparse
import json
import random
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')


# Modified extraction prompt with gold diagnosis removed
GRAPH_EXTRACTION_USER_NO_GOLD = """## Case Presentation

{case_presentation}

## Model Information

- Model: {model_name}
- Condition: {condition}

## Reasoning Trace

{reasoning_trace}

---

Extract the clinical reasoning graph from this trace as JSON. Return ONLY the JSON object,
no additional text or markdown formatting.
"""


def prepare(traces_path, output_dir):
    """Select 20 traces for re-extraction without gold diagnosis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(traces_path) as f:
        traces = [json.loads(l) for l in f]

    # Same 20 traces as validation V01-V20
    random.seed(42)
    by_cond = {"baseline": [], "adversarial": []}
    for t in traces:
        if t["condition"] in by_cond:
            by_cond[t["condition"]].append(t)

    sample = []
    for cond in ["baseline", "adversarial"]:
        sample.extend(random.sample(by_cond[cond], 10))

    traces_path = output_dir / "traces_no_gold.jsonl"
    with open(traces_path, "w") as f:
        for t in sample:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(sample)} traces to {traces_path}")
    for t in sample:
        print(f"  {t['case_id']} | {t['model'].split('/')[-1]} | {t['condition']}")

    print(f"\nNext: python3 -m experiments.gold_diagnosis_ablation --step extract")


def extract(output_dir, cases_path):
    """Extract graphs without gold diagnosis using modified prompt."""
    # extraction handled inline
    from prompts.extraction import GRAPH_EXTRACTION_SYSTEM

    output_dir = Path(output_dir)
    traces_path = output_dir / "traces_no_gold.jsonl"
    results_path = output_dir / "extracted_no_gold.jsonl"

    with open(cases_path) as f:
        cases = json.load(f)
    case_lookup = {c["case_id"]: c for c in cases}

    with open(traces_path) as f:
        traces = [json.loads(l) for l in f]

    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    extractor_model = os.environ.get("EXTRACTOR_MODEL", "openai/gpt-5.4")

    results = []
    for i, trace in enumerate(traces):
        case = case_lookup.get(trace["case_id"], {})
        case_presentation = case.get("presentation", trace.get("case_prompt", ""))

        # Build messages WITHOUT gold diagnosis
        messages = [
            {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": GRAPH_EXTRACTION_USER_NO_GOLD.format(
                    case_presentation=case_presentation,
                    model_name=trace["model"],
                    condition=trace["condition"],
                    reasoning_trace=trace["response"],
                ),
            },
        ]

        print(f"  [{i+1}/{len(traces)}] {trace['case_id']} | {trace['model'].split('/')[-1]} | {trace['condition']}", flush=True)

        try:
            response = client.chat.completions.create(
                model=extractor_model,
                messages=messages,
                max_tokens=32768,
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            graph = json.loads(content)
            graph["_metadata"] = {
                "case_id": trace["case_id"],
                "source_model": trace["model"],
                "condition": trace["condition"],
                "ground_truth": trace.get("ground_truth", ""),
                "extraction_variant": "no_gold_diagnosis",
                "extractor_model": extractor_model,
            }
            results.append(graph)

            with open(results_path, "a") as f:
                f.write(json.dumps(graph) + "\n")

            print(f"    nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}", flush=True)

        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            results.append({"_error": str(e), "_metadata": {
                "case_id": trace["case_id"],
                "source_model": trace["model"],
                "condition": trace["condition"],
            }})

    # Also save as single JSON
    all_path = output_dir / "all_graphs_no_gold.json"
    with open(all_path, "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if "_error" not in r]
    print(f"\nExtracted {len(valid)}/{len(traces)} graphs successfully")
    print(f"Saved to {all_path}")
    print(f"\nNext: python3 -m experiments.gold_diagnosis_ablation --step compare")


def compare(graphs_path, output_dir):
    """Compare gold-included vs gold-excluded extractions."""
    from src.similarity import composite_similarity

    output_dir = Path(output_dir)
    no_gold_path = output_dir / "all_graphs_no_gold.json"

    if not no_gold_path.exists():
        print(f"No-gold extractions not found at {no_gold_path}")
        print("Run --step extract first.")
        return

    # Load original (gold-included) extractions
    with open(graphs_path) as f:
        originals = {}
        for g in json.load(f):
            if "_error" in g:
                continue
            m = g["_metadata"]
            originals[(m["case_id"], m["source_model"], m["condition"])] = g

    # Load no-gold extractions
    with open(no_gold_path) as f:
        no_gold = [g for g in json.load(f) if "_error" not in g]

    # Pairwise comparison
    sims = []
    node_diffs = []
    edge_diffs = []
    dx_matches = []

    print("Comparing gold-included vs gold-excluded extractions:\n")

    for ng in no_gold:
        m = ng["_metadata"]
        key = (m["case_id"], m["source_model"], m["condition"])
        if key not in originals:
            continue

        orig = originals[key]
        sim = composite_similarity(orig, ng)["composite"]
        sims.append(sim)

        n_orig = len(orig.get("nodes", []))
        n_ng = len(ng.get("nodes", []))
        e_orig = len(orig.get("edges", []))
        e_ng = len(ng.get("edges", []))
        node_diffs.append(abs(n_orig - n_ng))
        edge_diffs.append(abs(e_orig - e_ng))

        # Check if final diagnosis changed
        fd_orig = (orig.get("final_diagnosis") or "").lower()
        fd_ng = (ng.get("final_diagnosis") or "").lower()
        # Simple check: do they share key terms?
        orig_words = set(fd_orig.split()) - {"with", "due", "to", "and", "of", "the", "a"}
        ng_words = set(fd_ng.split()) - {"with", "due", "to", "and", "of", "the", "a"}
        dx_match = len(orig_words & ng_words) / max(len(orig_words | ng_words), 1) > 0.3
        dx_matches.append(dx_match)

        print(f"  {m['case_id'][:20]:20s} | {m['source_model'].split('/')[-1]:15s} | {m['condition']:12s} | "
              f"sim={sim:.4f} | nodes {n_orig}→{n_ng} | edges {e_orig}→{e_ng} | "
              f"dx_match={'Y' if dx_match else 'N'}")

    print(f"\n{'='*70}")
    print(f"GOLD DIAGNOSIS ABLATION RESULTS (n={len(sims)})")
    print(f"{'='*70}")
    print(f"Composite similarity (gold vs no-gold):  {np.mean(sims):.4f} (SD={np.std(sims):.4f})")
    print(f"Test-retest (same extractor, same prompt): 0.825")
    print(f"Inter-extractor (GPT-5.4 vs Opus 4.7):    0.593")
    print(f"")
    print(f"Mean node count difference:   {np.mean(node_diffs):.1f}")
    print(f"Mean edge count difference:   {np.mean(edge_diffs):.1f}")
    print(f"Final diagnosis match rate:   {np.mean(dx_matches):.1%}")
    print(f"")

    if np.mean(sims) > 0.75:
        print("INTERPRETATION: Gold diagnosis has minimal effect on extraction.")
        print("Graphs are highly similar with or without the gold diagnosis.")
        print("")
        print("For the paper: 'Removing the gold diagnosis from the extraction prompt")
        print(f"changed composite graph similarity by {1 - np.mean(sims):.3f} (mean")
        print(f"similarity {np.mean(sims):.3f} between gold-included and gold-excluded")
        print("extractions, compared to test-retest of 0.825). The within-vs-between")
        print("conclusion is unchanged.'")
    elif np.mean(sims) > 0.60:
        print("INTERPRETATION: Gold diagnosis has moderate effect on extraction.")
        print("Some structural differences exist but overall patterns are preserved.")
        print("Report this as a limitation with the specific effect size.")
    else:
        print("INTERPRETATION: Gold diagnosis substantially affects extraction.")
        print("This is a significant concern that should be prominently disclosed.")
        print("Consider re-running all extractions without the gold diagnosis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold diagnosis ablation experiment")
    parser.add_argument("--step", required=True, choices=["prepare", "extract", "compare"])
    parser.add_argument("--traces", default="data/raw/results_incremental.jsonl")
    parser.add_argument("--cases", default="data/raw/cases.json")
    parser.add_argument("--graphs", default="data/extracted/all_graphs.json")
    parser.add_argument("--output", default="experiments/gold_ablation/")
    args = parser.parse_args()

    if args.step == "prepare":
        prepare(args.traces, args.output)
    elif args.step == "extract":
        extract(args.output, args.cases)
    elif args.step == "compare":
        compare(args.graphs, args.output)
