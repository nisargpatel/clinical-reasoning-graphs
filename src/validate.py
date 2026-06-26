from __future__ import annotations
"""
Validate graph extraction quality against human review.

Samples N extracted graphs, displays each alongside the source trace,
and records human judgments on extraction completeness and accuracy.

Usage:
    python -m src.validate --graphs data/extracted/all_graphs.json --sample 50
"""

import argparse
import json
import random
from pathlib import Path

from src.utils import save_json


def compute_graph_stats(graph: dict) -> dict:
    """Compute basic statistics for a single reasoning graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    reflections = graph.get("reflection_events", [])

    node_types = {}
    for n in nodes:
        t = n.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for e in edges:
        t = e.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    phase_counts = {}
    for e in edges:
        p = e.get("phase", "unknown")
        phase_counts[p] = phase_counts.get(p, 0) + 1

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_reflections": len(reflections),
        "node_types": node_types,
        "edge_types": edge_types,
        "phase_distribution": phase_counts,
        "has_problem_representation": bool(graph.get("problem_representation")),
        "has_final_diagnosis": bool(graph.get("final_diagnosis")),
        "has_discriminating_edges": edge_types.get("discriminates_between", 0) > 0,
        "graph_density": len(edges) / max(len(nodes), 1),
    }


def validate_graph_structure(graph: dict) -> list[str]:
    """Check for structural issues in an extracted graph."""
    issues = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {n["id"] for n in nodes}

    # Check edge references
    for e in edges:
        if e.get("source") not in node_ids:
            # Source might be a reflection event
            refl_ids = {r["id"] for r in graph.get("reflection_events", [])}
            if e.get("source") not in refl_ids:
                issues.append(f"Edge {e['id']}: source '{e.get('source')}' not found")
        targets = e.get("target")
        if isinstance(targets, list):
            for t in targets:
                if t not in node_ids:
                    refl_ids = {r["id"] for r in graph.get("reflection_events", [])}
                    if t not in refl_ids:
                        issues.append(f"Edge {e['id']}: target '{t}' not found")
        elif targets not in node_ids:
            refl_ids = {r["id"] for r in graph.get("reflection_events", [])}
            if targets not in refl_ids:
                issues.append(f"Edge {e['id']}: target '{targets}' not found")

    # Check for orphan nodes (no edges)
    connected = set()
    for e in edges:
        connected.add(e.get("source"))
        t = e.get("target")
        if isinstance(t, list):
            connected.update(t)
        else:
            connected.add(t)
    orphans = node_ids - connected
    if orphans:
        issues.append(f"Orphan nodes (no edges): {orphans}")

    # Check for duplicate node IDs
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate node IDs detected")

    # Check required fields
    for n in nodes:
        if "type" not in n:
            issues.append(f"Node {n.get('id')}: missing 'type'")
        if "label" not in n:
            issues.append(f"Node {n.get('id')}: missing 'label'")

    for e in edges:
        if "type" not in e:
            issues.append(f"Edge {e.get('id')}: missing 'type'")
        if "phase" not in e:
            issues.append(f"Edge {e.get('id')}: missing 'phase'")

    # Check condition-specific requirements
    meta = graph.get("_metadata", {})
    condition = meta.get("condition", "")

    if condition == "structured":
        if not graph.get("problem_representation"):
            issues.append("Structured condition but no problem_representation extracted")
        if not graph.get("reflection_events"):
            issues.append("Structured condition but no reflection_events extracted")

    if condition == "baseline":
        phases = {e.get("phase") for e in edges}
        if phases - {"phase1"}:
            issues.append(f"Baseline condition has non-phase1 edges: {phases}")

    return issues


def run_validation(graphs_path: str, sample_n: int = 50, output: str = None):
    """Run automated validation on extracted graphs."""
    with open(graphs_path) as f:
        graphs = json.load(f)

    # Filter out error entries
    valid_graphs = [g for g in graphs if "_error" not in g]
    error_graphs = [g for g in graphs if "_error" in g]

    print(f"Total graphs: {len(graphs)}")
    print(f"Valid: {len(valid_graphs)}, Errors: {len(error_graphs)}")

    if sample_n and sample_n < len(valid_graphs):
        sample = random.sample(valid_graphs, sample_n)
    else:
        sample = valid_graphs

    # Compute stats and validate each graph
    all_stats = []
    all_issues = []

    for graph in sample:
        stats = compute_graph_stats(graph)
        issues = validate_graph_structure(graph)
        meta = graph.get("_metadata", {})

        all_stats.append({
            "case_id": meta.get("case_id"),
            "model": meta.get("source_model"),
            "condition": meta.get("condition"),
            **stats,
            "structural_issues": issues,
            "issue_count": len(issues),
        })

        if issues:
            all_issues.append({
                "case_id": meta.get("case_id"),
                "model": meta.get("source_model"),
                "condition": meta.get("condition"),
                "issues": issues,
            })

    # Aggregate statistics
    import numpy as np

    node_counts = [s["total_nodes"] for s in all_stats]
    edge_counts = [s["total_edges"] for s in all_stats]
    densities = [s["graph_density"] for s in all_stats]

    summary = {
        "sample_size": len(sample),
        "graphs_with_issues": len(all_issues),
        "issue_rate": len(all_issues) / len(sample) if sample else 0,
        "node_stats": {
            "mean": float(np.mean(node_counts)),
            "median": float(np.median(node_counts)),
            "std": float(np.std(node_counts)),
            "min": int(np.min(node_counts)),
            "max": int(np.max(node_counts)),
        },
        "edge_stats": {
            "mean": float(np.mean(edge_counts)),
            "median": float(np.median(edge_counts)),
            "std": float(np.std(edge_counts)),
            "min": int(np.min(edge_counts)),
            "max": int(np.max(edge_counts)),
        },
        "density_stats": {
            "mean": float(np.mean(densities)),
            "median": float(np.median(densities)),
        },
        "problem_representation_rate": sum(
            1 for s in all_stats if s["has_problem_representation"]
        ) / len(all_stats),
        "discriminating_edge_rate": sum(
            1 for s in all_stats if s["has_discriminating_edges"]
        ) / len(all_stats),
        "issues": all_issues,
    }

    # Print report
    print(f"\n{'=' * 60}")
    print(f"VALIDATION REPORT (n={len(sample)})")
    print(f"{'=' * 60}")
    print(f"Graphs with structural issues: {len(all_issues)}/{len(sample)} "
          f"({summary['issue_rate']:.1%})")
    print(f"Nodes per graph: {summary['node_stats']['mean']:.1f} ± "
          f"{summary['node_stats']['std']:.1f} "
          f"(range {summary['node_stats']['min']}-{summary['node_stats']['max']})")
    print(f"Edges per graph: {summary['edge_stats']['mean']:.1f} ± "
          f"{summary['edge_stats']['std']:.1f}")
    print(f"Graph density: {summary['density_stats']['mean']:.2f}")
    print(f"Problem representation present: {summary['problem_representation_rate']:.1%}")
    print(f"Discriminating edges present: {summary['discriminating_edge_rate']:.1%}")

    if all_issues:
        print(f"\nTop issues:")
        from collections import Counter
        flat_issues = [i for entry in all_issues for i in entry["issues"]]
        for issue, count in Counter(flat_issues).most_common(5):
            print(f"  [{count}x] {issue[:80]}")

    if output:
        save_json(summary, output)
        print(f"\nFull report saved to {output}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate extracted reasoning graphs")
    parser.add_argument("--graphs", required=True, help="Path to all_graphs.json")
    parser.add_argument("--sample", type=int, default=50, help="Number to validate")
    parser.add_argument("--output", default="data/extracted/validation_report.json")
    args = parser.parse_args()

    run_validation(args.graphs, args.sample, args.output)
