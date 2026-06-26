from __future__ import annotations
"""
Reasoning motif extraction and analysis.

Identifies recurring reasoning patterns across graphs and compares
motif distributions across models, conditions, and case clusters.
"""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from src.similarity import extract_motifs, extract_phase_motifs


# Canonical motif labels for interpretability
MOTIF_LABELS = {
    ("clinical_feature", "supports", "diagnosis"): "feature_supports_dx",
    ("clinical_feature", "argues_against", "diagnosis"): "feature_argues_against_dx",
    ("discriminating_feature", "discriminates_between", "diagnosis"): "feature_discriminates_dx",
    ("clinical_feature", "triggered_reflection", "reflection_event"): "feature_triggers_reflection",
    ("reflection_event", "promoted", "diagnosis"): "reflection_promotes_dx",
    ("reflection_event", "demoted", "diagnosis"): "reflection_demotes_dx",
    ("reflection_event", "unchanged", "diagnosis"): "reflection_unchanged_dx",
    ("evidence_reference", "supports", "diagnosis"): "evidence_supports_dx",
    ("semantic_qualifier", "supports", "diagnosis"): "qualifier_supports_dx",
}


def label_motif(motif: tuple) -> str:
    """Convert a motif triple to a human-readable label."""
    return MOTIF_LABELS.get(motif, f"{motif[0]}→{motif[1]}→{motif[2]}")


def aggregate_motifs(graphs: list[dict]) -> pd.DataFrame:
    """Aggregate motif counts across a set of graphs.
    
    Returns a DataFrame with one row per graph and columns for each motif type.
    """
    all_motif_counts = []
    all_motif_types = set()

    for g in graphs:
        motifs = extract_motifs(g)
        labeled = {label_motif(k): v for k, v in motifs.items()}
        all_motif_counts.append(labeled)
        all_motif_types.update(labeled.keys())

    # Build DataFrame
    rows = []
    for i, g in enumerate(graphs):
        meta = g.get("_metadata", {})
        row = {
            "case_id": meta.get("case_id"),
            "model": meta.get("source_model"),
            "condition": meta.get("condition"),
        }
        for motif_type in all_motif_types:
            row[motif_type] = all_motif_counts[i].get(motif_type, 0)
        rows.append(row)

    return pd.DataFrame(rows)


def motif_distribution_by_condition(graphs: list[dict]) -> dict:
    """Compare motif distributions across prompt conditions.
    
    Returns per-condition motif frequency distributions and statistical
    comparison between baseline and structured reflection.
    """
    condition_motifs = {"baseline": Counter(), "adversarial": Counter(), "structured": Counter()}
    condition_counts = {"baseline": 0, "adversarial": 0, "structured": 0}

    for g in graphs:
        condition = g.get("_metadata", {}).get("condition", "unknown")
        if condition in condition_motifs:
            motifs = extract_motifs(g)
            for motif, count in motifs.items():
                condition_motifs[condition][label_motif(motif)] += count
            condition_counts[condition] += 1

    # Normalize to per-graph rates
    distributions = {}
    for cond in condition_motifs:
        n = condition_counts[cond] or 1
        distributions[cond] = {
            motif: count / n
            for motif, count in condition_motifs[cond].items()
        }

    # Statistical comparison: baseline vs structured for key motifs
    comparisons = {}
    key_motifs = [
        "feature_discriminates_dx",
        "feature_supports_dx",
        "feature_argues_against_dx",
        "feature_triggers_reflection",
    ]

    for motif_label in key_motifs:
        baseline_counts = []
        structured_counts = []

        for g in graphs:
            condition = g.get("_metadata", {}).get("condition")
            motifs = extract_motifs(g)
            labeled = {label_motif(k): v for k, v in motifs.items()}
            count = labeled.get(motif_label, 0)

            if condition == "baseline":
                baseline_counts.append(count)
            elif condition == "structured":
                structured_counts.append(count)

        if baseline_counts and structured_counts:
            stat, p = mannwhitneyu(
                baseline_counts, structured_counts, alternative="two-sided"
            )
            comparisons[motif_label] = {
                "baseline_mean": float(np.mean(baseline_counts)),
                "structured_mean": float(np.mean(structured_counts)),
                "baseline_median": float(np.median(baseline_counts)),
                "structured_median": float(np.median(structured_counts)),
                "mann_whitney_U": float(stat),
                "p_value": float(p),
                "effect_direction": "higher_in_structured"
                if np.mean(structured_counts) > np.mean(baseline_counts)
                else "higher_in_baseline",
            }

    return {
        "distributions": distributions,
        "comparisons": comparisons,
        "condition_counts": condition_counts,
    }


def discriminating_motif_ratio(graphs: list[dict]) -> pd.DataFrame:
    """For each graph, compute the ratio of discriminating motifs to total motifs.
    
    This is the key metric: expert diagnosticians have a higher ratio of
    discriminating reasoning to simple support reasoning.
    """
    rows = []
    for g in graphs:
        meta = g.get("_metadata", {})
        motifs = extract_motifs(g)
        labeled = {label_motif(k): v for k, v in motifs.items()}

        total = sum(labeled.values()) or 1
        discriminating = labeled.get("feature_discriminates_dx", 0)
        supporting = labeled.get("feature_supports_dx", 0)
        arguing = labeled.get("feature_argues_against_dx", 0)

        rows.append({
            "case_id": meta.get("case_id"),
            "model": meta.get("source_model"),
            "condition": meta.get("condition"),
            "total_motifs": total,
            "discriminating_count": discriminating,
            "supporting_count": supporting,
            "arguing_count": arguing,
            "discriminating_ratio": discriminating / total,
            "supporting_ratio": supporting / total,
            "expert_reasoning_index": (discriminating + arguing) / total,
        })

    return pd.DataFrame(rows)
