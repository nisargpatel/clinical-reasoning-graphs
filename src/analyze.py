from __future__ import annotations
"""
Main analysis: illness script consistency, reflection effect, cross-model comparison.

Usage:
    python -m src.analyze --graphs data/extracted/all_graphs.json \
                          --clusters data/clusters/clusters.json \
                          --output results/
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, kruskal

from src.similarity import (
    composite_similarity,
    pairwise_similarity_matrix,
    reasoning_depth,
    classify_cluster_pair,
)
from src.utils import save_json
from analysis.motifs import (
    motif_distribution_by_condition,
    discriminating_motif_ratio,
    aggregate_motifs,
)


def load_graphs_and_clusters(graphs_path: str, clusters_path: str):
    """Load extracted graphs and cluster assignments."""
    with open(graphs_path) as f:
        all_graphs = json.load(f)

    with open(clusters_path) as f:
        clusters = json.load(f)

    # Filter out error entries
    graphs = [g for g in all_graphs if "_error" not in g]
    print(f"Loaded {len(graphs)} valid graphs, {len(all_graphs) - len(graphs)} errors")

    return graphs, clusters


def get_case_cluster(case_id: str, clusters: dict) -> str | None:
    """Find which cluster a case belongs to."""
    for cid, info in clusters.items():
        if case_id in info.get("case_ids", []):
            return cid
    return None


# ──────────────────────────────────────────────
# ANALYSIS 1: ILLNESS SCRIPT CONSISTENCY
# ──────────────────────────────────────────────

def diagnostic_schema_consistency(graphs: list[dict], clusters: dict) -> dict:
    """Test whether models show higher reasoning graph similarity for
    clinically similar cases (same cluster) vs dissimilar cases.

    This is the core test of latent diagnostic schema stability.
    """
    # Group graphs by model and condition
    model_condition_graphs = defaultdict(list)
    for g in graphs:
        meta = g.get("_metadata", {})
        key = (meta.get("source_model"), meta.get("condition"))
        model_condition_graphs[key].append(g)

    results = {}

    for (model, condition), model_graphs in model_condition_graphs.items():
        within_cluster_sims = []
        between_cluster_sims = []

        for i, j in combinations(range(len(model_graphs)), 2):
            g_i, g_j = model_graphs[i], model_graphs[j]
            case_i = g_i.get("_metadata", {}).get("case_id")
            case_j = g_j.get("_metadata", {}).get("case_id")

            cluster_i = get_case_cluster(case_i, clusters)
            cluster_j = get_case_cluster(case_j, clusters)

            sim = composite_similarity(g_i, g_j)

            cls = classify_cluster_pair(case_i, case_j, cluster_i, cluster_j)
            if cls == "within":
                within_cluster_sims.append(sim["composite"])
            elif cls == "between":
                between_cluster_sims.append(sim["composite"])

        # Statistical test
        if within_cluster_sims and between_cluster_sims:
            stat, p = mannwhitneyu(
                within_cluster_sims,
                between_cluster_sims,
                alternative="greater",  # one-sided: within > between
            )
            effect_size = (
                np.mean(within_cluster_sims) - np.mean(between_cluster_sims)
            ) / np.std(within_cluster_sims + between_cluster_sims)
        else:
            stat, p, effect_size = None, None, None

        results[f"{model}__{condition}"] = {
            "model": model,
            "condition": condition,
            "within_cluster_mean": float(np.mean(within_cluster_sims)) if within_cluster_sims else None,
            "within_cluster_std": float(np.std(within_cluster_sims)) if within_cluster_sims else None,
            "between_cluster_mean": float(np.mean(between_cluster_sims)) if between_cluster_sims else None,
            "between_cluster_std": float(np.std(between_cluster_sims)) if between_cluster_sims else None,
            "within_n": len(within_cluster_sims),
            "between_n": len(between_cluster_sims),
            "mann_whitney_U": float(stat) if stat else None,
            "p_value": float(p) if p else None,
            "cohens_d": float(effect_size) if effect_size else None,
            "has_illness_scripts": p < 0.05 if p else None,
        }

    return results


# ──────────────────────────────────────────────
# ANALYSIS 2: REFLECTION EFFECT ON CONSISTENCY
# ──────────────────────────────────────────────

def reflection_consistency_effect(graphs: list[dict], clusters: dict) -> dict:
    """Test whether structured reflection increases within-cluster
    graph similarity compared to baseline.
    
    Uses paired comparison: same model, same case cluster,
    baseline vs structured condition.
    """
    # Group by model
    models = set(g["_metadata"]["source_model"] for g in graphs if "_metadata" in g)
    results = {}

    for model in models:
        model_graphs = [
            g for g in graphs
            if g.get("_metadata", {}).get("source_model") == model
        ]

        baseline_within = []
        structured_within = []

        # Get within-cluster similarities for each condition
        for condition in ["baseline", "structured"]:
            cond_graphs = [
                g for g in model_graphs
                if g.get("_metadata", {}).get("condition") == condition
            ]

            for i, j in combinations(range(len(cond_graphs)), 2):
                g_i, g_j = cond_graphs[i], cond_graphs[j]
                case_i = g_i["_metadata"]["case_id"]
                case_j = g_j["_metadata"]["case_id"]

                cluster_i = get_case_cluster(case_i, clusters)
                cluster_j = get_case_cluster(case_j, clusters)

                if classify_cluster_pair(case_i, case_j, cluster_i, cluster_j) == "within":
                    sim = composite_similarity(g_i, g_j)["composite"]
                    if condition == "baseline":
                        baseline_within.append(sim)
                    else:
                        structured_within.append(sim)

        if baseline_within and structured_within:
            stat, p = mannwhitneyu(
                structured_within, baseline_within,
                alternative="greater",
            )
            delta = np.mean(structured_within) - np.mean(baseline_within)
        else:
            stat, p, delta = None, None, None

        results[model] = {
            "baseline_within_mean": float(np.mean(baseline_within)) if baseline_within else None,
            "structured_within_mean": float(np.mean(structured_within)) if structured_within else None,
            "delta": float(delta) if delta else None,
            "mann_whitney_U": float(stat) if stat else None,
            "p_value": float(p) if p else None,
            "reflection_stabilizes": delta > 0 and p < 0.05 if delta and p else None,
            "n_baseline_pairs": len(baseline_within),
            "n_structured_pairs": len(structured_within),
        }

    return results


# ──────────────────────────────────────────────
# ANALYSIS 3: CROSS-MODEL COMPARISON
# ──────────────────────────────────────────────

def cross_model_consistency_ranking(graphs: list[dict], clusters: dict) -> dict:
    """Rank models by their within-cluster reasoning consistency.
    
    The most 'expert-like' model shows the highest within-cluster
    similarity — it has the most stable diagnostic schemas.
    """
    models = set(g["_metadata"]["source_model"] for g in graphs if "_metadata" in g)
    conditions = set(g["_metadata"]["condition"] for g in graphs if "_metadata" in g)

    rankings = {}

    for condition in conditions:
        model_scores = {}

        for model in models:
            model_cond_graphs = [
                g for g in graphs
                if g.get("_metadata", {}).get("source_model") == model
                and g.get("_metadata", {}).get("condition") == condition
            ]

            within_sims = []
            for i, j in combinations(range(len(model_cond_graphs)), 2):
                g_i, g_j = model_cond_graphs[i], model_cond_graphs[j]
                ci_id = g_i["_metadata"]["case_id"]
                cj_id = g_j["_metadata"]["case_id"]
                ci = get_case_cluster(ci_id, clusters)
                cj = get_case_cluster(cj_id, clusters)
                if classify_cluster_pair(ci_id, cj_id, ci, cj) == "within":
                    within_sims.append(
                        composite_similarity(g_i, g_j)["composite"]
                    )

            model_scores[model] = {
                "mean_within_similarity": float(np.mean(within_sims)) if within_sims else 0,
                "std_within_similarity": float(np.std(within_sims)) if within_sims else 0,
                "n_pairs": len(within_sims),
            }

        # Rank by mean within-cluster similarity
        ranked = sorted(
            model_scores.items(),
            key=lambda x: x[1]["mean_within_similarity"],
            reverse=True,
        )

        rankings[condition] = {
            "ranking": [
                {"rank": i + 1, "model": m, **scores}
                for i, (m, scores) in enumerate(ranked)
            ],
        }

        # Kruskal-Wallis test across models
        model_sim_groups = []
        for model in models:
            model_cond_graphs = [
                g for g in graphs
                if g.get("_metadata", {}).get("source_model") == model
                and g.get("_metadata", {}).get("condition") == condition
            ]
            within = []
            for i, j in combinations(range(len(model_cond_graphs)), 2):
                g_i, g_j = model_cond_graphs[i], model_cond_graphs[j]
                ci_id = g_i["_metadata"]["case_id"]
                cj_id = g_j["_metadata"]["case_id"]
                ci = get_case_cluster(ci_id, clusters)
                cj = get_case_cluster(cj_id, clusters)
                if classify_cluster_pair(ci_id, cj_id, ci, cj) == "within":
                    within.append(composite_similarity(g_i, g_j)["composite"])
            if within:
                model_sim_groups.append(within)

        if len(model_sim_groups) >= 2:
            h_stat, h_p = kruskal(*model_sim_groups)
            rankings[condition]["kruskal_wallis_H"] = float(h_stat)
            rankings[condition]["kruskal_wallis_p"] = float(h_p)
            rankings[condition]["models_differ"] = h_p < 0.05

    return rankings


# ──────────────────────────────────────────────
# ANALYSIS 4: REASONING DEPTH BY CONDITION
# ──────────────────────────────────────────────

def reasoning_depth_analysis(graphs: list[dict]) -> dict:
    """Compare reasoning depth metrics across conditions."""
    rows = []
    for g in graphs:
        meta = g.get("_metadata", {})
        depth = reasoning_depth(g)
        rows.append({
            "model": meta.get("source_model"),
            "condition": meta.get("condition"),
            **depth,
        })

    df = pd.DataFrame(rows)

    results = {}
    for metric in ["density", "discriminating_ratio", "reflection_density", "n_edges"]:
        condition_stats = {}
        for cond in ["baseline", "adversarial", "structured"]:
            vals = df[df["condition"] == cond][metric].dropna()
            condition_stats[cond] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(vals.median()),
            }

        # Test baseline vs structured
        baseline = df[df["condition"] == "baseline"][metric].dropna()
        structured = df[df["condition"] == "structured"][metric].dropna()
        if len(baseline) > 0 and len(structured) > 0:
            stat, p = mannwhitneyu(structured, baseline, alternative="two-sided")
        else:
            stat, p = None, None

        results[metric] = {
            "by_condition": condition_stats,
            "baseline_vs_structured_U": float(stat) if stat else None,
            "baseline_vs_structured_p": float(p) if p else None,
        }

    return results


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_analysis(graphs_path: str, clusters_path: str, output_dir: str):
    """Run all analyses and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs, clusters = load_graphs_and_clusters(graphs_path, clusters_path)

    print("Analysis 1: Illness script consistency...")
    consistency = diagnostic_schema_consistency(graphs, clusters)
    save_json(consistency, output_dir / "diagnostic_schema_consistency.json")

    print("Analysis 2: Reflection consistency effect...")
    reflection = reflection_consistency_effect(graphs, clusters)
    save_json(reflection, output_dir / "reflection_consistency_effect.json")

    print("Analysis 3: Cross-model consistency ranking...")
    ranking = cross_model_consistency_ranking(graphs, clusters)
    save_json(ranking, output_dir / "cross_model_ranking.json")

    print("Analysis 4: Reasoning depth by condition...")
    depth = reasoning_depth_analysis(graphs)
    save_json(depth, output_dir / "reasoning_depth.json")

    print("Analysis 5: Motif distributions by condition...")
    motif_dist = motif_distribution_by_condition(graphs)
    save_json(motif_dist, output_dir / "motif_distributions.json")

    print("Analysis 6: Discriminating motif ratios...")
    disc_df = discriminating_motif_ratio(graphs)
    disc_df.to_csv(output_dir / "discriminating_ratios.csv", index=False)

    # Print headline results
    print(f"\n{'=' * 60}")
    print("HEADLINE RESULTS")
    print(f"{'=' * 60}")

    print("\n--- Illness Script Consistency ---")
    for key, val in consistency.items():
        if val.get("p_value") is not None:
            sig = "***" if val["p_value"] < 0.001 else "**" if val["p_value"] < 0.01 else "*" if val["p_value"] < 0.05 else "ns"
            print(f"  {key}: within={val['within_cluster_mean']:.3f} vs "
                  f"between={val['between_cluster_mean']:.3f} "
                  f"(d={val['cohens_d']:.2f}, p={val['p_value']:.4f} {sig})")

    print("\n--- Reflection Stabilization Effect ---")
    for model, val in reflection.items():
        if val.get("delta") is not None:
            direction = "↑" if val["delta"] > 0 else "↓"
            sig = "*" if val.get("p_value", 1) < 0.05 else "ns"
            print(f"  {model}: Δ={val['delta']:+.3f} {direction} ({sig})")

    print("\n--- Cross-Model Ranking (Structured Condition) ---")
    if "structured" in ranking:
        for entry in ranking["structured"]["ranking"]:
            print(f"  #{entry['rank']}: {entry['model']} "
                  f"(consistency={entry['mean_within_similarity']:.3f})")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reasoning graph analyses")
    parser.add_argument("--graphs", required=True, help="Path to all_graphs.json")
    parser.add_argument("--clusters", required=True, help="Path to clusters.json")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    run_analysis(args.graphs, args.clusters, args.output)
