from __future__ import annotations
"""
Extended analysis for this study.
Runs all analyses beyond the original src/analyze.py, including:
- Case-level permutation tests (replacing Mann-Whitney)
- Bootstrap CIs for all comparisons
- Similarity ablation by component
- Noise floor and test-retest calibration
- Semantic matching robustness
- Accuracy-correctness orthogonality test
- Difficulty tier breakdown
- Same-case repeated-generation positive control
- Content-only composite analysis
- Save raw pairwise similarity arrays
- Empirical Figure 2

Usage:
    python -m src.extended_analysis --graphs data/extracted/all_graphs.json \
                                    --clusters data/clusters/clusters.json \
                                    --scored data/scored_results_public.csv \
                                    --embeddings data/extracted/label_embeddings.json \
                                    --output results/extended/

    # Or run individual analyses:
    python -m src.extended_analysis --graphs ... --clusters ... --only permutation
    python -m src.extended_analysis --graphs ... --clusters ... --only ablation
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from src.similarity import composite_similarity, reasoning_depth, classify_cluster_pair
from src.analyze import get_case_cluster, load_graphs_and_clusters
from src.utils import save_json


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def precompute_pairwise(graphs, clusters, condition=None):
    """Precompute all pairwise similarities for a condition subset.
    Returns list of (case_i, case_j, cluster_i, cluster_j, sim) tuples."""
    subset = graphs if condition is None else [
        g for g in graphs if g.get("_metadata", {}).get("condition") == condition
    ]
    pairs = []
    for i, j in combinations(range(len(subset)), 2):
        ci_id = subset[i]["_metadata"]["case_id"]
        cj_id = subset[j]["_metadata"]["case_id"]
        ci = get_case_cluster(ci_id, clusters)
        cj = get_case_cluster(cj_id, clusters)
        if ci and cj:
            sim = composite_similarity(subset[i], subset[j])["composite"]
            pairs.append((ci_id, cj_id, ci, cj, sim))
    return pairs


# ══════════════════════════════════════════════════════════════
# 1. CASE-LEVEL PERMUTATION TESTS + BOOTSTRAP CIs
# ══════════════════════════════════════════════════════════════

def permutation_tests(graphs, clusters, n_perm=10000, n_boot=10000):
    """Case-level permutation tests for all 15 model-condition comparisons.
    Replaces Mann-Whitney U with dependency-aware tests."""
    print("Running case-level permutation tests...", flush=True)

    models = sorted(set(g["_metadata"]["source_model"] for g in graphs))
    conditions = ["baseline", "adversarial", "structured"]

    case_ids = list(set(g["_metadata"]["case_id"] for g in graphs))
    case_to_cluster = {cid: get_case_cluster(cid, clusters) for cid in case_ids}
    valid_cases = [c for c in case_ids if case_to_cluster[c]]

    np.random.seed(42)
    results = {}

    for model in models:
        for condition in conditions:
            subset = [g for g in graphs
                      if g["_metadata"]["source_model"] == model
                      and g["_metadata"]["condition"] == condition]
            if len(subset) < 2:
                continue

            # Precompute pairs
            pair_data = []
            for i, j in combinations(range(len(subset)), 2):
                ci_id = subset[i]["_metadata"]["case_id"]
                cj_id = subset[j]["_metadata"]["case_id"]
                ci = case_to_cluster.get(ci_id)
                cj = case_to_cluster.get(cj_id)
                if ci and cj:
                    sim = composite_similarity(subset[i], subset[j])["composite"]
                    pair_data.append((ci_id, cj_id, ci, cj, sim))

            within = [s for ca, cb, ci, cj, s in pair_data
                      if classify_cluster_pair(ca, cb, ci, cj) == "within"]
            between = [s for ca, cb, ci, cj, s in pair_data
                       if classify_cluster_pair(ca, cb, ci, cj) == "between"]

            if not within or not between:
                continue

            observed_delta = np.mean(within) - np.mean(between)

            # Case-level permutation
            cluster_vals = [case_to_cluster[c] for c in valid_cases]
            n_extreme = 0
            for _ in range(n_perm):
                shuffled = dict(zip(valid_cases, np.random.permutation(cluster_vals)))
                w = [s for ca, cb, _, _, s in pair_data
                     if classify_cluster_pair(ca, cb, shuffled.get(ca), shuffled.get(cb)) == "within"]
                b = [s for ca, cb, _, _, s in pair_data
                     if classify_cluster_pair(ca, cb, shuffled.get(ca), shuffled.get(cb)) == "between"]
                if w and b:
                    if np.mean(w) - np.mean(b) >= observed_delta:
                        n_extreme += 1
            perm_p = (n_extreme + 1) / (n_perm + 1)

            # Bootstrap CI
            within_arr, between_arr = np.array(within), np.array(between)
            boot_deltas = []
            for _ in range(n_boot):
                w_boot = np.random.choice(within_arr, size=len(within_arr), replace=True)
                b_boot = np.random.choice(between_arr, size=len(between_arr), replace=True)
                boot_deltas.append(np.mean(w_boot) - np.mean(b_boot))
            ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

            model_short = model.split("/")[-1]
            key = f"{model_short}__{condition}"
            results[key] = {
                "model": model_short,
                "condition": condition,
                "within_mean": float(np.mean(within)),
                "between_mean": float(np.mean(between)),
                "delta": float(observed_delta),
                "perm_p": float(perm_p),
                "ci_95_lower": float(ci_lo),
                "ci_95_upper": float(ci_hi),
                "n_within": len(within),
                "n_between": len(between),
                "within_equivalence_bound": abs(observed_delta) < 0.05,
            }

            print(f"  {model_short:20s} {condition:12s} Δ={observed_delta:+.4f} "
                  f"p={perm_p:.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]", flush=True)

    return results


# ══════════════════════════════════════════════════════════════
# 2. SIMILARITY ABLATION BY COMPONENT
# ══════════════════════════════════════════════════════════════

def similarity_ablation(graphs, clusters):
    """Test illness script consistency using each component individually."""
    print("Running similarity ablation...", flush=True)
    from src.similarity import (feature_overlap, diagnosis_overlap, motif_similarity,
                                depth_similarity)

    def qualifier_overlap(a, b):
        qa = {n["label"] for n in a.get("nodes", []) if n.get("type") == "semantic_qualifier"}
        qb = {n["label"] for n in b.get("nodes", []) if n.get("type") == "semantic_qualifier"}
        if not qa and not qb:
            return 0.0
        return len(qa & qb) / len(qa | qb) if (qa | qb) else 0.0

    metrics = {
        "feature_overlap": feature_overlap,
        "diagnosis_overlap": diagnosis_overlap,
        "motif_similarity": motif_similarity,
        "qualifier_overlap": qualifier_overlap,
        # canonical depth_similarity — identical function/value to the composite's depth input
        "depth_similarity": depth_similarity,
        "composite": lambda a, b: composite_similarity(a, b)["composite"],
    }

    struct = [g for g in graphs if g.get("_metadata", {}).get("condition") == "structured"]
    results = {}

    for name, func in metrics.items():
        within, between = [], []
        for i, j in combinations(range(len(struct)), 2):
            ci_id = struct[i]["_metadata"]["case_id"]
            cj_id = struct[j]["_metadata"]["case_id"]
            ci = get_case_cluster(ci_id, clusters)
            cj = get_case_cluster(cj_id, clusters)
            cls = classify_cluster_pair(ci_id, cj_id, ci, cj)
            if cls:
                try:
                    sim = func(struct[i], struct[j])
                    (within if cls == "within" else between).append(sim)
                except Exception:
                    continue

        diff = np.mean(within) - np.mean(between) if within and between else 0
        results[name] = {
            "within": float(np.mean(within)) if within else None,
            "between": float(np.mean(between)) if between else None,
            "delta": float(diff),
            "n_within": len(within),
            "n_between": len(between),
        }
        print(f"  {name:25s} within={np.mean(within):.4f} between={np.mean(between):.4f} Δ={diff:+.4f}", flush=True)

    # Content-only composite
    content_keys = ["feature_overlap", "diagnosis_overlap", "qualifier_overlap"]
    content_within = np.mean([results[k]["within"] for k in content_keys])
    content_between = np.mean([results[k]["between"] for k in content_keys])
    results["content_only_composite"] = {
        "within": float(content_within),
        "between": float(content_between),
        "delta": float(content_within - content_between),
    }
    print(f"  {'content_only':25s} within={content_within:.4f} between={content_between:.4f} Δ={content_within - content_between:+.4f}", flush=True)

    return results


# ══════════════════════════════════════════════════════════════
# 3. NOISE FLOOR
# ══════════════════════════════════════════════════════════════

def noise_floor(graphs, n_pairs=50):
    """Compute similarity between graphs with randomly shuffled labels."""
    print("Computing noise floor...", flush=True)
    random.seed(42)
    sample = random.sample(graphs, min(n_pairs * 2, len(graphs)))

    sims = []
    for i in range(0, len(sample) - 1, 2):
        g1 = json.loads(json.dumps(sample[i]))
        g2 = json.loads(json.dumps(sample[i + 1]))
        labels = [n["label"] for n in g2["nodes"]]
        random.shuffle(labels)
        for k, n in enumerate(g2["nodes"]):
            n["label"] = labels[k]
        try:
            sims.append(composite_similarity(g1, g2)["composite"])
        except Exception:
            continue

    result = {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "n": len(sims),
    }
    print(f"  Noise floor: {result['mean']:.4f} ± {result['std']:.4f} (n={result['n']})", flush=True)
    return result


# ══════════════════════════════════════════════════════════════
# 4. ACCURACY-CORRECTNESS ORTHOGONALITY
# ══════════════════════════════════════════════════════════════

def accuracy_orthogonality(graphs, clusters, scored_path):
    """Test whether graph similarity differs between correct-correct
    and incorrect-incorrect model pairs on the same case."""
    print("Running accuracy orthogonality test...", flush=True)

    correct_lookup = {}
    with open(scored_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["case_id"], row["model"], row["condition"])
            correct_lookup[key] = row["top1_correct"].strip().lower() in ("true", "1", "yes")

    def is_correct(g):
        m = g["_metadata"]
        return correct_lookup.get((m["case_id"], m["source_model"], m["condition"]), False)

    # Group by case + condition, compare model pairs
    groups = defaultdict(list)
    for g in graphs:
        m = g["_metadata"]
        groups[(m["case_id"], m["condition"])].append(g)

    both_correct, both_incorrect = [], []
    for key, gs in groups.items():
        for i, j in combinations(range(len(gs)), 2):
            ci, cj = is_correct(gs[i]), is_correct(gs[j])
            if ci == cj:
                sim = composite_similarity(gs[i], gs[j])["composite"]
                (both_correct if ci else both_incorrect).append(sim)

    diff = np.mean(both_correct) - np.mean(both_incorrect)
    se = np.sqrt(np.var(both_correct)/len(both_correct) + np.var(both_incorrect)/len(both_incorrect))
    stat, p = mannwhitneyu(both_correct, both_incorrect, alternative='two-sided')

    result = {
        "both_correct_mean": float(np.mean(both_correct)),
        "both_correct_n": len(both_correct),
        "both_incorrect_mean": float(np.mean(both_incorrect)),
        "both_incorrect_n": len(both_incorrect),
        "delta": float(diff),
        "ci_95_lower": float(diff - 1.96 * se),
        "ci_95_upper": float(diff + 1.96 * se),
        "mann_whitney_U": float(stat),
        "p_value": float(p),
    }
    print(f"  Correct-correct: {result['both_correct_mean']:.4f} (n={result['both_correct_n']})", flush=True)
    print(f"  Incorrect-incorrect: {result['both_incorrect_mean']:.4f} (n={result['both_incorrect_n']})", flush=True)
    print(f"  Δ={result['delta']:.4f}, p={result['p_value']:.4f}", flush=True)
    return result


# ══════════════════════════════════════════════════════════════
# 5. DIFFICULTY TIER BREAKDOWN
# ══════════════════════════════════════════════════════════════

def difficulty_breakdown(graphs, clusters):
    """Within vs between split by difficulty tier."""
    print("Running difficulty tier breakdown...", flush=True)
    struct = [g for g in graphs if g.get("_metadata", {}).get("condition") == "structured"]

    results = {}
    for difficulty in ["easy", "moderate", "hard"]:
        within, between = [], []
        for i, j in combinations(range(len(struct)), 2):
            di = struct[i]["_metadata"].get("difficulty")
            dj = struct[j]["_metadata"].get("difficulty")
            if di != difficulty or dj != difficulty:
                continue
            ci_id = struct[i]["_metadata"]["case_id"]
            cj_id = struct[j]["_metadata"]["case_id"]
            ci = get_case_cluster(ci_id, clusters)
            cj = get_case_cluster(cj_id, clusters)
            cls = classify_cluster_pair(ci_id, cj_id, ci, cj)
            if cls:
                sim = composite_similarity(struct[i], struct[j])["composite"]
                (within if cls == "within" else between).append(sim)

        if within and between:
            diff = np.mean(within) - np.mean(between)
            results[difficulty] = {
                "within": float(np.mean(within)),
                "between": float(np.mean(between)),
                "delta": float(diff),
                "n_within": len(within),
                "n_between": len(between),
            }
            print(f"  {difficulty:10s} Δ={diff:+.4f} (n_w={len(within)}, n_b={len(between)})", flush=True)

    return results


# ══════════════════════════════════════════════════════════════
# 6. SEMANTIC MATCHING
# ══════════════════════════════════════════════════════════════

def semantic_matching(graphs, clusters, embeddings_path):
    """Re-run illness script test with embedding-based soft Jaccard."""
    print("Running semantic matching...", flush=True)

    with open(embeddings_path) as f:
        raw_emb = json.load(f)

    emb_keys = list(raw_emb.keys())
    emb_matrix = np.array([raw_emb[k] for k in emb_keys], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-10)
    emb_index = {k: i for i, k in enumerate(emb_keys)}

    def soft_jaccard(ga, gb, threshold=0.85):
        la = [n["label"].strip().lower() for n in ga.get("nodes", []) if n.get("type") == "clinical_feature"]
        lb = [n["label"].strip().lower() for n in gb.get("nodes", []) if n.get("type") == "clinical_feature"]
        if not la or not lb:
            return 0.0
        idx_a = [emb_index[l] for l in la if l in emb_index]
        idx_b = [emb_index[l] for l in lb if l in emb_index]
        if not idx_a or not idx_b:
            return 0.0
        cos_sim = emb_matrix[idx_a] @ emb_matrix[idx_b].T
        matched = int(np.sum(np.max(cos_sim, axis=1) >= threshold))
        total = len(idx_a) + len(idx_b) - matched
        return matched / total if total > 0 else 0.0

    struct = [g for g in graphs if g.get("_metadata", {}).get("condition") == "structured"]
    within, between = [], []
    pairs = list(combinations(range(len(struct)), 2))
    for count, (i, j) in enumerate(pairs):
        if count % 5000 == 0:
            print(f"  {count}/{len(pairs)} pairs...", flush=True)
        ci_id = struct[i]["_metadata"]["case_id"]
        cj_id = struct[j]["_metadata"]["case_id"]
        ci = get_case_cluster(ci_id, clusters)
        cj = get_case_cluster(cj_id, clusters)
        cls = classify_cluster_pair(ci_id, cj_id, ci, cj)
        if cls:
            sim = soft_jaccard(struct[i], struct[j])
            (within if cls == "within" else between).append(sim)

    diff = np.mean(within) - np.mean(between)
    result = {
        "within": float(np.mean(within)),
        "between": float(np.mean(between)),
        "delta": float(diff),
        "n_within": len(within),
        "n_between": len(between),
    }
    print(f"  Soft Jaccard: within={result['within']:.4f} between={result['between']:.4f} Δ={diff:+.4f}", flush=True)
    return result


# ══════════════════════════════════════════════════════════════
# 7. SAVE RAW PAIRWISE SIMILARITIES
# ══════════════════════════════════════════════════════════════

def save_pairwise_arrays(graphs, clusters, output_dir):
    """Save raw within/between similarity arrays for empirical figures."""
    print("Saving pairwise similarity arrays...", flush=True)
    struct = [g for g in graphs if g.get("_metadata", {}).get("condition") == "structured"]

    within, between = [], []
    for i, j in combinations(range(len(struct)), 2):
        ci_id = struct[i]["_metadata"]["case_id"]
        cj_id = struct[j]["_metadata"]["case_id"]
        ci = get_case_cluster(ci_id, clusters)
        cj = get_case_cluster(cj_id, clusters)
        cls = classify_cluster_pair(ci_id, cj_id, ci, cj)
        if cls:
            sim = composite_similarity(struct[i], struct[j])["composite"]
            (within if cls == "within" else between).append(sim)

    np.save(output_dir / "within_sims.npy", np.array(within))
    np.save(output_dir / "between_sims.npy", np.array(between))
    print(f"  Saved within (n={len(within)}) and between (n={len(between)})", flush=True)
    return {"within_n": len(within), "between_n": len(between)}


# ══════════════════════════════════════════════════════════════
# 8. SAME-CASE REPEATED GENERATION (POSITIVE CONTROL)
# ══════════════════════════════════════════════════════════════

def same_case_positive_control(graphs, clusters):
    """Compare same-case cross-model similarity to different-case similarity.
    If models reason about the same case, their graphs should be more similar
    than graphs from different cases — even if not illness-script-consistent.
    This is the positive control showing the metric detects real similarity."""
    print("Running same-case positive control...", flush=True)

    struct = [g for g in graphs if g.get("_metadata", {}).get("condition") == "structured"]

    same_case_sims = []
    diff_case_sims = []

    for i, j in combinations(range(len(struct)), 2):
        ci_id = struct[i]["_metadata"]["case_id"]
        cj_id = struct[j]["_metadata"]["case_id"]
        mi = struct[i]["_metadata"]["source_model"]
        mj = struct[j]["_metadata"]["source_model"]

        if mi == mj:
            continue  # Only compare different models

        sim = composite_similarity(struct[i], struct[j])["composite"]
        if ci_id == cj_id:
            same_case_sims.append(sim)
        else:
            diff_case_sims.append(sim)

    if same_case_sims and diff_case_sims:
        delta = np.mean(same_case_sims) - np.mean(diff_case_sims)
        stat, p = mannwhitneyu(same_case_sims, diff_case_sims, alternative='greater')

        result = {
            "same_case_mean": float(np.mean(same_case_sims)),
            "same_case_n": len(same_case_sims),
            "diff_case_mean": float(np.mean(diff_case_sims)),
            "diff_case_n": len(diff_case_sims),
            "delta": float(delta),
            "p_value": float(p),
            "positive_control_passes": delta > 0 and p < 0.05,
        }
        print(f"  Same case: {result['same_case_mean']:.4f} (n={result['same_case_n']})", flush=True)
        print(f"  Diff case: {result['diff_case_mean']:.4f} (n={result['diff_case_n']})", flush=True)
        print(f"  Δ={result['delta']:+.4f}, p={result['p_value']:.6f}", flush=True)
        print(f"  Positive control {'PASSES' if result['positive_control_passes'] else 'FAILS'}", flush=True)
        return result

    return {"error": "insufficient pairs"}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

ANALYSES = {
    "permutation": ("Case-level permutation tests", lambda args, g, c: permutation_tests(g, c)),
    "ablation": ("Similarity ablation", lambda args, g, c: similarity_ablation(g, c)),
    "noise": ("Noise floor", lambda args, g, c: noise_floor(g)),
    "accuracy": ("Accuracy orthogonality", lambda args, g, c: accuracy_orthogonality(g, c, args.scored)),
    "difficulty": ("Difficulty breakdown", lambda args, g, c: difficulty_breakdown(g, c)),
    "semantic": ("Semantic matching", lambda args, g, c: semantic_matching(g, c, args.embeddings)),
    "pairwise": ("Save pairwise arrays", lambda args, g, c: save_pairwise_arrays(g, c, Path(args.output))),
    "positive_control": ("Same-case positive control", lambda args, g, c: same_case_positive_control(g, c)),
}


def run_all(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs, clusters = load_graphs_and_clusters(args.graphs, args.clusters)

    analyses_to_run = [args.only] if args.only else list(ANALYSES.keys())

    all_results = {}
    for name in analyses_to_run:
        if name not in ANALYSES:
            print(f"Unknown analysis: {name}")
            continue

        label, func = ANALYSES[name]
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")

        # Check required files
        if name == "accuracy" and not args.scored:
            print("  Skipping: --scored not provided")
            continue
        if name == "semantic" and not args.embeddings:
            print("  Skipping: --embeddings not provided")
            continue

        try:
            result = func(args, graphs, clusters)
            all_results[name] = result
            save_json(result, output_dir / f"{name}.json")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save consolidated results
    save_json(all_results, output_dir / "extended_analysis_results.json")
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended analysis for this study")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--scored", default="data/scored_results_public.csv",
                        help="Path to scored results CSV (released: data/scored_results_public.csv)")
    parser.add_argument("--embeddings", default=None, help="Path to label_embeddings.json")
    parser.add_argument("--output", default="results/extended/")
    parser.add_argument("--only", default=None, choices=list(ANALYSES.keys()),
                        help="Run only one analysis")
    args = parser.parse_args()

    run_all(args)
