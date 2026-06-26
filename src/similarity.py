from __future__ import annotations
"""
Graph similarity metrics for comparing clinical reasoning graphs.

Implements multiple complementary similarity measures:
1. Feature overlap (Jaccard on node labels)
2. Diagnostic overlap (Jaccard on diagnosis nodes)
3. Edge motif similarity (Jensen-Shannon divergence on motif distributions)
4. Structural similarity (normalized graph edit distance approximation)
5. Reasoning depth metrics (discriminating edge ratio, reflection density)
"""

import numpy as np
from collections import Counter
from itertools import combinations
from typing import Any


# ──────────────────────────────────────────────
# NODE-LEVEL SIMILARITY
# ──────────────────────────────────────────────

def get_nodes_by_type(graph: dict, node_type: str) -> set[str]:
    """Get set of node labels of a given type."""
    return {
        n["label"].lower().strip()
        for n in graph.get("nodes", [])
        if n.get("type") == node_type
    }


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def classify_cluster_pair(case_i, case_j, cluster_i, cluster_j) -> str | None:
    """Canonical classification of a graph pair for the within-vs-between
    consistency contrast. Returns 'within', 'between', or None:

      - 'within'  : same cluster AND distinct cases (cross-case schema reuse)
      - 'between' : different clusters (necessarily distinct cases)
      - None      : a case unassigned to a cluster, OR a same-case pair.

    Same-case pairs (the identical case extracted by two models, or twice) are
    EXCLUDED from the contrast: a case paired with itself is not a clinically
    *similar* case, so such pairs measure inter-model / inter-extraction
    agreement rather than cross-case schema reuse. They are analyzed separately
    as a positive control. This is the single source of truth for the
    within-cluster definition; all pooling code must route through it.
    """
    if cluster_i is None or cluster_j is None:
        return None
    if case_i == case_j:
        return None
    return "within" if cluster_i == cluster_j else "between"


def feature_overlap(graph_a: dict, graph_b: dict) -> float:
    """Jaccard similarity of clinical feature nodes."""
    feats_a = get_nodes_by_type(graph_a, "clinical_feature")
    feats_b = get_nodes_by_type(graph_b, "clinical_feature")
    return jaccard(feats_a, feats_b)


def diagnosis_overlap(graph_a: dict, graph_b: dict) -> float:
    """Jaccard similarity of diagnosis nodes."""
    dx_a = get_nodes_by_type(graph_a, "diagnosis")
    dx_b = get_nodes_by_type(graph_b, "diagnosis")
    return jaccard(dx_a, dx_b)


def semantic_qualifier_overlap(graph_a: dict, graph_b: dict) -> float:
    """Jaccard similarity of semantic qualifier nodes."""
    sq_a = get_nodes_by_type(graph_a, "semantic_qualifier")
    sq_b = get_nodes_by_type(graph_b, "semantic_qualifier")
    return jaccard(sq_a, sq_b)


# ──────────────────────────────────────────────
# MOTIF-LEVEL SIMILARITY
# ──────────────────────────────────────────────

def extract_motifs(graph: dict) -> Counter:
    """Extract reasoning motifs (typed edge patterns) from a graph.
    
    A motif is a triple: (source_node_type, edge_type, target_node_type).
    This abstracts away the specific entities and captures the reasoning
    *pattern* — e.g., "clinical_feature → supports → diagnosis" regardless
    of which specific feature or diagnosis.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    # Include reflection events as pseudo-nodes
    for r in graph.get("reflection_events", []):
        nodes[r["id"]] = {"id": r["id"], "type": "reflection_event", "label": r.get("trigger_description", "")}

    motifs = Counter()
    for edge in graph.get("edges", []):
        source = nodes.get(edge.get("source"), {})
        target_val = edge.get("target")
        if isinstance(target_val, list):
            target = nodes.get(target_val[0], {})
        else:
            target = nodes.get(target_val, {})
        motif = (
            source.get("type", "unknown"),
            edge.get("type", "unknown"),
            target.get("type", "unknown"),
        )
        motifs[motif] += 1

    return motifs


def extract_phase_motifs(graph: dict) -> dict[str, Counter]:
    """Extract motifs grouped by reasoning phase."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    for r in graph.get("reflection_events", []):
        nodes[r["id"]] = {"id": r["id"], "type": "reflection_event", "label": ""}

    phase_motifs = {"phase1": Counter(), "phase2": Counter(), "phase3": Counter()}
    for edge in graph.get("edges", []):
        phase = edge.get("phase", "phase1")
        source = nodes.get(edge.get("source"), {})
        target_val = edge.get("target")
        if isinstance(target_val, list):
            target = nodes.get(target_val[0], {})
        else:
            target = nodes.get(target_val, {})
        motif = (source.get("type", "unknown"), edge.get("type", "unknown"), target.get("type", "unknown"))
        if phase in phase_motifs:
            phase_motifs[phase][motif] += 1

    return phase_motifs


def motif_similarity(graph_a: dict, graph_b: dict) -> float:
    """Jensen-Shannon divergence-based similarity of motif distributions.
    
    Returns 1.0 for identical distributions, 0.0 for maximally different.
    """
    motifs_a = extract_motifs(graph_a)
    motifs_b = extract_motifs(graph_b)

    all_motifs = set(motifs_a.keys()) | set(motifs_b.keys())
    if not all_motifs:
        return 1.0

    # Build probability distributions
    total_a = sum(motifs_a.values()) or 1
    total_b = sum(motifs_b.values()) or 1

    p = np.array([motifs_a.get(m, 0) / total_a for m in all_motifs])
    q = np.array([motifs_b.get(m, 0) / total_b for m in all_motifs])

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    jsd = 0.5 * (kl_pm + kl_qm)

    # Convert to similarity (0 = identical, log(2) = maximally different)
    return 1.0 - (jsd / np.log(2))


# ──────────────────────────────────────────────
# STRUCTURAL METRICS
# ──────────────────────────────────────────────

def graph_density(graph: dict) -> float:
    """Edge count / node count ratio."""
    n_nodes = len(graph.get("nodes", []))
    n_edges = len(graph.get("edges", []))
    return n_edges / max(n_nodes, 1)


def discriminating_edge_ratio(graph: dict) -> float:
    """Fraction of edges that are discriminating (vs. simple supports/argues_against)."""
    edges = graph.get("edges", [])
    if not edges:
        return 0.0
    disc = sum(1 for e in edges if e.get("type") == "discriminates_between")
    return disc / len(edges)


def reflection_density(graph: dict) -> float:
    """Number of reflection events per diagnosis node."""
    n_reflections = len(graph.get("reflection_events", []))
    n_diagnoses = len([n for n in graph.get("nodes", []) if n.get("type") == "diagnosis"])
    return n_reflections / max(n_diagnoses, 1)


def depth_similarity(graph_a: dict, graph_b: dict) -> float:
    """Pairwise reasoning-depth similarity: 1 - |Δ ratio| on the discriminating-edge
    ratio (discriminates_between edges / total edges). SINGLE SOURCE OF TRUTH —
    used both as the composite's depth component and as the Table 4 depth row, so
    the displayed depth never diverges from the composite's depth input.
    """
    return 1.0 - abs(discriminating_edge_ratio(graph_a) - discriminating_edge_ratio(graph_b))


def reasoning_depth(graph: dict) -> dict:
    """Composite measure of reasoning depth."""
    return {
        "density": graph_density(graph),
        "discriminating_ratio": discriminating_edge_ratio(graph),
        "reflection_density": reflection_density(graph),
        "n_nodes": len(graph.get("nodes", [])),
        "n_edges": len(graph.get("edges", [])),
        "n_phases": len(set(e.get("phase") for e in graph.get("edges", []))),
        "has_problem_representation": bool(graph.get("problem_representation")),
    }


# ──────────────────────────────────────────────
# COMPOSITE SIMILARITY
# ──────────────────────────────────────────────

def composite_similarity(
    graph_a: dict,
    graph_b: dict,
    weights: dict | None = None,
) -> dict:
    """Compute all similarity metrics between two graphs.
    
    Returns individual metrics plus a weighted composite score.
    """
    if weights is None:
        weights = {
            "diagnosis_overlap": 0.30,
            "feature_overlap": 0.15,
            "motif_similarity": 0.35,
            "qualifier_overlap": 0.10,
            "depth_similarity": 0.10,
        }

    dx_sim = diagnosis_overlap(graph_a, graph_b)
    feat_sim = feature_overlap(graph_a, graph_b)
    motif_sim = motif_similarity(graph_a, graph_b)
    qual_sim = semantic_qualifier_overlap(graph_a, graph_b)

    # Depth similarity: canonical depth_similarity (same function as the Table 4 row)
    depth_sim = depth_similarity(graph_a, graph_b)

    composite = (
        weights["diagnosis_overlap"] * dx_sim
        + weights["feature_overlap"] * feat_sim
        + weights["motif_similarity"] * motif_sim
        + weights["qualifier_overlap"] * qual_sim
        + weights["depth_similarity"] * depth_sim
    )

    return {
        "diagnosis_overlap": dx_sim,
        "feature_overlap": feat_sim,
        "motif_similarity": motif_sim,
        "qualifier_overlap": qual_sim,
        "depth_similarity": depth_sim,
        "composite": composite,
    }


def pairwise_similarity_matrix(
    graphs: list[dict],
    metric: str = "composite",
) -> np.ndarray:
    """Compute pairwise similarity matrix for a list of graphs."""
    n = len(graphs)
    matrix = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        sims = composite_similarity(graphs[i], graphs[j])
        val = sims[metric] if metric in sims else sims["composite"]
        matrix[i, j] = val
        matrix[j, i] = val

    np.fill_diagonal(matrix, 1.0)
    return matrix
