#!/usr/bin/env python3
"""
crg_analysis.py
===============
Similarity analysis utilities for Clinical Reasoning Graphs.

Provides:

  (1) Component decomposition.
      Decomposes the 5-component similarity into content / structural / full
      composites, recomputes the within- vs between-cluster delta per
      model-condition cell with a CASE-LEVEL permutation test, and adds a
      directional SIGN TEST across the 15 cells. Also reports a minimum
      detectable effect size (MDES) via simulation so the null is correctly
      scoped as "no LARGE effect."

  (2) DIAGNOSIS-SIMILARITY REGRESSION.
      The clean illness-script test: does pairwise graph similarity track
      pairwise gold-diagnosis similarity? Implemented as a Mantel test per
      cell (handles the matrix dependence) plus a pooled case-resampling
      bootstrap slope (handles the pair-level non-independence the naive OLS
      CI ignores).

------------------------------------------------------------------------------
SIMILARITY COMPONENTS
------------------------------------------------------------------------------
The component definitions below (feature/diagnosis/qualifier Jaccard,
motif 1-JSD, reasoning depth) are used for the component decomposition; the
primary weighted composite is imported from src.similarity.composite_similarity.
Everything downstream (composites, permutation test, sign test, Mantel,
bootstrap, MDES) is agnostic to those definitions.

------------------------------------------------------------------------------
EXPECTED INPUT
------------------------------------------------------------------------------
A directory of graph JSONs (one per trace). Field names are configurable in
SCHEMA below. Each graph needs: case_id, model, condition, nodes (type+label),
edges (source_type, edge_type, target_type, phase). For the dx regression you
also need gold-diagnosis text per case (parsed from the graph metadata or a
sidecar) and embeddings for those gold diagnoses.

Run a self-test with synthetic data (no real files needed):
    python -m analysis.crg_analysis --synthetic

Run on your data:
    python -m analysis.crg_analysis \
        --graphs data/extracted/all_graphs.json \
        --clusters data/clusters/clusters.json \
        --dx-embeddings data/gold_dx_embeddings.json \
        --out results/
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ============================================================================
# SCHEMA  -- matches the real all_graphs.json. Identity fields live under
# `_metadata`; edges carry node IDs (source/target) whose types are resolved
# via the node table; the edge relation field is `type`. parse_graph also
# accepts the older flat layout (top-level id fields, edge *_type fields) so
# the synthetic self-test keeps working.
# ============================================================================
SCHEMA = {
    "metadata":  "_metadata",     # identity block in the real graphs
    "case_id":   "case_id",       # unique CPC case identifier (under _metadata)
    "model":     "source_model",  # under _metadata
    "condition": "condition",     # under _metadata: 'baseline'|'adversarial'|'structured'
    "gold_dx":   "ground_truth",  # under _metadata
    "final_dx":  "final_diagnosis",  # top-level
    "nodes":     "nodes",
    "edges":     "edges",
    "reflection_events": "reflection_events",
    # node fields
    "node_id":    "id",
    "node_type":  "type",         # clinical_feature|diagnosis|semantic_qualifier|
                                  # discriminating_feature|evidence_reference
    "node_label": "label",
    # edge fields (edges store node IDs; types resolved via the node table)
    "edge_type":  "type",         # supports|argues_against|discriminates_between|
                                  # triggered_reflection|promoted|demoted|unchanged
    "edge_src":   "source",
    "edge_tgt":   "target",
    "edge_phase": "phase",
}

NODE_TYPES = ["clinical_feature", "diagnosis", "semantic_qualifier",
              "discriminating_feature", "evidence_reference"]
SUPPORT_EDGE = "supports"
DISCRIM_EDGE = "discriminates_between"

# Which components count as "content" vs "structural".
CONTENT_COMPONENTS    = ["feature", "diagnosis", "qualifier", "discriminating"]
STRUCTURAL_COMPONENTS = ["motif", "depth"]
ALL_COMPONENTS        = ["feature", "diagnosis", "qualifier", "discriminating",
                         "motif", "depth"]
# The paper's PRIMARY composite is the unweighted mean of the original 5
# (feature, diagnosis, qualifier, motif, depth). 'discriminating' is added
# here as an extra content channel; include/exclude via --primary-components.
PRIMARY_COMPONENTS = ["feature", "diagnosis", "qualifier", "motif", "depth"]

RNG = np.random.default_rng(20260615)


# ============================================================================
# GRAPH CONTAINER + LOADING
# ============================================================================
@dataclass
class Graph:
    case_id: str
    model: str
    condition: str
    gold_dx: Optional[str]
    final_dx: Optional[str]
    correct: Optional[bool]
    # label sets by node type
    features: set = field(default_factory=set)
    diagnoses: set = field(default_factory=set)
    qualifiers: set = field(default_factory=set)
    discriminating: set = field(default_factory=set)
    evidence: set = field(default_factory=set)
    # edge bookkeeping
    motif_counts: Counter = field(default_factory=Counter)   # (src,etype,tgt) -> n
    phase_counts: Counter = field(default_factory=Counter)    # phase -> n_edges
    n_nodes: int = 0
    n_edges: int = 0
    n_support: int = 0
    n_discrim: int = 0
    raw: Optional[dict] = None       # original graph dict (for the weighted composite)


def _norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def parse_graph(obj: dict) -> Graph:
    """Parse a graph dict into a Graph. Handles the real layout (identity under
    `_metadata`, edges carrying node IDs) and the legacy flat layout used by the
    synthetic fixture (top-level id fields, edges with *_type fields)."""
    s = SCHEMA
    meta = obj.get(s["metadata"], {}) or {}

    def pick(*keys, src=None):
        for k in keys:
            for d in ((src,) if src is not None else (meta, obj)):
                if d and d.get(k) is not None:
                    return d[k]
        return None

    g = Graph(
        case_id=str(pick("case_id")),
        model=str(pick("source_model", "model")),
        condition=str(pick("condition")),
        gold_dx=(_norm_label(pick("ground_truth", "gold_diagnosis"))
                 if pick("ground_truth", "gold_diagnosis") else None),
        final_dx=(_norm_label(obj[s["final_dx"]]) if obj.get(s["final_dx"]) else None),
        correct=obj.get("correct"),
        raw=obj,
    )
    bucket = {
        "clinical_feature": g.features,
        "diagnosis": g.diagnoses,
        "semantic_qualifier": g.qualifiers,
        "discriminating_feature": g.discriminating,
        "evidence_reference": g.evidence,
    }
    # node table: id -> type (reflection events are pseudo-nodes, as in src.extract_motifs)
    id2type: dict = {}
    for n in obj.get(s["nodes"], []):
        ntype = n.get(s["node_type"])
        id2type[n.get(s["node_id"])] = ntype
        lbl = _norm_label(n.get(s["node_label"], ""))
        if ntype in bucket and lbl:
            bucket[ntype].add(lbl)
    for r in obj.get(s["reflection_events"], []):
        id2type[r.get("id")] = "reflection_event"

    g.n_nodes = len(obj.get(s["nodes"], []))
    for e in obj.get(s["edges"], []):
        et = e.get(s["edge_type"], e.get("edge_type"))
        # source/target type: prefer explicit *_type (flat/synthetic), else resolve node ID
        if "source_type" in e:
            src = e.get("source_type", "?")
        else:
            src = id2type.get(e.get(s["edge_src"]), "?")
        if "target_type" in e:
            tgt = e.get("target_type", "?")
        else:
            tv = e.get(s["edge_tgt"])
            if isinstance(tv, list):           # discriminates_between -> [dxA, dxB]
                tv = tv[0] if tv else None
            tgt = id2type.get(tv, "?")
        phase = e.get(s["edge_phase"], "p1")
        g.motif_counts[(src, et, tgt)] += 1
        g.phase_counts[str(phase)] += 1
        g.n_edges += 1
        if et == SUPPORT_EDGE:
            g.n_support += 1
        if et == DISCRIM_EDGE:
            g.n_discrim += 1
    return g


def load_graphs(graph_path: str) -> list[Graph]:
    """Load graphs from a single JSON file (a list, e.g. all_graphs.json) or a
    directory of per-trace JSONs. Skips extraction-error rows and non-graph files."""
    p = Path(graph_path)
    if p.is_dir():
        objs = []
        for fp in sorted(p.glob("*.json")):
            with open(fp) as f:
                objs.append(json.load(f))
        if not objs:
            raise FileNotFoundError(f"No *.json found in {graph_path}")
    else:
        with open(p) as f:
            data = json.load(f)
        objs = data if isinstance(data, list) else [data]
    objs = [o for o in objs if isinstance(o, dict) and "_error" not in o and "nodes" in o]
    if not objs:
        raise ValueError(f"No valid graphs in {graph_path}")
    return [parse_graph(o) for o in objs]


# ============================================================================
# === SIMILARITY COMPONENTS ===   (reconstructed -- replace with your canon)
# ============================================================================
def jaccard(a: set, b: set) -> float:
    """Both-empty -> 0.0 (no shared content). Document if you prefer 1.0."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def soft_jaccard(a: set, b: set, emb: dict[str, np.ndarray], thr: float) -> float:
    """Greedy embedding match: labels match if cosine >= thr."""
    if not a and not b:
        return 0.0
    A, B = list(a), list(b)
    Av = np.array([emb[x] for x in A]) if A else np.zeros((0, 1))
    Bv = np.array([emb[x] for x in B]) if B else np.zeros((0, 1))
    if len(A) == 0 or len(B) == 0:
        return 0.0
    Av = Av / (np.linalg.norm(Av, axis=1, keepdims=True) + 1e-9)
    Bv = Bv / (np.linalg.norm(Bv, axis=1, keepdims=True) + 1e-9)
    sim = Av @ Bv.T
    matches, used_b = 0, set()
    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    used_a = set()
    for i, j in order:
        if sim[i, j] < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); matches += 1
    return matches / (len(A) + len(B) - matches)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2 -> [0,1]."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / (y[mask] + 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def motif_similarity(gi: Graph, gj: Graph) -> float:
    keys = sorted(set(gi.motif_counts) | set(gj.motif_counts))
    if not keys:
        return 0.0
    p = np.array([gi.motif_counts.get(k, 0) for k in keys], float)
    q = np.array([gj.motif_counts.get(k, 0) for k in keys], float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    return 1.0 - _jsd(p, q)


def make_depth_similarity(graphs: list[Graph] | None = None) -> Callable[[Graph, Graph], float]:
    """Reasoning-depth similarity, delegating to the canonical
    src.similarity.depth_similarity: 1 - |discriminating-edge-ratio diff|, where the
    ratio is discriminates_between edges / total edges. The ``graphs`` argument is
    accepted for call-site compatibility but unused."""
    from src.similarity import depth_similarity as _depth_similarity

    def depth_sim(gi: Graph, gj: Graph) -> float:
        return _depth_similarity(gi.raw, gj.raw)
    return depth_sim


def make_components_fn(graphs: list[Graph],
                       feat_emb: Optional[dict] = None,
                       soft_thr: float = 0.85) -> Callable[[Graph, Graph], dict]:
    depth_sim = make_depth_similarity(graphs)

    def components(gi: Graph, gj: Graph) -> dict[str, float]:
        if feat_emb is not None:
            feat = soft_jaccard(gi.features, gj.features, feat_emb, soft_thr)
            diag = soft_jaccard(gi.diagnoses, gj.diagnoses, feat_emb, soft_thr)
        else:
            feat = jaccard(gi.features, gj.features)
            diag = jaccard(gi.diagnoses, gj.diagnoses)
        return {
            "feature": feat,
            "diagnosis": diag,
            "qualifier": jaccard(gi.qualifiers, gj.qualifiers),
            "discriminating": jaccard(gi.discriminating, gj.discriminating),
            "motif": motif_similarity(gi, gj),
            "depth": depth_sim(gi, gj),
        }
    return components


def composite(comp: dict[str, float], which: list[str]) -> float:
    return float(np.mean([comp[c] for c in which]))


# ============================================================================
# SIMILARITY MATRICES  (per model-condition cell, indexed by case)
# ============================================================================
@dataclass
class Cell:
    model: str
    condition: str
    cases: list[str]                       # ordered case ids
    comp_mats: dict[str, np.ndarray]       # component name -> NxN matrix
    graphs: list = field(default_factory=list)  # ordered Graph objects (carry .raw)


def build_cells(graphs: list[Graph],
                components_fn: Callable[[Graph, Graph], dict]) -> list[Cell]:
    by_cell: dict[tuple, dict[str, Graph]] = defaultdict(dict)
    for g in graphs:
        by_cell[(g.model, g.condition)][g.case_id] = g
    cells = []
    for (model, cond), case_map in sorted(by_cell.items()):
        cases = sorted(case_map)
        N = len(cases)
        mats = {c: np.full((N, N), np.nan) for c in ALL_COMPONENTS}
        for a in range(N):
            for b in range(a + 1, N):
                comp = components_fn(case_map[cases[a]], case_map[cases[b]])
                for c in ALL_COMPONENTS:
                    mats[c][a, b] = mats[c][b, a] = comp[c]
        cells.append(Cell(model, cond, cases, mats, [case_map[c] for c in cases]))
    return cells


def composite_matrix(cell: Cell, which: list[str]) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean([cell.comp_mats[c] for c in which], axis=0)


# --- Weighted composite: the repo's canonical composite, imported from src.similarity ---
_CANON_KEYMAP = {"feature": "feature_overlap", "diagnosis": "diagnosis_overlap",
                 "qualifier": "qualifier_overlap", "motif": "motif_similarity",
                 "depth": "depth_similarity", "composite": "composite"}


def canonical_matrices(cell: Cell) -> dict[str, np.ndarray]:
    """Per-pair canonical components + weighted composite straight from
    src.similarity.composite_similarity (weights 0.30/0.15/0.35/0.10/0.10,
    Jaccard both-empty=1.0, depth = 1-|disc/total diff|). No reimplementation."""
    from src.similarity import composite_similarity
    N = len(cell.cases)
    M = {k: np.full((N, N), np.nan) for k in _CANON_KEYMAP}
    for a in range(N):
        for b in range(a + 1, N):
            s = composite_similarity(cell.graphs[a].raw, cell.graphs[b].raw)
            for k, sk in _CANON_KEYMAP.items():
                M[k][a, b] = M[k][b, a] = s[sk]
    return M


# ============================================================================
# (1) WITHIN vs BETWEEN  +  PERMUTATION TEST  +  SIGN TEST
# ============================================================================
def within_between(sim: np.ndarray, cases: list[str],
                   cluster_of: dict[str, int]) -> tuple[float, int, int]:
    """Returns (delta, n_within, n_between). Singleton/uncluster -> between."""
    N = len(cases)
    cl = np.array([cluster_of.get(c, -(i + 1)) for i, c in enumerate(cases)])
    sizes = Counter(cl[cl >= 0])
    w, b = [], []
    for a in range(N):
        for c in range(a + 1, N):
            same = (cl[a] >= 0 and cl[a] == cl[c] and sizes[cl[a]] >= 2)
            (w if same else b).append(sim[a, c])
    if not w or not b:
        return float("nan"), len(w), len(b)
    return float(np.mean(w) - np.mean(b)), len(w), len(b)


def perm_test_delta(sim: np.ndarray, cases: list[str],
                    cluster_of: dict[str, int], n_perm: int = 10000
                    ) -> tuple[float, float, int, int]:
    """Case-level permutation: shuffle cluster labels across cases (preserves
    cluster-size multiset), recompute delta. Two-sided p."""
    obs, nw, nb = within_between(sim, cases, cluster_of)
    if math.isnan(obs):
        return obs, float("nan"), nw, nb
    labels = np.array([cluster_of.get(c, -(i + 1)) for i, c in enumerate(cases)])
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = RNG.permutation(labels)
        cmap = {cases[i]: int(perm[i]) for i in range(len(cases))}
        d, _, _ = within_between(sim, cases, cmap)
        null[k] = d
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return obs, float(p), nw, nb


def sign_test(deltas: list[float]) -> dict:
    d = [x for x in deltas if not math.isnan(x)]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    # two-sided exact binomial vs p=0.5
    from math import comb
    k = max(pos, n - pos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {"n": n, "positive": pos, "negative": n - pos, "p_two_sided": p}


def mdes_simulation(cells: list[Cell], which: list[str],
                    clusters: dict[str, int],
                    deltas_grid=np.linspace(0.0, 0.08, 17),
                    alpha: float = 0.0033, n_perm: int = 2000,
                    n_sim: int = 200, target_power: float = 0.80) -> dict:
    """Inject a known additive delta into within-cluster similarities and
    estimate power of the per-cell permutation test at alpha. Returns the
    smallest delta reaching target_power (the MDES). Uses one representative
    cell to keep runtime bounded; report as order-of-magnitude."""
    cell = max(cells, key=lambda c: len(c.cases))   # most cases = most power
    base = composite_matrix(cell, which)
    cases = cell.cases
    N = len(cases)
    cl = np.array([clusters.get(c, -(i + 1)) for i, c in enumerate(cases)])
    sizes = Counter(cl[cl >= 0])
    within_mask = np.zeros((N, N), bool)
    for a in range(N):
        for c in range(a + 1, N):
            if cl[a] >= 0 and cl[a] == cl[c] and sizes[cl[a]] >= 2:
                within_mask[a, c] = within_mask[c, a] = True
    curve = {}
    mdes = None
    for delta in deltas_grid:
        hits = 0
        for _ in range(n_sim):
            sim = base.copy()
            sim[within_mask] = np.clip(sim[within_mask] + delta, 0, 1)
            _, p, _, _ = perm_test_delta(sim, cases, clusters, n_perm=n_perm)
            if p < alpha:
                hits += 1
        power = hits / n_sim
        curve[round(float(delta), 4)] = round(power, 3)
        if mdes is None and power >= target_power:
            mdes = float(delta)
    return {"representative_cell": f"{cell.model}/{cell.condition}",
            "alpha": alpha, "target_power": target_power,
            "mdes": mdes, "power_curve": curve}


# ============================================================================
# (2) DIAGNOSIS-SIMILARITY REGRESSION  (Mantel + case bootstrap)
# ============================================================================
def cosine_matrix(cases: list[str], emb: dict[str, np.ndarray]) -> np.ndarray:
    M = np.array([emb[c] for c in cases])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    np.fill_diagonal(S, np.nan)
    return S


def _offdiag(m: np.ndarray) -> np.ndarray:
    N = m.shape[0]
    iu = np.triu_indices(N, k=1)
    return m[iu]


def mantel(graph_sim: np.ndarray, dx_sim: np.ndarray, n_perm: int = 10000
           ) -> tuple[float, float]:
    """Mantel r (Pearson on off-diagonal) + permutation p (permute case labels
    of dx matrix). Two-sided."""
    x = _offdiag(graph_sim); y = _offdiag(dx_sim)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan"), float("nan")
    r_obs = float(np.corrcoef(x, y)[0, 1])
    N = graph_sim.shape[0]
    null = np.empty(n_perm)
    iu = np.triu_indices(N, k=1)
    for k in range(n_perm):
        perm = RNG.permutation(N)
        dperm = dx_sim[np.ix_(perm, perm)]
        yp = dperm[iu][ok]
        null[k] = np.corrcoef(x, yp)[0, 1]
    p = (np.sum(np.abs(null) >= abs(r_obs) - 1e-12) + 1) / (n_perm + 1)
    return r_obs, float(p)


def pooled_bootstrap_slope(cells: list[Cell], which: list[str],
                           dx_emb: dict[str, np.ndarray],
                           n_boot: int = 2000) -> dict:
    """Pool all same-cell case-pairs; regress graph_sim ~ dx_sim. CI by
    resampling CASES (not pairs), which respects pair-level dependence."""
    # universe of cases (union across cells; dx_sim is case-level, model-invariant)
    all_cases = sorted({c for cell in cells for c in cell.cases if c in dx_emb})
    if len(all_cases) < 5:
        return {"error": "too few cases with dx embeddings"}
    idx = {c: i for i, c in enumerate(all_cases)}
    dx_full = cosine_matrix(all_cases, dx_emb)

    def slope_for(case_subset_idx: np.ndarray) -> float:
        xs, ys = [], []
        sub = set(int(i) for i in case_subset_idx)
        for cell in cells:
            cmat = composite_matrix(cell, which)
            ci = [idx[c] for c in cell.cases if c in idx]
            local = [k for k, c in enumerate(cell.cases) if c in idx]
            for a in range(len(local)):
                for b in range(a + 1, len(local)):
                    ga, gb = ci[a], ci[b]
                    if ga in sub and gb in sub:
                        xs.append(dx_full[ga, gb])
                        ys.append(cmat[local[a], local[b]])
        xs, ys = np.array(xs), np.array(ys)
        ok = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 3 or np.std(xs) < 1e-9:
            return float("nan")
        return float(np.polyfit(xs, ys, 1)[0])

    point = slope_for(np.arange(len(all_cases)))
    boots = []
    n = len(all_cases)
    for _ in range(n_boot):
        samp = RNG.integers(0, n, n)            # resample cases w/ replacement
        s = slope_for(samp)
        if not math.isnan(s):
            boots.append(s)
    boots = np.array(boots)
    return {"slope": point,
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
            "n_case_pairs_universe": int(n * (n - 1) / 2)}


# ============================================================================
# CLUSTERS
# ============================================================================
def load_clusters(path: str) -> dict[str, int]:
    """Return {case_id: cluster_int}. Accepts three layouts:
      - the real clusters.json: {cluster_id: {"case_ids": [...], ...}}
      - {case_id: cluster_int_or_label}
      - {cluster_label: [case_ids]}
    """
    with open(path) as f:
        obj = json.load(f)
    vals = list(obj.values())
    # Real layout: each value is a cluster record carrying case_ids.
    if vals and all(isinstance(v, dict) and "case_ids" in v for v in vals):
        out = {}
        for i, (cid, info) in enumerate(obj.items()):
            for c in info.get("case_ids", []):
                out[str(c)] = i
        return out
    if all(isinstance(v, (int, str)) for v in vals):
        return {str(k): int(v) if str(v).lstrip("-").isdigit() else hash(v) & 0xFFFF
                for k, v in obj.items()}
    out = {}
    for i, (lbl, cases) in enumerate(obj.items()):
        for c in cases:
            out[str(c)] = i
    return out


# ============================================================================
# REPORTING
# ============================================================================
def run_within_between(cells, clusters, which, label, n_perm):
    rows, deltas = [], []
    for cell in cells:
        sim = composite_matrix(cell, which)
        obs, p, nw, nb = perm_test_delta(sim, cell.cases, clusters, n_perm=n_perm)
        rows.append({"model": cell.model, "condition": cell.condition,
                     "delta": round(obs, 4) if not math.isnan(obs) else None,
                     "p": round(p, 4) if not math.isnan(p) else None,
                     "n_within": nw, "n_between": nb})
        deltas.append(obs)
    st = sign_test(deltas)
    return {"composite": label, "components": which,
            "cells": rows, "sign_test": st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs")
    ap.add_argument("--clusters")
    ap.add_argument("--dx-embeddings",
                    help="JSON {case_id: [floats]} gold-dx embeddings")
    ap.add_argument("--feat-embeddings",
                    help="JSON {label: [floats]} for soft feature/dx matching")
    ap.add_argument("--soft-threshold", type=float, default=0.85)
    ap.add_argument("--primary-components", nargs="+", default=PRIMARY_COMPONENTS)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-mdes", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true",
                    help="run self-test on generated data")
    args = ap.parse_args()

    if args.synthetic:
        graphs, clusters, dx_emb = make_synthetic()
        feat_emb = None
    else:
        if not (args.graphs and args.clusters):
            ap.error("--graphs and --clusters required (or use --synthetic)")
        graphs = load_graphs(args.graphs)
        clusters = load_clusters(args.clusters)
        dx_emb = None
        if args.dx_embeddings:
            with open(args.dx_embeddings) as f:
                raw = json.load(f)
            dx_emb = {str(k): np.array(v, float) for k, v in raw.items()}
        feat_emb = None
        if args.feat_embeddings:
            with open(args.feat_embeddings) as f:
                raw = json.load(f)
            feat_emb = {_norm_label(k): np.array(v, float) for k, v in raw.items()}

    os.makedirs(args.out, exist_ok=True)
    print(f"Loaded {len(graphs)} graphs, "
          f"{len({g.case_id for g in graphs})} cases, "
          f"{len({(g.model, g.condition) for g in graphs})} cells.")

    components_fn = make_components_fn(graphs, feat_emb, args.soft_threshold)
    cells = build_cells(graphs, components_fn)

    results = {"deliverable_1_content_decomposition": {}, "deliverable_2_dx_regression": {}}

    # ---- (1) composite decomposition + permutation + sign test ----
    for label, which in [
        ("primary",      args.primary_components),
        ("content_only", CONTENT_COMPONENTS),
        ("structural",   STRUCTURAL_COMPONENTS),
        ("full_six",     ALL_COMPONENTS),
    ]:
        res = run_within_between(cells, clusters, which, label, args.n_perm)
        results["deliverable_1_content_decomposition"][label] = res
        st = res["sign_test"]
        print(f"\n[{label}] sign test: {st['positive']}/{st['n']} positive, "
              f"p={st['p_two_sided']:.4f}")

    if not args.skip_mdes:
        print("\nEstimating MDES (simulation)...")
        results["deliverable_1_content_decomposition"]["mdes"] = mdes_simulation(
            cells, args.primary_components, clusters)
        print("MDES:", results["deliverable_1_content_decomposition"]["mdes"]["mdes"])

    # ---- (2) diagnosis-similarity regression ----
    if dx_emb is not None:
        mant = []
        for cell in cells:
            cases_with = [c for c in cell.cases if c in dx_emb]
            if len(cases_with) < 3:
                continue
            keep = [i for i, c in enumerate(cell.cases) if c in dx_emb]
            gmat = composite_matrix(cell, args.primary_components)[np.ix_(keep, keep)]
            dmat = cosine_matrix(cases_with, dx_emb)
            r, p = mantel(gmat, dmat, n_perm=args.n_perm)
            mant.append({"model": cell.model, "condition": cell.condition,
                         "mantel_r": round(r, 4) if not math.isnan(r) else None,
                         "p": round(p, 4) if not math.isnan(p) else None})
        boot = pooled_bootstrap_slope(cells, args.primary_components, dx_emb, args.n_boot)
        results["deliverable_2_dx_regression"] = {"mantel_per_cell": mant,
                                                   "pooled_slope_bootstrap": boot}
        print("\n[dx regression] pooled slope:", boot.get("slope"),
              "CI95:", boot.get("ci95"))
        rs = [m["mantel_r"] for m in mant if m["mantel_r"] is not None]
        if rs:
            print(f"[dx regression] mean Mantel r = {np.mean(rs):.4f} "
                  f"({sum(1 for r in rs if r > 0)}/{len(rs)} positive)")
    else:
        results["deliverable_2_dx_regression"] = {
            "skipped": "no --dx-embeddings provided"}
        print("\n[dx regression] skipped (provide --dx-embeddings)")

    out_path = Path(args.out) / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


# ============================================================================
# SYNTHETIC FIXTURE (self-test only)
# ============================================================================
def make_synthetic(n_cases=30, n_clusters=10, models=("m1", "m2", "m3"),
                   conditions=("baseline", "adversarial", "structured")):
    """Generate graphs with a small, tunable within-cluster content signal so
    the harness can be exercised end to end. NOT real data."""
    rng = np.random.default_rng(7)
    cases = [f"case{i:02d}" for i in range(n_cases)]
    clusters = {c: int(rng.integers(0, n_clusters)) for c in cases}
    # 5 singletons
    for c in cases[:5]:
        clusters[c] = 100 + cases.index(c)
    # per-cluster shared feature/dx vocab (the injected signal)
    cluster_feats = {k: {f"feat_{k}_{j}" for j in range(6)} for k in set(clusters.values())}
    cluster_dx = {k: {f"dx_{k}_{j}" for j in range(3)} for k in set(clusters.values())}
    # gold dx embeddings: cluster-structured so dx regression has something to find
    dim = 16
    cluster_vec = {k: rng.normal(size=dim) for k in set(clusters.values())}
    dx_emb = {c: cluster_vec[clusters[c]] + 0.5 * rng.normal(size=dim) for c in cases}

    graphs = []
    for m in models:
        for cond in conditions:
            for c in cases:
                k = clusters[c]
                # 70% of features drawn from cluster vocab, 30% case-idiosyncratic
                feats = set(rng.choice(list(cluster_feats[k]),
                                       size=min(4, len(cluster_feats[k])), replace=False))
                feats |= {f"u_{c}_{m}_{j}" for j in range(3)}
                dxs = set(rng.choice(list(cluster_dx[k]),
                                     size=min(2, len(cluster_dx[k])), replace=False))
                nodes = ([{"type": "clinical_feature", "label": x} for x in feats]
                         + [{"type": "diagnosis", "label": x} for x in dxs]
                         + [{"type": "semantic_qualifier", "label": f"q_{rng.integers(0,4)}"}]
                         + [{"type": "discriminating_feature", "label": f"disc_{c}_{rng.integers(0,3)}"}])
                edges = []
                dlist, flist = list(dxs), list(feats)
                for _ in range(max(8, len(nodes))):
                    edges.append({"source_type": "clinical_feature",
                                  "edge_type": rng.choice(["supports", "argues_against",
                                                           "discriminates_between"]),
                                  "target_type": "diagnosis",
                                  "phase": rng.choice(["p1", "p2"])})
                obj = {"case_id": c, "model": m, "condition": cond,
                       "gold_diagnosis": f"gold_{clusters[c]}",
                       "final_diagnosis": list(dxs)[0] if dxs else "none",
                       "correct": bool(rng.random() > 0.4),
                       "nodes": nodes, "edges": edges}
                graphs.append(parse_graph(obj))
    return graphs, clusters, dx_emb


if __name__ == "__main__":
    main()
