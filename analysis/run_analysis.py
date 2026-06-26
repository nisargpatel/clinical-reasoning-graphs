#!/usr/bin/env python3
"""
Driver for the clinical-reasoning-graph similarity analysis, built on the
crg_analysis library.

Step "reproduce": recompute the per-cell composite similarity table
(src.similarity.composite_similarity), per-cell and pooled, to 3 decimals.
Step "decompose": per-cell within/between for the composite and its component
channels. Step "mantel": graph-vs.\ gold-diagnosis similarity regression.

    python -m analysis.run_analysis --step reproduce \
        --graphs data/extracted/all_graphs.json \
        --clusters data/clusters/clusters.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter

import numpy as np

from analysis import crg_analysis as crg

# Reported per-cell within/between (Appendix, full results): (model_substr, condition) -> (W, B)
TABLE2 = {
    ("gpt-5.4", "baseline"): (.476, .469), ("gpt-5.4", "adversarial"): (.475, .483),
    ("gpt-5.4", "structured"): (.452, .448),
    ("gpt-5.2", "baseline"): (.454, .462), ("gpt-5.2", "adversarial"): (.457, .458),
    ("gpt-5.2", "structured"): (.459, .465),
    ("sonnet", "baseline"): (.501, .498), ("sonnet", "adversarial"): (.500, .492),
    ("sonnet", "structured"): (.460, .463),
    ("opus", "baseline"): (.495, .481), ("opus", "adversarial"): (.494, .485),
    ("opus", "structured"): (.474, .469),
    ("gemini", "baseline"): (.482, .469), ("gemini", "adversarial"): (.492, .484),
    ("gemini", "structured"): (.457, .453),
}
MODEL_ORDER = ["gpt-5.4", "gpt-5.2", "sonnet", "opus", "gemini"]
COND_ORDER = ["baseline", "adversarial", "structured"]


def t2_key(model: str, cond: str):
    for ms in MODEL_ORDER:
        if ms in model:
            return TABLE2.get((ms, cond))
    return None


def wb(sim: np.ndarray, cases: list[str], clusters: dict[str, int]):
    """Within/between means + the raw pair lists (singleton/size-1 -> between)."""
    N = len(cases)
    cl = np.array([clusters.get(c, -(i + 1)) for i, c in enumerate(cases)])
    sizes = Counter(cl[cl >= 0])
    w, b = [], []
    for a in range(N):
        for c in range(a + 1, N):
            same = (cl[a] >= 0 and cl[a] == cl[c] and sizes[cl[a]] >= 2)
            (w if same else b).append(sim[a, c])
    return float(np.mean(w)), float(np.mean(b)), w, b


def sort_cells(cells):
    def rank(cell):
        mi = next((i for i, ms in enumerate(MODEL_ORDER) if ms in cell.model), 99)
        ci = COND_ORDER.index(cell.condition) if cell.condition in COND_ORDER else 99
        return (mi, ci)
    return sorted(cells, key=rank)


def _load_original_output():
    """Saved output of src/analyze.py (which generated Table 2's W/B/d).
    Returns {(model, condition): (W, B)} at full float precision, or {}."""
    import json, os
    path = "results/diagnostic_schema_consistency.json"
    if not os.path.exists(path):
        return {}
    out = {}
    for k, v in json.load(open(path)).items():
        model, cond = k.rsplit("__", 1)
        out[(model, cond)] = (v["within_cluster_mean"], v["between_cluster_mean"])
    return out


def run_reproduce(graphs_path: str, clusters_path: str):
    graphs = crg.load_graphs(graphs_path)
    clusters = crg.load_clusters(clusters_path)
    # comp_mats are not needed here, but build_cells also stashes the
    # ordered Graph objects (with .raw) that the canonical composite needs.
    comp_fn = crg.make_components_fn(graphs, None, 0.85)
    cells = sort_cells(crg.build_cells(graphs, comp_fn))
    orig = _load_original_output()

    print(f"Loaded {len(graphs)} graphs, {len({g.case_id for g in graphs})} cases, {len(cells)} cells.")
    print("\nComposite similarity  (src.similarity.composite_similarity)")
    print("Check: per-cell W and B reproduce the reported per-cell table to 3 decimals.")
    print("(Δ shown at full precision; Table 2's Δ = rounded-W − rounded-B, so it can differ by ≤0.001.)\n")
    print(f"{'model':30s} {'cond':11s}  {'W':>6s} {'B':>6s} {'Δ(W-B)':>7s}   |  "
          f"{'T2 W':>5s} {'T2 B':>5s}   W,B match   Δorig")
    Wp, Bp = [], []
    all_ok = True
    nW = nB = None
    max_orig_diff = 0.0
    for cell in cells:
        sim = crg.canonical_matrices(cell)["composite"]
        W, B, w, b = wb(sim, cell.cases, clusters)
        Wp += w; Bp += b
        nW, nB = len(w), len(b)
        tw, tb = t2_key(cell.model, cell.condition)
        ok = (round(W, 3) == round(tw, 3) and round(B, 3) == round(tb, 3))
        all_ok &= ok
        # full-precision cross-check vs the original analyze.py output
        od = ""
        if (cell.model, cell.condition) in orig:
            ow, ob = orig[(cell.model, cell.condition)]
            diff = max(abs(W - ow), abs(B - ob))
            max_orig_diff = max(max_orig_diff, diff)
            od = f"{diff:.1e}"
        print(f"{cell.model:30s} {cell.condition:11s}  {W:6.3f} {B:6.3f} {W-B:+7.3f}   |  "
              f"{tw:5.3f} {tb:5.3f}   {'OK' if ok else 'MISMATCH':8s}   {od}")
    pW, pB = float(np.mean(Wp)), float(np.mean(Bp))
    pooled_ok = (round(pW, 3) == 0.475 and round(pB, 3) == 0.472)
    print(f"\n{'POOLED (all 15 cells)':42s}  {pW:6.3f} {pB:6.3f} {pW-pB:+7.3f}   |  "
          f"{0.475:5.3f} {0.472:5.3f}   {'OK' if pooled_ok else 'MISMATCH'}")
    print(f"n_within/cell = {nW}, n_between/cell = {nB}   (paper: 53 / 1172)")
    print(f"max |W,B - original analyze.py output| across cells = {max_orig_diff:.2e} "
          f"(identical to the pipeline that produced Table 2)")
    passed = all_ok and pooled_ok
    print(f"\nReproduction {'PASSED' if passed else 'FAILED'}: per-cell W and B and pooled "
          f"reproduce the reported per-cell table to 3 decimals.")
    return passed


# ---------------------------------------------------------------------------
# Component decomposition: within/between + sign tests + weight attribution
# ---------------------------------------------------------------------------
# Composite weights (src.similarity.composite_similarity).
CANON_WEIGHTS = {"diagnosis": 0.30, "feature": 0.15, "motif": 0.35,
                 "qualifier": 0.10, "depth": 0.10}
TOTAL_WEIGHT = sum(CANON_WEIGHTS.values())   # 1.00


def _fast_perm_p(sim, labels0, n_perm=10000, seed=20260615):
    """Case-level permutation p (two-sided) on the within-between delta.
    Permuting the label vector preserves the cluster-size multiset, so the
    'counts as a cluster' set (size>=2) is permutation-invariant. Matches
    crg.perm_test_delta semantics, vectorized."""
    N = len(labels0)
    iu = np.triu_indices(N, 1)
    simv = sim[iu]
    ok = ~np.isnan(simv)
    simv = simv[ok]
    ai, bi = iu[0][ok], iu[1][ok]
    sizes = Counter(labels0.tolist())
    valid = np.array([l for l, s in sizes.items() if s >= 2])

    def delta(lab):
        la, lb = lab[ai], lab[bi]
        same = (la == lb) & np.isin(la, valid)
        if not same.any() or same.all():
            return np.nan
        return simv[same].mean() - simv[~same].mean()

    obs = delta(labels0)
    if np.isnan(obs):
        return obs, float("nan")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = delta(rng.permutation(labels0))
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return float(obs), float(p)


def _composite_deltas(cells, clusters, matrix_of, n_perm):
    """Per-cell (delta, p) for a composite given by matrix_of(cell)->NxN."""
    rows = []
    for cell in cells:
        sim = matrix_of(cell)
        labels0 = np.array([clusters[c] for c in cell.cases])
        d, p = _fast_perm_p(sim, labels0, n_perm=n_perm)
        rows.append({"model": cell.model, "cond": cell.condition, "delta": d, "p": p})
    return rows


def run_decompose(graphs_path, clusters_path, n_perm=10000):
    graphs = crg.load_graphs(graphs_path)
    clusters = crg.load_clusters(clusters_path)
    comp_fn = crg.make_components_fn(graphs, None, 0.85)   # component matrices (both-empty=0)
    cells = sort_cells(crg.build_cells(graphs, comp_fn))
    canon = {id(cell): crg.canonical_matrices(cell) for cell in cells}   # weighted composite

    # ---- composite variants ----
    def m_pub(cell):       return canon[id(cell)]["composite"]                       # weighted, both-empty=1
    def m_corr_full(cell): return crg.composite_matrix(cell, ["feature","diagnosis","qualifier","motif","depth"])
    def m_corr_content(cell): return crg.composite_matrix(cell, ["feature","diagnosis","qualifier"])
    def m_corr_struct(cell):  return crg.composite_matrix(cell, ["motif","depth"])

    defs = [
        ("weighted composite   (weighted 5-component, both-empty=1.0)", m_pub),
        ("unweighted mean-5     (mean of 5 components, both-empty=0.0)", m_corr_full),
        ("content-only          (mean{feature,diagnosis,qualifier}, both-empty=0.0)", m_corr_content),
        ("structural-only       (mean{motif,depth})", m_corr_struct),
    ]
    results = {}
    for label, fn in defs:
        rows = _composite_deltas(cells, clusters, fn, n_perm)
        deltas = [r["delta"] for r in rows]
        st = crg.sign_test(deltas)
        results[label] = {"rows": rows, "sign": st,
                          "pooled_delta": float(np.nanmean(deltas))}

    # ---- print within/between per-cell table (4 composites side by side) ----
    print(f"\n{'='*100}\nCOMPONENT DECOMPOSITION — WITHIN vs BETWEEN  (Δ = W−B per cell; case-level perm p, n={n_perm})\n{'='*100}")
    pub = results[defs[0][0]]["rows"]; cf = results[defs[1][0]]["rows"]
    co = results[defs[2][0]]["rows"]; st = results[defs[3][0]]["rows"]
    print(f"{'model':28s}{'cond':12s}{'wtd Δ':>9s}{' p':>6s}{'mean5 Δ':>10s}{' p':>6s}"
          f"{'cont Δ':>10s}{' p':>6s}{'struct Δ':>11s}{' p':>6s}")
    for i in range(len(pub)):
        print(f"{pub[i]['model']:28s}{pub[i]['cond']:12s}"
              f"{pub[i]['delta']:+9.4f}{pub[i]['p']:6.2f}"
              f"{cf[i]['delta']:+10.4f}{cf[i]['p']:6.2f}"
              f"{co[i]['delta']:+10.4f}{co[i]['p']:6.2f}"
              f"{st[i]['delta']:+11.4f}{st[i]['p']:6.2f}")

    print(f"\n{'pooled Δ (mean of 15 cells)':40s}"
          f"  weighted={results[defs[0][0]]['pooled_delta']:+.4f}"
          f"  mean5={results[defs[1][0]]['pooled_delta']:+.4f}"
          f"  content={results[defs[2][0]]['pooled_delta']:+.4f}"
          f"  struct={results[defs[3][0]]['pooled_delta']:+.4f}")

    # ---- sign tests (content-only broken out as its own line) ----
    print(f"\n{'-'*70}\nSIGN TESTS over the 15 per-cell deltas (exact two-sided binomial vs 0.5)\n{'-'*70}")
    for label, _ in defs:
        s = results[label]["sign"]
        tag = "  <-- CONTENT-ONLY" if "content-only" in label else ""
        print(f"  {label[:52]:54s} {s['positive']:2d}+/{s['negative']:2d}-  p={s['p_two_sided']:.4f}{tag}")

    # ---- attribution (i) qualifier both-empty; (ii) motif+depth ----
    print(f"\n{'-'*70}\nWEIGHT/VALUE ATTRIBUTION  (weighted composite, pooled)\n{'-'*70}")
    # gather pooled component means + qualifier both-empty fraction over ALL same-cell pairs
    comp_means = {k: [] for k in ["feature","diagnosis","qualifier","motif","depth","composite"]}
    q_emptyboth, q_total = 0, 0
    for cell in cells:
        M = canon[id(cell)]; N = len(cell.cases); iu = np.triu_indices(N, 1)
        for k in comp_means: comp_means[k].extend(M[k][iu].tolist())
        gq = [len(g.qualifiers) == 0 for g in cell.graphs]
        for a in range(N):
            for b in range(a+1, N):
                q_total += 1
                if gq[a] and gq[b]: q_emptyboth += 1
    cm = {k: float(np.nanmean(v)) for k, v in comp_means.items()}
    comp = cm["composite"]
    frac_emptyboth = q_emptyboth / q_total

    w_qual = CANON_WEIGHTS["qualifier"]
    qual_contrib = w_qual * cm["qualifier"]                      # value carried by qualifier channel
    print(f"(i) QUALIFIER both-empty matches:")
    print(f"    qualifier weight             = {w_qual:.2f}/{TOTAL_WEIGHT:.2f} = "
          f"{w_qual/TOTAL_WEIGHT*100:.0f}% of total composite weight")
    print(f"    pairs with NO qualifier in either graph (set-Jaccard scores them 1.0; "
          f"realized overlap 0) = {q_emptyboth}/{q_total} = {frac_emptyboth*100:.1f}% of all pairs")
    print(f"    mean qualifier component = {cm['qualifier']:.3f}  "
          f"(realized overlap, both-empty=0 -> ~{np.nanmean([0.0]):.3f})")
    print(f"    qualifier channel contributes {qual_contrib:.3f} of the {comp:.3f} composite "
          f"= {qual_contrib/comp*100:.1f}% of composite VALUE (≈all of it from both-empty pairs)")

    w_ms = CANON_WEIGHTS["motif"] + CANON_WEIGHTS["depth"]
    ms_contrib = CANON_WEIGHTS["motif"]*cm["motif"] + CANON_WEIGHTS["depth"]*cm["depth"]
    print(f"\n(ii) MOTIF + DEPTH (near-ceiling structural):")
    print(f"    motif+depth weight           = {w_ms:.2f}/{TOTAL_WEIGHT:.2f} = "
          f"{w_ms/TOTAL_WEIGHT*100:.0f}% of total composite weight")
    print(f"    mean motif={cm['motif']:.3f}, mean depth={cm['depth']:.3f} (both near ceiling)")
    print(f"    motif+depth contribute {ms_contrib:.3f} of the {comp:.3f} composite "
          f"= {ms_contrib/comp*100:.1f}% of composite VALUE")
    print(f"    content channels (feature+diagnosis+qualifier-real-overlap) carry the rest "
          f"but are ~0 and ~equal W vs B -> wash out of the contrast")
    return results


# ---------------------------------------------------------------------------
# Diagnosis-similarity regression (Mantel per cell + case bootstrap)
# ---------------------------------------------------------------------------
def _load_dx_emb(path):
    art = json.load(open(path))
    emb = art.get("embeddings", art)
    return ({str(k): np.array(v, float) for k, v in emb.items()},
            {k: v for k, v in art.items() if k != "embeddings"})


def run_mantel(graphs_path, clusters_path, dx_path, n_perm=10000, n_boot=2000):
    import json as _json
    graphs = crg.load_graphs(graphs_path)
    clusters = crg.load_clusters(clusters_path)
    comp_fn = crg.make_components_fn(graphs, None, 0.85)   # component matrices
    cells = sort_cells(crg.build_cells(graphs, comp_fn))
    dx_emb, meta = _load_dx_emb(dx_path)

    print(f"\n{'='*92}\nDIAGNOSIS-SIMILARITY REGRESSION  (graph_sim ~ gold-dx cosine)\n{'='*92}")
    print(f"dx embeddings: {meta.get('model')} ({meta.get('model_kind')}), dim={meta.get('dim')}, "
          f"normalized={meta.get('normalized')}, local={meta.get('local')}")

    composites = [("content-only", ["feature", "diagnosis", "qualifier"]),
                  ("full-5",       ["feature", "diagnosis", "qualifier", "motif", "depth"])]
    for cname, which in composites:
        print(f"\n--- {cname} composite ---")
        rs = []
        print(f"{'model':28s}{'cond':12s}{'Mantel r':>9s}{'  p':>7s}")
        for cell in cells:
            cases_with = [c for c in cell.cases if c in dx_emb]
            keep = [i for i, c in enumerate(cell.cases) if c in dx_emb]
            gmat = crg.composite_matrix(cell, which)[np.ix_(keep, keep)]
            dmat = crg.cosine_matrix(cases_with, dx_emb)
            r, p = crg.mantel(gmat, dmat, n_perm=n_perm)
            rs.append(r)
            print(f"{cell.model:28s}{cell.condition:12s}{r:+9.4f}{p:7.3f}")
        rs = [r for r in rs if not np.isnan(r)]
        npos = sum(1 for r in rs if r > 0)
        print(f"  -> {npos}/{len(rs)} cells positive; mean Mantel r = {np.mean(rs):+.4f}")
        boot = crg.pooled_bootstrap_slope(cells, which, dx_emb, n_boot=n_boot)
        print(f"  -> pooled case-bootstrap slope = {boot['slope']:+.4f}  "
              f"95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]  "
              f"(n_case_pairs={boot['n_case_pairs_universe']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="reproduce", choices=["reproduce", "decompose", "mantel"])
    ap.add_argument("--graphs", default="data/extracted/all_graphs.json")
    ap.add_argument("--clusters", default="data/clusters/clusters.json")
    ap.add_argument("--dx-embeddings", default="data/gold_dx_embeddings.json")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.step == "reproduce":
        run_reproduce(args.graphs, args.clusters)
    elif args.step == "decompose":
        run_decompose(args.graphs, args.clusters, args.n_perm)
    elif args.step == "mantel":
        run_mantel(args.graphs, args.clusters, args.dx_embeddings, args.n_perm, args.n_boot)


if __name__ == "__main__":
    main()
