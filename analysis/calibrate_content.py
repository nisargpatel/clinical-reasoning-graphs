#!/usr/bin/env python3
"""
calibrate_content.py
====================
Recompute the Table 3 calibration bounds under the CONTENT-ONLY metric
(mean of feature/diagnosis/qualifier Jaccard, empty-vs-empty = 0), alongside
the weighted composite for validation (should reproduce 0.466 / 0.825 / 0.593
/ 0.862). Bounds: noise floor (shuffled labels), test-retest, inter-extractor,
gold-ablation. Also reports within/between content-only (per-cell pooling, the
primary Table-2 methodology) so we can check the content within-cluster signal
sits above its own noise floor.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from analysis import crg_analysis as crg
from analysis.run_analysis import sort_cells
from src.similarity import composite_similarity

RNG = np.random.default_rng(20260615)
ORIG = "data/extracted/all_graphs.json"


def content_only(gi_raw: dict, gj_raw: dict) -> float:
    """mean{feature, diagnosis, qualifier} Jaccard, empty-vs-empty = 0 (crg.jaccard)."""
    a, b = crg.parse_graph(gi_raw), crg.parse_graph(gj_raw)
    return float(np.mean([crg.jaccard(a.features, b.features),
                          crg.jaccard(a.diagnoses, b.diagnoses),
                          crg.jaccard(a.qualifiers, b.qualifiers)]))


def composite(gi_raw: dict, gj_raw: dict) -> float:
    return composite_similarity(gi_raw, gj_raw)["composite"]


def load(path):
    return [g for g in json.load(open(path)) if isinstance(g, dict) and "nodes" in g]


def key(g):
    m = g["_metadata"]
    return (m["case_id"], m["source_model"], m["condition"])


def paired_bound(exp_path, orig_by_key, label):
    """Pair each experiment graph with its original trace; compute both metrics."""
    comp, cont = [], []
    for g in load(exp_path):
        o = orig_by_key.get(key(g))
        if o is None:
            continue
        comp.append(composite(o, g))
        cont.append(content_only(o, g))
    return label, np.array(comp), np.array(cont)


# ---- noise floor: shuffle node labels (preserve types/edges/structure) ----
def build_label_pools(graphs):
    pools = defaultdict(list)
    for g in graphs:
        for n in g.get("nodes", []):
            pools[n.get("type")].append(n.get("label", ""))
    return pools


def shuffle_labels(g, pools):
    h = json.loads(json.dumps(g))   # deep copy
    for n in h.get("nodes", []):
        pool = pools.get(n.get("type"))
        if pool:
            n["label"] = str(RNG.choice(pool))
    return h


def noise_floor(graphs, pools, n=50):
    """n pairs of distinct same-(model,condition) graphs, labels shuffled in both."""
    by_cell = defaultdict(list)
    for g in graphs:
        m = g["_metadata"]
        by_cell[(m["source_model"], m["condition"])].append(g)
    cells = [c for c in by_cell.values() if len(c) >= 2]
    comp, cont = [], []
    for _ in range(n):
        cell = cells[RNG.integers(len(cells))]
        i, j = RNG.choice(len(cell), size=2, replace=False)
        a, b = shuffle_labels(cell[i], pools), shuffle_labels(cell[j], pools)
        comp.append(composite(a, b)); cont.append(content_only(a, b))
    return np.array(comp), np.array(cont)


def within_between_content(graphs, clusters):
    """Per-cell pooling (53 within / 1172 between per cell), content-only + composite."""
    comp_fn = crg.make_components_fn([crg.parse_graph(g) for g in graphs], None, 0.85)
    # rebuild cells from raw so we can call composite on .raw
    cells = sort_cells(crg.build_cells([crg.parse_graph(g) for g in graphs], comp_fn))
    Wc, Bc, Wk, Bk = [], [], [], []
    for cell in cells:
        N = len(cell.cases)
        cl = np.array([clusters.get(c, -(i + 1)) for i, c in enumerate(cell.cases)])
        from collections import Counter
        sizes = Counter(cl[cl >= 0])
        for a in range(N):
            for b in range(a + 1, N):
                same = (cl[a] >= 0 and cl[a] == cl[b] and sizes[cl[a]] >= 2)
                co = float(np.mean([cell.comp_mats["feature"][a, b],
                                    cell.comp_mats["diagnosis"][a, b],
                                    cell.comp_mats["qualifier"][a, b]]))
                cp = composite(cell.graphs[a].raw, cell.graphs[b].raw)
                (Wc if same else Bc).append(co)
                (Wk if same else Bk).append(cp)
    return (np.array(Wc), np.array(Bc), np.array(Wk), np.array(Bk))


def fmt(name, comp, cont, ref):
    print(f"  {name:26s}  composite {comp.mean():.3f} (SD {comp.std():.3f}, n={len(comp)})  "
          f"[ref {ref}]   |   CONTENT-ONLY {cont.mean():.4f} (SD {cont.std():.4f})")


def main():
    graphs = load(ORIG)
    orig_by_key = {key(g): g for g in graphs}
    clusters = crg.load_clusters("data/clusters/clusters.json")
    pools = build_label_pools(graphs)

    print("CALIBRATION BOUNDS — composite (validation) vs content-only")
    print("=" * 96)
    nf_comp, nf_cont = noise_floor(graphs, pools, n=50)
    fmt("noise floor (shuffled)", nf_comp, nf_cont, "0.466")
    for path, lab, ref in [
        ("experiments/retest/retest_extracted/all_graphs.json", "test-retest", "0.825"),
        ("experiments/extractor_comparison/opus_extracted/all_graphs.json", "inter-extractor", "0.593"),
        ("experiments/gold_ablation/all_graphs_no_gold.json", "gold-ablation", "0.862"),
    ]:
        _, comp, cont = paired_bound(path, orig_by_key, lab)
        fmt(lab, comp, cont, ref)

    Wc, Bc, Wk, Bk = within_between_content(graphs, clusters)
    print("\nWITHIN / BETWEEN (per-cell pooling, n_within=%d n_between=%d)" % (len(Wc), len(Bc)))
    print(f"  composite      within {Wk.mean():.3f}  between {Bk.mean():.3f}   [ref 0.475 / 0.472]")
    print(f"  CONTENT-ONLY   within {Wc.mean():.4f}  between {Bc.mean():.4f}   Δ={Wc.mean()-Bc.mean():+.4f}")

    print("\n" + "=" * 96)
    print("THE LOAD-BEARING CHECK (content-only):")
    print(f"  noise floor          = {nf_cont.mean():.4f}  (SD {nf_cont.std():.4f})")
    print(f"  between-cluster      = {Bc.mean():.4f}")
    print(f"  within-cluster       = {Wc.mean():.4f}")
    above = Wc.mean() - nf_cont.mean()
    sd = nf_cont.std() if nf_cont.std() > 0 else float('nan')
    print(f"  within − noise floor = {above:+.4f}   ({above/sd:.1f} noise-floor SDs)" if sd==sd
          else f"  within − noise floor = {above:+.4f}")
    print(f"  --> content within {'ABOVE' if Wc.mean() > nf_cont.mean() else 'NOT above'} its own noise floor")


if __name__ == "__main__":
    main()
