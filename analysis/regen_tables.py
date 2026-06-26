#!/usr/bin/env python3
"""
Regenerate Table 3 / Table 4 / Figure 2 / content-only appendix numbers under
the distinct-case within-cluster pooling: within-cluster pairs are restricted
to distinct cases (same-case cross-model pairs measure inter-model agreement,
not cross-case schema reuse).
Structured condition (the condition Table 3/4/Fig 2 report), 250 graphs.
"""
import json, sys, numpy as np
from itertools import combinations
sys.path.insert(0, ".")
from src.similarity import (feature_overlap, diagnosis_overlap, motif_similarity,
                            composite_similarity, discriminating_edge_ratio)
from analysis import crg_analysis as crg

graphs=[g for g in json.load(open("data/extracted/all_graphs.json")) if "nodes" in g]
clusters=json.load(open("data/clusters/clusters.json"))
case2cl={c:(cid if info["size"]>=2 else None) for cid,info in clusters.items() for c in info["case_ids"]}
def cm(g): m=g["_metadata"]; return m["case_id"],m["source_model"]
S=[g for g in graphs if g["_metadata"]["condition"]=="structured"]

def split(include_same_case):
    W,B=[],[]
    for gi,gj in combinations(S,2):
        ci,_=cm(gi); cj,_=cm(gj)
        if ci==cj:
            if include_same_case: W.append((gi,gj))
            continue
        a,b=case2cl[ci],case2cl[cj]
        if a is not None and a==b: W.append((gi,gj))
        else: B.append((gi,gj))
    return W,B

# label embeddings for soft Jaccard
LE={k:np.array(v,float) for k,v in json.load(open("data/extracted/label_embeddings.json")).items()}
def feats(g): return crg.parse_graph(g).features
def soft_jac(a,b,thr=0.85):
    A=[x for x in a if x in LE]; Bs=[x for x in b if x in LE]
    if not A and not Bs: return 0.0
    if not A or not Bs: return 0.0
    Av=np.array([LE[x] for x in A]); Bv=np.array([LE[x] for x in Bs])
    Av/=np.linalg.norm(Av,axis=1,keepdims=True)+1e-9; Bv/=np.linalg.norm(Bv,axis=1,keepdims=True)+1e-9
    sim=Av@Bv.T
    order=np.dstack(np.unravel_index(np.argsort(-sim,axis=None),sim.shape))[0]
    ua,ub,m=set(),set(),0
    for i,j in order:
        if sim[i,j]<thr: break
        if i in ua or j in ub: continue
        ua.add(i);ub.add(j);m+=1
    return m/(len(A)+len(Bs)-m)

def comp_rows(pairs, soft_cap=None):
    fj,dj,qj,mo,de,co,sj=[],[],[],[],[],[],[]
    for n,(gi,gj) in enumerate(pairs):
        gi_p,gj_p=crg.parse_graph(gi),crg.parse_graph(gj)
        fj.append(crg.jaccard(gi_p.features,gj_p.features))
        dj.append(crg.jaccard(gi_p.diagnoses,gj_p.diagnoses))
        qj.append(crg.jaccard(gi_p.qualifiers,gj_p.qualifiers))
        c=composite_similarity(gi,gj)
        mo.append(c["motif_similarity"]); de.append(c["depth_similarity"]); co.append(c["composite"])
        if soft_cap is None or n<soft_cap:
            sj.append(soft_jac(gi_p.features,gj_p.features))
    f=lambda x:float(np.mean(x))
    return dict(feature=f(fj),diagnosis=f(dj),qualifier=f(qj),motif=f(mo),depth=f(de),
                composite=f(co),soft=f(sj),content_only=f(np.mean([fj,dj,qj],axis=0)),n=len(pairs))

# between sample (stable, low variance) for speed; within full
rng=np.random.default_rng(0)
W,B=split(include_same_case=False)
Bsamp=[B[i] for i in rng.choice(len(B),size=min(8000,len(B)),replace=False)]
wr=comp_rows(W,soft_cap=len(W)); br=comp_rows(Bsamp,soft_cap=2000)
print(f"\n=== Structured condition, distinct-case pooling ===  n_within={wr['n']}  n_between={len(B)} (sampled {len(Bsamp)})")
for k in ["feature","diagnosis","qualifier","motif","depth","soft","composite","content_only"]:
    print(f"  {k:13s} within {wr[k]:.4f}   between {br[k]:.4f}")
