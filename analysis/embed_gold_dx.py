#!/usr/bin/env python3
"""
embed_gold_dx.py
================
Release artifact generator: embed the 50 verified gold (ground-truth) CPC
diagnoses LOCALLY with a sentence-transformer, for the diagnosis-similarity
regression (Mantel + case-bootstrap).

These are LOCAL embeddings -- NOT OpenAI text-embedding-3-large. The configured
key is OpenRouter, which has no /embeddings endpoint, so the as-originally-
specified embedding could not be produced. For 50 short diagnosis strings the
model choice does not change the cosine ordering materially.

Writes data/gold_dx_embeddings.json keyed by case_id, with a metadata block.
Idempotent: skips if a complete artifact already exists (use --force to redo).

    /tmp/crg_venv/bin/python -m analysis.embed_gold_dx
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

DATE = "2026-06-15"
SOURCE_FIELD = "_metadata.ground_truth"
OUT = "data/gold_dx_embeddings.json"
GRAPHS = "data/extracted/all_graphs.json"

# biomedical first, general fallback
CANDIDATES = [
    ("NeuML/pubmedbert-base-embeddings", "biomedical (PubMedBERT embeddings)"),
    ("pritamdeka/S-PubMedBert-MS-MARCO", "biomedical (S-PubMedBERT MS-MARCO)"),
    ("sentence-transformers/all-MiniLM-L6-v2", "general (all-MiniLM-L6-v2)"),
]


def load_gold() -> dict[str, str]:
    graphs = [g for g in json.load(open(GRAPHS)) if "_error" not in g]
    seen = defaultdict(set)
    for g in graphs:
        m = g["_metadata"]
        seen[m["case_id"]].add(m.get("ground_truth"))
    bad = {c: v for c, v in seen.items() if len(v) != 1 or not next(iter(v))}
    if bad:
        raise SystemExit(f"inconsistent/empty ground_truth: {bad}")
    return {c: next(iter(seen[c])) for c in sorted(seen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(OUT) and not args.force:
        art = json.load(open(OUT))
        if art.get("n_cases") == 50 and "embeddings" in art:
            print(f"{OUT} already exists ({art['model']}, dim={art['dim']}). Use --force to redo.")
            return

    gold = load_gold()
    cases = list(gold)               # already sorted by case_id
    texts = [gold[c] for c in cases]  # verbatim (incl. #38 trailing period)

    from sentence_transformers import SentenceTransformer
    import sentence_transformers as st

    model = name = kind = None
    for cand, klabel in CANDIDATES:
        try:
            print(f"loading {cand} ...")
            model = SentenceTransformer(cand)
            name, kind = cand, klabel
            break
        except Exception as e:
            print(f"  unavailable ({type(e).__name__}: {str(e)[:80]}); trying next")
    if model is None:
        raise SystemExit("no sentence-transformer model could be loaded")

    # cosine = dot product when normalized; store normalized vectors
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    dim = int(vecs.shape[1])

    artifact = {
        "model": name,
        "model_kind": kind,
        "is_text_embedding_3_large": False,
        "local": True,
        "library": f"sentence-transformers {st.__version__}",
        "dim": dim,
        "normalized": True,
        "date": DATE,
        "source_field": SOURCE_FIELD,
        "n_cases": len(cases),
        "embeddings": {c: [round(float(x), 7) for x in v] for c, v in zip(cases, vecs)},
    }
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(artifact, f)
    print(f"wrote {OUT}: {len(cases)} cases | model={name} ({kind}) | dim={dim} | "
          f"normalized=True | local=True (not text-embedding-3-large)")


if __name__ == "__main__":
    main()
