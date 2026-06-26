from __future__ import annotations
"""
Cluster NEJM CPC cases by clinical similarity.

Two-level clustering:
  1. Primary: organ system (manually annotated or LLM-classified)
  2. Secondary: embedding similarity within organ system clusters

Usage:
    python -m src.cluster --cases data/raw/cases.json --output data/clusters/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from prompts.embedding import CASE_SUMMARY_SYSTEM, CASE_SUMMARY_USER
from src.utils import get_openai_client, get_case_presentation, parse_json_response, save_json


def classify_cases(cases: list[dict], client) -> list[dict]:
    """Use GPT-5.4 to classify each case by organ system and clinical features."""
    classified = []

    for case in cases:
        presentation = get_case_presentation(case)
        diagnosis = case.get("ground_truth", case.get("correct_diagnosis", "Unknown"))

        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": CASE_SUMMARY_SYSTEM},
                    {"role": "user", "content": CASE_SUMMARY_USER.format(
                        case_presentation=presentation,
                        correct_diagnosis=diagnosis,
                    )},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            summary = parse_json_response(response.choices[0].message.content)
        except Exception as e:
            print(f"  Classification error for {case.get('case_id')}: {e}")
            summary = {
                "organ_systems": ["unknown"],
                "primary_presenting_features": [],
                "reasoning_challenge_type": "unknown",
            }

        classified.append({
            "case_id": case.get("case_id"),
            "correct_diagnosis": diagnosis,
            "classification": summary,
        })
        time.sleep(0.3)

    return classified


def embed_cases(cases: list[dict], client) -> np.ndarray:
    """Generate embeddings for case presentations using OpenAI embeddings API."""
    texts = []
    for case in cases:
        presentation = get_case_presentation(case)
        # Truncate to avoid token limits on embedding model
        texts.append(presentation[:8000])

    # Batch embed
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    embeddings = np.array([r.embedding for r in response.data])
    return embeddings


def build_clusters(
    classified: list[dict],
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.7,
) -> dict:
    """Build two-level clusters: organ system → embedding subclusters."""

    # Level 1: Group by primary organ system
    organ_groups = {}
    for i, c in enumerate(classified):
        primary_system = c["classification"].get("organ_systems", ["unknown"])[0]
        if primary_system not in organ_groups:
            organ_groups[primary_system] = []
        organ_groups[primary_system].append(i)

    clusters = {}
    cluster_id = 0

    for system, indices in organ_groups.items():
        if len(indices) < min_cluster_size:
            # Too small to subcluster — treat as one cluster
            clusters[f"cluster_{cluster_id}"] = {
                "organ_system": system,
                "subcluster": 0,
                "case_ids": [classified[i]["case_id"] for i in indices],
                "case_indices": indices,
                "size": len(indices),
                "diagnoses": [classified[i]["correct_diagnosis"] for i in indices],
                "reasoning_types": [
                    classified[i]["classification"].get("reasoning_challenge_type", "unknown")
                    for i in indices
                ],
            }
            cluster_id += 1
        else:
            # Level 2: Subcluster by embedding similarity
            sub_embeddings = embeddings[indices]
            sim_matrix = cosine_similarity(sub_embeddings)

            # Use agglomerative clustering with distance threshold
            n_subclusters = min(len(indices) // min_cluster_size, 3)
            if n_subclusters > 1:
                agg = AgglomerativeClustering(
                    n_clusters=n_subclusters,
                    metric="cosine",
                    linkage="average",
                )
                labels = agg.fit_predict(sub_embeddings)
            else:
                labels = [0] * len(indices)

            for sub_label in set(labels):
                sub_indices = [indices[j] for j, l in enumerate(labels) if l == sub_label]
                clusters[f"cluster_{cluster_id}"] = {
                    "organ_system": system,
                    "subcluster": int(sub_label),
                    "case_ids": [classified[i]["case_id"] for i in sub_indices],
                    "case_indices": sub_indices,
                    "size": len(sub_indices),
                    "diagnoses": [classified[i]["correct_diagnosis"] for i in sub_indices],
                    "reasoning_types": [
                        classified[i]["classification"].get("reasoning_challenge_type", "unknown")
                        for i in sub_indices
                    ],
                    "mean_pairwise_similarity": float(np.mean(
                        cosine_similarity(embeddings[sub_indices])
                    )) if len(sub_indices) > 1 else 1.0,
                }
                cluster_id += 1

    return clusters


def run_clustering(cases_path: str, output_dir: str):
    """Run full clustering pipeline."""
    client = get_openai_client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(cases_path) as f:
        cases = json.load(f)

    print(f"Classifying {len(cases)} cases...")
    classified = classify_cases(cases, client)
    save_json(classified, output_dir / "case_classifications.json")

    print("Generating embeddings...")
    embeddings = embed_cases(cases, client)
    np.save(output_dir / "case_embeddings.npy", embeddings)

    print("Building clusters...")
    clusters = build_clusters(classified, embeddings)
    save_json(clusters, output_dir / "clusters.json")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"CLUSTERING SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total cases: {len(cases)}")
    print(f"Total clusters: {len(clusters)}")
    for cid, info in clusters.items():
        print(f"  {cid}: {info['organ_system']}"
              f" (n={info['size']}, sub={info['subcluster']})")
        for dx in info["diagnoses"]:
            print(f"    - {dx}")

    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster NEJM CPC cases")
    parser.add_argument("--cases", required=True, help="Path to cases.json")
    parser.add_argument("--output", default="data/clusters/", help="Output directory")
    args = parser.parse_args()

    run_clustering(args.cases, args.output)
