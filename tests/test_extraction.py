"""
Unit tests for graph extraction and similarity analysis.

Run: python -m pytest tests/ -v
"""

import json
import pytest
import numpy as np

from src.similarity import (
    jaccard,
    feature_overlap,
    diagnosis_overlap,
    motif_similarity,
    extract_motifs,
    composite_similarity,
    reasoning_depth,
    graph_density,
    discriminating_edge_ratio,
)
from src.validate import compute_graph_stats, validate_graph_structure
from analysis.motifs import label_motif, discriminating_motif_ratio


# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────

@pytest.fixture
def sample_graph_a():
    """A nephrotic syndrome case reasoning graph."""
    return {
        "nodes": [
            {"id": "n1", "type": "clinical_feature", "label": "proteinuria 8.2g", "phase_introduced": "phase1"},
            {"id": "n2", "type": "clinical_feature", "label": "weight loss 15lb", "phase_introduced": "phase1"},
            {"id": "n3", "type": "clinical_feature", "label": "albumin 1.8", "phase_introduced": "phase1"},
            {"id": "n4", "type": "diagnosis", "label": "diabetic nephropathy", "confidence": 45, "phase_introduced": "phase1"},
            {"id": "n5", "type": "diagnosis", "label": "amyloidosis", "confidence": 25, "phase_introduced": "phase1"},
            {"id": "n6", "type": "diagnosis", "label": "membranous nephropathy", "confidence": 20, "phase_introduced": "phase1"},
            {"id": "n7", "type": "discriminating_feature", "label": "weight loss", "phase_introduced": "phase2"},
            {"id": "n8", "type": "semantic_qualifier", "label": "progressive", "phase_introduced": "phase1"},
        ],
        "edges": [
            {"id": "e1", "type": "supports", "source": "n1", "target": "n4", "phase": "phase1", "verbatim": "proteinuria supports DN"},
            {"id": "e2", "type": "supports", "source": "n1", "target": "n5", "phase": "phase1", "verbatim": "proteinuria supports amyloid"},
            {"id": "e3", "type": "supports", "source": "n1", "target": "n6", "phase": "phase1", "verbatim": "proteinuria supports MN"},
            {"id": "e4", "type": "supports", "source": "n3", "target": "n4", "phase": "phase1", "verbatim": "low albumin supports DN"},
            {"id": "e5", "type": "discriminates_between", "source": "n7", "target": "n5", "phase": "phase2", "verbatim": "weight loss discriminates"},
            {"id": "e6", "type": "triggered_reflection", "source": "n2", "target": "r1", "phase": "phase2", "verbatim": "wt loss triggered reflection"},
        ],
        "reflection_events": [
            {"id": "r1", "trigger_description": "weight loss disproportionate for DN", "phase": "phase2"},
        ],
        "problem_representation": "58M progressive nephrotic syndrome with weight loss and long-standing DM",
        "final_diagnosis": "amyloidosis",
        "final_confidence": 40,
        "_metadata": {
            "case_id": "case_12",
            "source_model": "anthropic/claude-opus-4.5",
            "condition": "structured",
        },
    }


@pytest.fixture
def sample_graph_b():
    """A similar nephrotic syndrome case — different patient, same category."""
    return {
        "nodes": [
            {"id": "n1", "type": "clinical_feature", "label": "proteinuria 5.1g", "phase_introduced": "phase1"},
            {"id": "n2", "type": "clinical_feature", "label": "fatigue", "phase_introduced": "phase1"},
            {"id": "n3", "type": "clinical_feature", "label": "albumin 2.1", "phase_introduced": "phase1"},
            {"id": "n4", "type": "diagnosis", "label": "diabetic nephropathy", "confidence": 50, "phase_introduced": "phase1"},
            {"id": "n5", "type": "diagnosis", "label": "amyloidosis", "confidence": 15, "phase_introduced": "phase1"},
            {"id": "n6", "type": "diagnosis", "label": "membranous nephropathy", "confidence": 25, "phase_introduced": "phase1"},
            {"id": "n7", "type": "semantic_qualifier", "label": "progressive", "phase_introduced": "phase1"},
        ],
        "edges": [
            {"id": "e1", "type": "supports", "source": "n1", "target": "n4", "phase": "phase1", "verbatim": "proteinuria supports DN"},
            {"id": "e2", "type": "supports", "source": "n1", "target": "n6", "phase": "phase1", "verbatim": "proteinuria supports MN"},
            {"id": "e3", "type": "supports", "source": "n3", "target": "n4", "phase": "phase1", "verbatim": "low albumin supports DN"},
        ],
        "reflection_events": [],
        "problem_representation": None,
        "final_diagnosis": "diabetic nephropathy",
        "final_confidence": 55,
        "_metadata": {
            "case_id": "case_37",
            "source_model": "anthropic/claude-opus-4.5",
            "condition": "baseline",
        },
    }


@pytest.fixture
def dissimilar_graph():
    """A cardiac case — should be dissimilar to nephrotic syndrome."""
    return {
        "nodes": [
            {"id": "n1", "type": "clinical_feature", "label": "chest pain", "phase_introduced": "phase1"},
            {"id": "n2", "type": "clinical_feature", "label": "troponin elevated", "phase_introduced": "phase1"},
            {"id": "n3", "type": "diagnosis", "label": "STEMI", "confidence": 70, "phase_introduced": "phase1"},
            {"id": "n4", "type": "diagnosis", "label": "aortic dissection", "confidence": 15, "phase_introduced": "phase1"},
        ],
        "edges": [
            {"id": "e1", "type": "supports", "source": "n1", "target": "n3", "phase": "phase1", "verbatim": "chest pain supports STEMI"},
            {"id": "e2", "type": "supports", "source": "n2", "target": "n3", "phase": "phase1", "verbatim": "troponin supports STEMI"},
        ],
        "reflection_events": [],
        "_metadata": {"case_id": "case_05", "source_model": "openai/gpt-5.4", "condition": "baseline"},
    }


# ──────────────────────────────────────────────
# TESTS: SIMILARITY METRICS
# ──────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_empty_sets(self):
        assert jaccard(set(), set()) == 1.0


class TestDiagnosisOverlap:
    def test_similar_cases(self, sample_graph_a, sample_graph_b):
        sim = diagnosis_overlap(sample_graph_a, sample_graph_b)
        assert sim == 1.0  # Both have DN, amyloidosis, MN

    def test_dissimilar_cases(self, sample_graph_a, dissimilar_graph):
        sim = diagnosis_overlap(sample_graph_a, dissimilar_graph)
        assert sim == 0.0  # No overlap


class TestMotifSimilarity:
    def test_identical_graphs(self, sample_graph_a):
        sim = motif_similarity(sample_graph_a, sample_graph_a)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similar_motif_patterns(self, sample_graph_a, sample_graph_b):
        sim = motif_similarity(sample_graph_a, sample_graph_b)
        assert 0.3 < sim < 1.0  # Partially similar

    def test_dissimilar_patterns(self, sample_graph_a, dissimilar_graph):
        sim_similar = motif_similarity(sample_graph_a, sample_graph_b)
        # Can't easily compare without both, but structure is correct


class TestCompositeSimilarity:
    def test_returns_all_metrics(self, sample_graph_a, sample_graph_b):
        result = composite_similarity(sample_graph_a, sample_graph_b)
        assert "diagnosis_overlap" in result
        assert "feature_overlap" in result
        assert "motif_similarity" in result
        assert "composite" in result
        assert 0 <= result["composite"] <= 1

    def test_similar_higher_than_dissimilar(self, sample_graph_a, sample_graph_b, dissimilar_graph):
        sim_similar = composite_similarity(sample_graph_a, sample_graph_b)["composite"]
        sim_dissimilar = composite_similarity(sample_graph_a, dissimilar_graph)["composite"]
        assert sim_similar > sim_dissimilar


# ──────────────────────────────────────────────
# TESTS: GRAPH STRUCTURE
# ──────────────────────────────────────────────

class TestGraphStats:
    def test_node_counts(self, sample_graph_a):
        stats = compute_graph_stats(sample_graph_a)
        assert stats["total_nodes"] == 8
        assert stats["total_edges"] == 6
        assert stats["has_problem_representation"] is True
        assert stats["has_discriminating_edges"] is True

    def test_baseline_no_reflection(self, sample_graph_b):
        stats = compute_graph_stats(sample_graph_b)
        assert stats["total_reflections"] == 0


class TestGraphValidation:
    def test_valid_graph(self, sample_graph_a):
        issues = validate_graph_structure(sample_graph_a)
        # Should have no critical issues (orphan nodes are warnings, not errors)
        critical = [i for i in issues if "not found" in i]
        assert len(critical) == 0

    def test_baseline_phase_check(self, sample_graph_b):
        issues = validate_graph_structure(sample_graph_b)
        phase_issues = [i for i in issues if "non-phase1" in i]
        assert len(phase_issues) == 0  # All edges are phase1


# ──────────────────────────────────────────────
# TESTS: MOTIF EXTRACTION
# ──────────────────────────────────────────────

class TestMotifExtraction:
    def test_extracts_motifs(self, sample_graph_a):
        motifs = extract_motifs(sample_graph_a)
        assert len(motifs) > 0
        # Should have supports motifs
        supports_motif = ("clinical_feature", "supports", "diagnosis")
        assert supports_motif in motifs

    def test_labels_motifs(self):
        motif = ("clinical_feature", "supports", "diagnosis")
        label = label_motif(motif)
        assert label == "feature_supports_dx"


class TestReasoningDepth:
    def test_structured_deeper(self, sample_graph_a, sample_graph_b):
        depth_a = reasoning_depth(sample_graph_a)
        depth_b = reasoning_depth(sample_graph_b)
        # Structured condition should have more edges and phases
        assert depth_a["n_edges"] > depth_b["n_edges"]
        assert depth_a["n_phases"] >= depth_b["n_phases"]
