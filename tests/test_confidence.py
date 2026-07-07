"""Tests for bootstrap confidence estimation."""

import sys
sys.path.insert(0, ".")

from src.evaluation.confidence import bootstrap_ci, estimate_confidence


def test_bootstrap_ci_basic():
    scores = [85.0, 90.0, 88.0, 92.0, 87.0]
    ci = bootstrap_ci(scores, n_resamples=1000, seed=42)
    assert ci.lower <= ci.mean <= ci.upper
    assert ci.mean > 0
    assert ci.std >= 0


def test_bootstrap_ci_single_value():
    ci = bootstrap_ci([85.0], n_resamples=100)
    assert ci.lower == 85.0
    assert ci.upper == 85.0


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n_resamples=100)
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_bootstrap_ci_all_same():
    ci = bootstrap_ci([80.0, 80.0, 80.0, 80.0, 80.0], n_resamples=100, seed=42)
    assert ci.mean == 80.0


def test_estimate_confidence_empty():
    result = estimate_confidence([])
    assert result.level == "N/A"
    assert result.score == 0.0


def test_estimate_confidence_high():
    results = [
        {"overall": 92, "hallucination_score": 95,
         "intent_coverage": 95, "action_completeness": 90,
         "tone_alignment": 90, "semantic_similarity": 85,
         "policy_compliance": 100},
        {"overall": 88, "hallucination_score": 90,
         "intent_coverage": 90, "action_completeness": 85,
         "tone_alignment": 85, "semantic_similarity": 80,
         "policy_compliance": 100},
        {"overall": 90, "hallucination_score": 92,
         "intent_coverage": 92, "action_completeness": 88,
         "tone_alignment": 88, "semantic_similarity": 82,
         "policy_compliance": 100},
    ]
    result = estimate_confidence(results)
    assert result.calibrated
    assert result.n_examples == 3
    assert result.confidence_interval.lower <= result.score <= result.confidence_interval.upper


def test_estimate_confidence_low():
    results = [
        {"overall": 35, "hallucination_score": 30,
         "intent_coverage": 30, "action_completeness": 25,
         "tone_alignment": 30, "semantic_similarity": 20,
         "policy_compliance": 50},
        {"overall": 40, "hallucination_score": 25,
         "intent_coverage": 35, "action_completeness": 30,
         "tone_alignment": 35, "semantic_similarity": 25,
         "policy_compliance": 60},
    ]
    result = estimate_confidence(results)
    assert result.level == "Low"
