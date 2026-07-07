"""Tests for evaluation metrics — deterministic only (LLM tests require API)."""

import sys
sys.path.insert(0, ".")

from src.evaluation.metrics import (
    evaluate_action_completeness_deterministic,
    evaluate_readability_deterministic,
    evaluate_policy_compliance_deterministic,
    evaluate_safety_deterministic,
    evaluate_semantic_similarity,
)


def test_action_completeness_good():
    reply = "I will process your refund today. Please contact us if you need help. You can follow these steps to check status."
    result = evaluate_action_completeness_deterministic(reply)
    assert result.score >= 70, f"Expected >=70, got {result.score}"
    assert result.method == "deterministic"
    assert result.details is not None


def test_action_completeness_poor():
    reply = "Okay."
    result = evaluate_action_completeness_deterministic(reply)
    assert result.score < 60


def test_readability_too_short():
    result = evaluate_readability_deterministic("Thanks.", {"complexity": "simple"})
    assert result.score < 80


def test_readability_normal():
    result = evaluate_readability_deterministic(
        "Thank you for reaching out. I have processed your refund. You will receive it within 5-7 business days. Let me know if you need anything else.",
        {"complexity": "simple"},
    )
    assert result.score >= 70


def test_policy_compliance_clean():
    result = evaluate_policy_compliance_deterministic(
        "I understand your concern. Let me help you resolve this issue."
    )
    assert result.score == 100.0


def test_policy_compliance_violation():
    result = evaluate_policy_compliance_deterministic(
        "I don't know how to fix this. Not my problem."
    )
    assert result.score < 100


def test_safety_clean():
    result = evaluate_safety_deterministic(
        "Please contact our support team for assistance."
    )
    assert result.score == 100.0


def test_safety_issue():
    result = evaluate_safety_deterministic(
        "Don't worry about it, just ignore the error message."
    )
    assert result.score < 100


def test_semantic_similarity_no_gold():
    result = evaluate_semantic_similarity("Hello.", None)
    assert result.score == 0.0
    assert "No gold reply" in result.reason


def test_semantic_similarity_with_gold():
    result = evaluate_semantic_similarity(
        "I've processed your refund.",
        "Your refund has been processed successfully.",
    )
    assert result.score >= 0
    assert result.method == "sentence-transformers"
