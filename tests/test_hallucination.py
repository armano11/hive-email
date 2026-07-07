"""Tests for hallucination detection."""

import sys
sys.path.insert(0, ".")

from src.evaluation.hallucination import detect_hallucinations


def test_heuristic_no_hallucination():
    reply = "Let me check your account and get back to you."
    email = "I need help with my account."
    analysis = {"entities": {}}
    result = detect_hallucinations(reply, email, analysis, [])
    assert result.risk == "low"
    assert result.score >= 90


def test_heuristic_hallucination_detected():
    reply = "I have processed your refund of $500."
    email = "I need help with my account. Thanks."
    analysis = {"entities": {}}
    result = detect_hallucinations(reply, email, analysis, [])
    assert result.risk != "low"


def test_heuristic_missing_action():
    reply = "Thanks for reaching out."
    email_text = "I need a refund for my order."
    analysis = {"entities": {}}
    result = detect_hallucinations(reply, email_text, analysis, ["process refund"])
    assert result.score < 100


def test_specific_claim_unsupported():
    reply = "You will receive the refund within 3-5 business days."
    email = "I want a refund."
    analysis = {"entities": {}}
    result = detect_hallucinations(reply, email, analysis, [])
    assert result.score < 100


def test_claim_extraction():
    """Test that claim extraction works on multi-sentence replies."""
    from src.evaluation.hallucination import _extract_claims
    reply = "I've processed your refund. You should see it in 5-7 days. Let me know if you need help."
    claims = _extract_claims(reply)
    assert len(claims) >= 2
    assert all(len(c.split()) >= 3 for c in claims)
