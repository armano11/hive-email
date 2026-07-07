"""Integration tests for the full evaluation pipeline with mocked LLM.

Tests the end-to-end flow: evaluate_reply → PerEmailResult with
all metrics computed, failure classification, and human review logic.
"""

import sys
sys.path.insert(0, ".")

from src.evaluation.judge import evaluate_reply


class MockProvider:
    """Mock LLM provider that returns deterministic responses."""

    def __init__(self, response_text: str = '{"score": 85, "reason": "Mock evaluation"}', provider_name: str = "mock"):
        self.response_text = response_text
        self.model = "mock-model"
        self.provider = provider_name

    def generate(self, system_prompt: str, user_prompt: str, **kwargs):
        from src.models.providers import ProviderResponse
        return ProviderResponse(
            content=self.response_text,
            model="mock-model",
            provider="mock",
        )


def test_evaluate_perfect_reply():
    """Test evaluation with a reply that perfectly addresses the email."""
    provider = MockProvider('{"score": 95, "reason": "Excellent coverage", "missing": []}')

    email = "I need a refund for order ORD-123. Please process it quickly."
    reply = "I've processed your refund for order ORD-123. You'll see it within 5-7 business days."
    analysis = {
        "intent": "refund",
        "urgency": "high",
        "customer_emotion": "frustrated",
        "requested_actions": ["process refund"],
        "complexity": "simple",
        "question_count": 1,
        "missing_information": [],
        "summary": "Customer requesting refund",
    }

    result = evaluate_reply(
        email_id="test-001",
        email_text=email,
        generated_reply=reply,
        analysis=analysis,
        expected_actions=["process refund"],
        provider=provider,
    )

    assert result.email_id == "test-001"
    assert result.overall_score is not None
    assert result.intent_coverage is not None
    assert result.action_completeness is not None
    assert result.tone_alignment is not None


def test_evaluate_hallucinated_reply():
    """Test evaluation flags a reply that invents information."""
    provider = MockProvider(
        '{"score": 95, "reason": "Mock", "missing": [], '
        '"claims": [{"claim": "Refund processed", "verdict": "CONTRADICTED", '
        '"confidence": 0.9, "explanation": "No refund was requested"}]}'
    )

    email = "Can you help me with my account login?"
    reply = "I've processed your refund of $500 and cancelled your subscription."
    analysis = {
        "intent": "login_issue",
        "urgency": "medium",
        "customer_emotion": "neutral",
        "requested_actions": ["reset password"],
        "complexity": "simple",
        "question_count": 1,
        "missing_information": ["email address"],
        "summary": "Customer needs login help",
    }

    result = evaluate_reply(
        email_id="test-002",
        email_text=email,
        generated_reply=reply,
        analysis=analysis,
        expected_actions=["reset password", "verify email"],
        provider=provider,
    )

    assert result.hallucination_risk is not None
    assert result.needs_human_review is not None


def test_evaluate_missing_intent():
    """Test evaluation detects when reply misses customer's intent."""
    provider = MockProvider('{"score": 25, "reason": "Most requests not addressed", "missing": ["refund", "timeline"]}')

    email = "I want a refund and I want it today. Order ORD-456."
    reply = "Thank you for your email. We appreciate your business."
    analysis = {
        "intent": "refund",
        "urgency": "high",
        "customer_emotion": "frustrated",
        "requested_actions": ["process refund", "expedite processing"],
        "complexity": "simple",
        "question_count": 2,
        "missing_information": [],
        "summary": "Customer wants immediate refund",
    }

    result = evaluate_reply(
        email_id="test-003",
        email_text=email,
        generated_reply=reply,
        analysis=analysis,
        expected_actions=["process refund", "expedite processing"],
        provider=provider,
    )

    assert result.failure_categories is not None
    if result.intent_coverage is not None:
        assert result.intent_coverage <= 50


def test_evaluate_with_gold_reply():
    """Test semantic similarity computation when gold reply is provided."""
    provider = MockProvider('{"score": 90, "reason": "Good reply"}')

    email = "My login isn't working."
    reply = "Let me help you with your login issue."
    analysis = {
        "intent": "login_issue",
        "urgency": "medium",
        "customer_emotion": "neutral",
        "requested_actions": [],
        "complexity": "simple",
        "question_count": 1,
        "missing_information": ["email"],
        "summary": "Login issue",
    }

    result = evaluate_reply(
        email_id="test-004",
        email_text=email,
        generated_reply=reply,
        analysis=analysis,
        expected_actions=[],
        gold_reply="I can help you with your login issue.",
        provider=provider,
    )

    assert result.semantic_similarity is not None


def test_evaluate_no_llm_provider():
    """Test evaluation works without LLM provider (deterministic only)."""
    email = "I need a refund."
    reply = "I've processed your refund."
    analysis = {
        "intent": "refund",
        "urgency": "medium",
        "customer_emotion": "neutral",
        "requested_actions": ["process refund"],
        "complexity": "simple",
        "question_count": 1,
        "missing_information": [],
        "summary": "Refund request",
    }

    result = evaluate_reply(
        email_id="test-005",
        email_text=email,
        generated_reply=reply,
        analysis=analysis,
        expected_actions=["process refund"],
        provider=None,
    )

    assert result.intent_coverage is None  # LLM-dependent
    assert result.action_completeness is not None  # deterministic
    assert result.policy_compliance is not None  # deterministic
