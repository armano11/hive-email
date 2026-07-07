"""Validate the LLM judge actually penalizes bad replies and rewards good ones.

This is the minimum bar for trusting any metric: feed the judge a known-bad
reply and assert the score is low, then feed a known-good reply and assert
the score is high. If either fails, the metric is not measuring quality.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metrics import (
    evaluate_intent_coverage_llm,
    evaluate_tone_llm,
)
from src.models.providers import NVIDIAProvider
from dotenv import load_dotenv


def test_judge_penalizes_bad_reply():
    load_dotenv()
    provider = NVIDIAProvider(
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
        model="meta/llama-3.1-8b-instruct"
    )

    email = "I need a refund for order ORD-123. Please process it urgently."
    bad_reply = "We don't care about your problem. Good luck."
    analysis = {"customer_emotion": "frustrated", "complexity": "simple", "summary": "Refund request"}

    result = evaluate_intent_coverage_llm(email, bad_reply, ["process refund"], provider)
    assert result.score < 50, (
        f"Bad reply scored {result.score} on intent — should be low"
    )
    result2 = evaluate_tone_llm(email, bad_reply, analysis, provider)
    assert result2.score < 50, (
        f"Bad reply scored {result2.score} on tone — should be low"
    )


def test_judge_rewards_good_reply():
    load_dotenv()
    provider = NVIDIAProvider(
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
        model="meta/llama-3.1-8b-instruct"
    )

    email = "I need a refund for order ORD-123. Please process it urgently."
    good_reply = "I'm sorry for the trouble. I've processed a full refund for order ORD-123. You should see it within 5-7 business days."
    analysis = {"customer_emotion": "frustrated", "complexity": "simple", "summary": "Refund request"}

    result = evaluate_intent_coverage_llm(email, good_reply, ["process refund"], provider)
    assert result.score >= 70, (
        f"Good reply scored {result.score} on intent — should be high"
    )
    result2 = evaluate_tone_llm(email, good_reply, analysis, provider)
    assert result2.score >= 70, (
        f"Good reply scored {result2.score} on tone — should be high"
    )


if __name__ == "__main__":
    print("Test 1: Judge penalizes bad reply...")
    test_judge_penalizes_bad_reply()
    print("  PASSED")

    print("Test 2: Judge rewards good reply...")
    test_judge_rewards_good_reply()
    print("  PASSED")

    print("\nBoth validation checks passed. The judge is measuring actual quality.")
