"""Tests for synthetic dataset generation."""

import sys
sys.path.insert(0, ".")

from dataset.generate_dataset import generate_dataset


def test_dataset_size():
    dataset = generate_dataset(120)
    assert len(dataset) == 120


def test_all_have_required_fields():
    dataset = generate_dataset(50)
    required = ["id", "customer_email", "intent", "urgency", "tone",
                 "expected_actions", "gold_reply"]
    for example in dataset:
        for field in required:
            assert field in example, f"Missing {field} in {example['id']}"
        assert len(example["customer_email"]) > 10
        assert len(example["gold_reply"]) > 10


def test_valid_intents():
    valid_intents = {
        "refund", "billing", "login_issue", "subscription", "cancellation",
        "complaint", "bug_report", "feature_request", "integration",
        "enterprise_sales", "password_reset", "pricing", "account_verification",
        "positive_feedback",
    }
    dataset = generate_dataset(100)
    for example in dataset:
        assert example["intent"] in valid_intents, \
            f"Invalid intent: {example['intent']}"


def test_valid_urgency():
    dataset = generate_dataset(100)
    for example in dataset:
        assert example["urgency"] in ("low", "medium", "high")


def test_all_intents_represented():
    dataset = generate_dataset(200)
    intents = {e["intent"] for e in dataset}
    assert len(intents) >= 12, f"Only {len(intents)} intents represented"


def test_expected_actions_is_list():
    dataset = generate_dataset(50)
    for example in dataset:
        assert isinstance(example["expected_actions"], list)


def test_id_format():
    dataset = generate_dataset(20)
    for example in dataset:
        assert example["id"].startswith("email-")


def test_no_empty_gold_reply():
    dataset = generate_dataset(50)
    for example in dataset:
        assert example["gold_reply"].strip(), f"Empty gold_reply in {example['id']}"
