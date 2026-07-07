#!/usr/bin/env python3
"""Email Response Evaluator — main entry point.

Runs the full pipeline: dataset → analysis → RAG retrieval → generation →
multi-dimensional evaluation → reports.

Usage:
    python main.py [--emails N] [--provider openai] [--no-rag]
                   [--generate-dataset] [--dataset-size 120]
                   [--verbose]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.utils import load_config, load_env, get_api_key, setup_logging
from src.models.providers import get_provider
from src.models.generator import generate_reply
from src.evaluation.judge import evaluate_reply, PerEmailResult
from src.evaluation.confidence import estimate_confidence
from src.evaluation.report import generate_report
from src.evaluation.error_analysis import ErrorAnalysis

logger = logging.getLogger(__name__)


def format_terminal_output(
    email_id: str,
    email_text: str,
    reply: str,
    result: PerEmailResult,
    analysis: dict,
):
    """Print a clean, readable evaluation result to terminal."""
    sep = "=" * 72
    intent_display = analysis.get("intent", "unknown").replace("_", " ").title()
    print(f"\n{sep}")
    print(f"  Email #{email_id}")
    print(f"  Intent: {intent_display}")
    print(f"  Urgency: {analysis.get('urgency', 'unknown').title()}")
    print(f"  Emotion: {analysis.get('customer_emotion', 'unknown').title()}")
    print(sep)

    print(f"\n  Customer:")
    lines = email_text.strip().split("\n")
    for line in lines[:5]:
        print(f"    {line[:120]}")
    if len(lines) > 5:
        print(f"    ... ({len(lines)} lines total)")

    print(f"\n  Generated Reply:")
    for line in reply.strip().split("\n"):
        print(f"    {line}")

    print(f"\n  Evaluation:")

    metrics = [
        ("Intent Coverage", result.intent_coverage),
        ("Tone Alignment", result.tone_alignment),
        ("Helpfulness", result.helpfulness),
        ("Semantic Similarity", result.semantic_similarity),
        ("Action Completeness", result.action_completeness),
        ("Readability", result.readability),
        ("Policy Compliance", result.policy_compliance),
        ("Safety", result.safety),
    ]

    for name, score in metrics:
        if score is not None:
            bar_len = max(1, int(score / 5))
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    {name:25s} [{bar}] {score:5.1f}")

    if result.hallucination_risk:
        bar = " " * 22
        hr = f" {result.hallucination_risk:>8s}"
        print(f"    {'Hallucination Risk':25s} [{bar}] {hr}")

    if result.confidence_level:
        bar = " " * 22
        print(f"    {'Confidence':25s} [{bar}] {result.confidence_level:>8s} ({result.confidence_score:.0f}%)")

    if result.needs_human_review:
        print(f"    !! NEEDS HUMAN REVIEW")
        for reason in result.human_review_reasons[:3]:
            print(f"       -> {reason[:120]}")

    overall = result.overall_score or 0
    bar = " " * 22
    print(f"\n    {'Overall':25s} [{bar}] {overall:.1f}")
    print(sep)


def run_pipeline(config: dict, args: argparse.Namespace):
    """Run the full evaluation pipeline with RAG and multi-dimensional eval."""
    results_dir = Path(config.get("pipeline", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset or config.get("pipeline", {}).get("dataset_path", "dataset/emails.json")
    max_emails = args.emails or config.get("pipeline", {}).get("max_emails", 50)
    provider_name = args.provider or config.get("generation", {}).get("provider", "openai")
    model_name = args.model or config.get("generation", {}).get("model", "gpt-4o-mini")

    api_key = get_api_key(provider_name)
    if not api_key:
        logger.error(
            "No API key found for provider '%s'. Set %s_API_KEY in .env",
            provider_name,
            provider_name.upper(),
        )
        sys.exit(1)

    provider = get_provider(provider_name, api_key, model_name)
    gen_config = config.get("generation", {})
    eval_config = config.get("evaluation", {})

    logger.info("Loading dataset from %s", dataset_path)
    with open(dataset_path) as f:
        all_emails = json.load(f)

    emails = all_emails[:max_emails]
    logger.info("Processing %d emails out of %d total", len(emails), len(all_emails))

    # Initialize RAG retriever
    retriever = None
    if not args.no_rag:
        try:
            from src.models.retriever import build_retriever_from_dataset
            logger.info("Building RAG retriever from dataset...")
            retriever = build_retriever_from_dataset(dataset_path)
            logger.info("RAG retriever ready (%d examples indexed)", len(emails))
        except Exception as e:
            logger.warning("RAG retriever initialization failed, proceeding without: %s", e)

    # Run pipeline
    results: list[PerEmailResult] = []
    total = len(emails)

    for i, example in enumerate(emails):
        email_id = example["id"]
        email_text = example["customer_email"]
        expected_actions = example.get("expected_actions", [])
        gold_reply = example.get("gold_reply", "")

        logger.info("[%d/%d] %s — %s", i + 1, total, email_id, example.get("intent", "?"))

        try:
            reply, analysis = generate_reply(
                provider=provider,
                email_text=email_text,
                retriever=retriever,
                generation_config=gen_config,
            )
        except Exception as e:
            logger.error("Generation failed for %s: %s", email_id, e)
            continue

        try:
            result = evaluate_reply(
                email_id=email_id,
                email_text=email_text,
                generated_reply=reply,
                analysis=analysis,
                expected_actions=expected_actions,
                gold_reply=gold_reply,
                provider=provider,
                config=eval_config,
            )
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", email_id, e)
            continue

        results.append(result)

        show = args.verbose or result.needs_human_review or (i == total - 1)
        if show:
            format_terminal_output(email_id, email_text, reply, result, analysis)

    if not results:
        logger.warning("No results generated. Check API key and dataset.")
        return

    # Aggregate confidence via bootstrap
    logger.info("Computing bootstrap confidence intervals...")
    flat_results = [r.to_flat_dict() for r in results]
    confidence = estimate_confidence(flat_results)

    # Generate reports
    logger.info("Generating reports...")
    analysis_obj = ErrorAnalysis(results)
    report_data = generate_report(results, str(results_dir), confidence)

    # Print summary
    avg_scores = analysis_obj.metric_averages()
    hc = analysis_obj.needs_human_review_count()

    print(f"\n{'=' * 72}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Processed: {len(results)}/{total} emails")
    print(f"  Overall average: {avg_scores.get('overall', 0):.1f}")
    print(f"  95% CI: [{confidence.confidence_interval.lower:.1f}, {confidence.confidence_interval.upper:.1f}]")
    print(f"  Needs human review: {hc}")
    print(f"  Confidence: {confidence.level} ({confidence.score:.1f})")
    print(f"  Reports: {results_dir.resolve()}")
    print(f"{'=' * 72}")

    logger.info(
        "Done. Processed %d emails. Avg: %.1f. Human review: %d/%d. CI: [%.1f, %.1f]",
        len(results),
        avg_scores.get("overall", 0),
        hc,
        len(results),
        confidence.confidence_interval.lower,
        confidence.confidence_interval.upper,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Email Response Evaluator — evaluate LLM-generated support replies with RAG."
    )
    parser.add_argument("--emails", type=int, default=None,
                        help="Number of emails to process")
    parser.add_argument("--provider", type=str, default=None,
                        help="LLM provider (openai, anthropic, gemini, nvidia)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g., gpt-4o-mini, claude-3-haiku)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset JSON")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed per-email output")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG retrieval (generation without few-shot examples)")
    parser.add_argument("--generate-dataset", action="store_true",
                        help="Generate synthetic dataset")
    parser.add_argument("--dataset-size", type=int, default=120,
                        help="Number of dataset examples to generate")
    args = parser.parse_args()

    load_env()
    config = load_config(args.config)
    setup_logging(config)

    if args.generate_dataset:
        from dataset.generate_dataset import generate_dataset, save_dataset
        dataset = generate_dataset(args.dataset_size)
        save_dataset(dataset)
        return

    run_pipeline(config, args)


if __name__ == "__main__":
    main()
