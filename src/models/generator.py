"""Orchestrates email analysis, RAG retrieval, prompt construction, and generation.

Pipeline:
1. Analyze email → structured metadata
2. Retrieve similar past cases via RAG
3. Build dynamic prompt with few-shot examples
4. Call LLM → generated reply
"""

import logging
from typing import Any

from src.models.providers import Provider
from src.models.email_analyzer import analyze_email
from src.models.prompt_builder import build_prompt, from_analysis
from src.models.retriever import Retriever

logger = logging.getLogger(__name__)


def generate_reply(
    provider: Provider,
    email_text: str,
    retriever: Retriever | None = None,
    generation_config: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """Generate a support reply with optional RAG grounding.

    Args:
        provider: LLM provider instance
        email_text: Raw customer email
        retriever: Optional RAG retriever for few-shot examples
        generation_config: Generation parameters (temperature, max_tokens)

    Returns:
        (reply_text, analysis_metadata) tuple
    """
    config = generation_config or {}

    logger.info("Analyzing email...")
    analysis = analyze_email(provider, email_text)

    logger.info(
        "Analysis: intent=%s urgency=%s emotion=%s questions=%d",
        analysis.get("intent"),
        analysis.get("urgency"),
        analysis.get("customer_emotion"),
        analysis.get("question_count", 0),
    )

    prompt_config = from_analysis(analysis)

    few_shot_examples = None
    if retriever is not None:
        try:
            results = retriever.retrieve(
                email_text=email_text,
                intent=analysis.get("intent"),
                top_k=3,
                same_intent_only=True,
            )
            if results:
                few_shot_examples = [
                    {"email_text": ex.email_text, "gold_reply": ex.gold_reply}
                    for ex, _ in results
                ]
                logger.info("Retrieved %d few-shot examples for intent=%s",
                            len(results), analysis.get("intent"))
        except Exception as e:
            logger.warning("RAG retrieval failed, proceeding without examples: %s", e)

    system_prompt, user_prompt = build_prompt(prompt_config, few_shot_examples)

    logger.info("Generating reply...")
    response = provider.generate(
        system_prompt,
        user_prompt,
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 512),
    )

    usage = response.usage or {}
    logger.info(
        "Reply generated (%d tokens, model=%s)",
        usage.get("total_tokens", 0) or usage.get("output_tokens", 0),
        response.model,
    )

    return response.content, analysis
