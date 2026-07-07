"""Extract structured metadata from customer emails.

Uses the LLM to produce a consistent JSON description of the email's
intent, urgency, tone, entities, questions, and missing information.
"""

import json
import logging

from src.models.providers import Provider, ProviderResponse

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an email analysis expert for a customer support platform.
Analyze the given customer email and return a structured JSON with these fields:

{
  "intent": "One of: refund, billing, login_issue, subscription, cancellation, complaint, bug_report, feature_request, integration, enterprise_sales, password_reset, pricing, account_verification, positive_feedback",
  "urgency": "low | medium | high",
  "sentiment": "negative | neutral | positive",
  "customer_emotion": "frustrated | confused | polite | angry | urgent | appreciative | neutral | anxious",
  "requested_actions": ["list", "of", "specific", "actions", "customer", "expects"],
  "entities": {"product": "", "order_id": "", "email": "", "plan": "", "amount": ""},
  "language": "en",
  "complexity": "simple | medium | complex",
  "question_count": 0,
  "missing_information": ["anything", "the", "customer", "didn't", "provide"],
  "summary": "One sentence summary of the issue."
}

Return ONLY valid JSON. No markdown, no explanation."""


def analyze_email(provider: Provider, email_text: str) -> dict:
    user_prompt = f"Analyze this customer email:\n\n{email_text}"
    response = provider.generate(ANALYSIS_SYSTEM_PROMPT, user_prompt, max_tokens=512)
    return _parse_analysis(response)


def _parse_analysis(response: ProviderResponse) -> dict:
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse analysis JSON, using fallback. Raw: %s", content[:200])
        return {
            "intent": "unknown",
            "urgency": "medium",
            "sentiment": "neutral",
            "customer_emotion": "neutral",
            "requested_actions": [],
            "entities": {},
            "language": "en",
            "complexity": "medium",
            "question_count": 0,
            "missing_information": [],
            "summary": "Analysis failed to parse.",
        }
