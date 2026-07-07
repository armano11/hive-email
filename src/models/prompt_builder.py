"""Dynamically construct prompts based on email metadata and RAG examples.

The prompt adapts to intent, urgency, tone, emotion, and company policies.
When few-shot examples are available from the retriever, they're included
to ground the generation in past resolved cases.
"""

from dataclasses import dataclass


@dataclass
class PromptConfig:
    intent: str
    urgency: str
    tone: str
    emotion: str
    actions: list[str]
    complexity: str
    question_count: int
    missing_info: list[str]
    summary: str


SYSTEM_CORE = """You are a professional customer support agent for Hiver, an AI-powered customer support platform.
Your job is to write helpful, accurate, and empathetic email replies to customers."""

INTENT_GUIDELINES = {
    "refund": "Process refunds proactively. State the timeline clearly. Do not promise refunds outside policy.",
    "billing": "Investigate billing issues. Explain charges clearly. Correct errors. Offer credits for inconvenience.",
    "login_issue": "Provide step-by-step troubleshooting. Prioritize account access. Offer escalation path.",
    "subscription": "Explain plan differences clearly. Process upgrades/downgrades. Confirm no data loss.",
    "cancellation": "Acknowledge the request. Offer retention help. Explain data retention policy.",
    "complaint": "Apologize sincerely. Investigate the issue. Offer compensation for valid complaints.",
    "bug_report": "Thank the customer. Confirm reproduction. Share workaround. Provide fix timeline.",
    "feature_request": "Thank for the suggestion. Share roadmap status. Provide workaround if available.",
    "integration": "Diagnose configuration issues. Provide step-by-step setup help. Escalate if needed.",
    "enterprise_sales": "Be consultative. Understand requirements. Offer demo. Share relevant documentation.",
    "password_reset": "Help regain access quickly. Verify identity securely. Confirm email delivery.",
    "pricing": "Explain pricing transparently. Highlight value. Offer relevant discounts if available.",
    "account_verification": "Verify identity. Process verification. Communicate timeline clearly.",
    "positive_feedback": "Thank warmly. Share with the team. Encourage continued engagement.",
}

URGENCY_ADJUSTMENTS = {
    "high": "Respond with urgency. Prioritize resolution. Set clear expectations on timelines.",
    "medium": "Respond promptly. Provide thorough assistance. Follow up if needed.",
    "low": "Respond helpfully. Provide complete information. No rush implied.",
}

EMOTION_ADJUSTMENTS = {
    "frustrated": "Be empathetic and apologetic. Validate their frustration. Focus on solutions, not excuses.",
    "angry": "Stay calm and professional. Acknowledge their anger. Focus on resolving the issue.",
    "confused": "Be patient and explanatory. Avoid jargon. Confirm understanding.",
    "polite": "Match their politeness. Be warm and professional.",
    "urgent": "Be direct and action-oriented. Show you understand the time sensitivity.",
    "appreciative": "Be warm and grateful. Reflect their positive tone.",
    "neutral": "Be professional and helpful. Clear and direct communication.",
    "anxious": "Reassure the customer. Be clear about timelines. Follow up proactively.",
}

CONSTRAINTS = """Constraints:
- Professional and empathetic tone
- No hallucinations — only state facts you're certain of
- No promises unless supported by company policy
- Concise but complete
- Actionable — clearly explain next steps
- Do not invent specific timelines or policies
- If unsure, ask for clarification rather than guessing"""


def build_prompt(
    config: PromptConfig,
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts.

    Args:
        config: Structured prompt configuration from email analysis
        few_shot_examples: Optional list of {"email": ..., "reply": ...} dicts
                           retrieved by the RAG system

    Returns:
        (system_prompt, user_prompt) tuple
    """
    intent_guidance = INTENT_GUIDELINES.get(config.intent, "Handle the inquiry professionally.")
    urgency_guidance = URGENCY_ADJUSTMENTS.get(config.urgency, "")
    emotion_guidance = EMOTION_ADJUSTMENTS.get(config.emotion, "Be professional and helpful.")

    system_parts = [
        SYSTEM_CORE,
        f"\nIntent: {config.intent} — {intent_guidance}",
        f"\nUrgency: {config.urgency} — {urgency_guidance}" if config.urgency != "medium" else "",
        f"\nCustomer Emotion: {config.emotion} — {emotion_guidance}",
    ]

    if config.actions:
        system_parts.append(f"\nRequired actions: {', '.join(config.actions)}")

    if config.missing_info:
        system_parts.append(f"\nMissing information to clarify: {', '.join(config.missing_info)}")

    system_parts.append(f"\n{CONSTRAINTS}")

    system_prompt = "\n".join(part for part in system_parts if part)

    user_parts = []
    user_parts.append(f"Write a customer support reply for the following inquiry.")
    if config.summary:
        user_parts.append(f"\nIssue Summary: {config.summary}")
    user_parts.append(f"\nQuestions to answer: {config.question_count}")

    length_guide = {
        "simple": "Keep the reply concise (2-3 sentences).",
        "medium": "Write a moderate-length reply (3-5 sentences).",
        "complex": "Write a detailed reply (4-6 sentences) addressing all aspects.",
    }
    user_parts.append(f"\n{length_guide.get(config.complexity, '')}")

    if few_shot_examples:
        user_parts.append("\n\nHere are similar past cases and their ideal replies for reference:")
        for i, ex in enumerate(few_shot_examples[:3], 1):
            user_email = ex.get("email", ex.get("email_text", ""))
            user_reply = ex.get("reply", ex.get("gold_reply", ""))
            user_parts.append(f"\nPast Case {i}:")
            user_parts.append(f"Customer: {user_email[:200]}")
            user_parts.append(f"Ideal Reply: {user_reply[:200]}")

    user_parts.append(f"\n\nCustomer Email:\n{config.summary if config.summary else 'See summary above.'}")

    user_prompt = "\n".join(user_parts)

    return system_prompt, user_prompt


def from_analysis(analysis: dict) -> PromptConfig:
    return PromptConfig(
        intent=analysis.get("intent", "unknown"),
        urgency=analysis.get("urgency", "medium"),
        tone=analysis.get("tone", "neutral"),
        emotion=analysis.get("customer_emotion", "neutral"),
        actions=analysis.get("requested_actions", []),
        complexity=analysis.get("complexity", "medium"),
        question_count=analysis.get("question_count", 0),
        missing_info=analysis.get("missing_information", []),
        summary=analysis.get("summary", ""),
    )
