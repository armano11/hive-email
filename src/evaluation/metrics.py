"""Multi-dimensional evaluation metrics for email replies.

Full per-metric methodology (definition, trust, weaknesses, comparison method):
  src/evaluation/METRICS_METHODOLOGY.md

Three evaluation strategies, chosen per-dimension based on suitability:

1. LLM-as-Judge (intent coverage, tone, helpfulness)
   - Uses a separate evaluator LLM with structured rubrics
   - Returns score + explanation + missing items

2. Semantic Embedding Similarity (semantic alignment with gold)
   - Sentence transformer cosine similarity (all-MiniLM-L6-v2)
   - Measures meaning overlap independent of phrasing (unlike BLEU/ROUGE)

3. Deterministic Heuristics (action completeness, readability, policy, safety)
   - Fast, reproducible, cheap — explicit auditable rules
   - Used where rules are well-defined

Why not BLEU/ROUGE? They compare form, not function. A safe, tone-appropriate
reply with zero n-gram overlap scores 0 with BLEU but is the right answer.

Inspired by: G-Eval, DeepEval, Anthropic's evaluation framework
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    score: float
    reason: str
    method: str
    details: dict | None = None


# ── LLM-as-Judge Rubrics ──────────────────────────────────────────

INTENT_JUDGE_PROMPT = """You are evaluating a customer support reply. Rate how well it addresses the customer's intent.

Customer Email: {email}

Generated Reply: {reply}

Expected Actions (what the customer wanted): {actions}

Rate the reply on Intent Coverage from 0-100 based on:
- Did it address every specific request the customer made?
- Did it acknowledge all questions asked?
- Did it provide answers for each issue raised?

Score 90-100: All requests addressed completely
Score 70-89: Most requests addressed, minor gaps
Score 50-69: Some requests addressed, notable gaps
Score 0-49: Most requests missed

Return ONLY JSON:
{{"score": <int>, "reason": "<why this score>", "missing": ["<list of missed items>"]}}"""

TONE_JUDGE_PROMPT = """You are evaluating a customer support reply's tone appropriateness.

Customer Email: {email}
Customer Emotion: {emotion}

Generated Reply: {reply}

Rate the reply's Tone from 0-100 based on:
- If customer is frustrated/angry: reply must show empathy and apology
- If customer is confused: reply must be clear and patient
- If customer is appreciative: reply should be warm and grateful
- If customer is neutral/polite: reply should be professional
- Does the tone match what the situation demands?
- Is it professional? Empathetic? Helpful?

Score 90-100: Perfect tone for the situation
Score 70-89: Good tone, minor improvements possible
Score 50-69: Acceptable but noticeably off in some way
Score 0-49: Inappropriate tone for the situation

Return ONLY JSON:
{{"score": <int>, "reason": "<why this score>", "tone_used": "<observed tone>"}}"""

HELPFULNESS_JUDGE_PROMPT = """You are evaluating how helpful a customer support reply is.

Customer Email: {email}
Generated Reply: {reply}

Rate Helpfulness from 0-100:
- Would this reply likely resolve the customer's issue?
- Does it provide clear, actionable next steps?
- Does it leave the customer needing to write another email?
- Is the information accurate and relevant?

Score 90-100: Likely resolves issue completely
Score 70-89: Mostly helpful, minor gaps
Score 50-69: Partially helpful, notable gaps
Score 0-49: Not helpful or misleading

Return ONLY JSON:
{{"score": <int>, "reason": "<why this score>", "next_steps_clear": <bool>, "customer_effort": "low|medium|high"}}"""


def _call_llm_judge(provider, system_prompt: str, user_prompt: str) -> dict:
    """Call LLM judge and parse structured response."""
    response = provider.generate(system_prompt, user_prompt, max_tokens=512)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]

    import json
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM judge JSON parse failed, raw: %s", content[:100])
        return {"score": 50, "reason": "Failed to parse judge response."}


# ── Deterministic Helpers ─────────────────────────────────────────

def _check_action_phrases(reply_lower: str) -> dict:
    action_signals = [
        "will", "can", "please", "let me", "i've", "i have",
        "you can", "we'll", "send", "check", "click", "go to",
        "follow these", "steps", "contact", "reach out",
    ]
    count = sum(1 for s in action_signals if s in reply_lower)
    has_timeline = any(w in reply_lower for w in
                       ["day", "hour", "minute", "week", "today", "tomorrow"])
    has_contact = any(w in reply_lower for w in
                      ["let me know", "contact", "reach out", "email us"])
    return {"action_signals": count, "has_timeline": has_timeline,
            "has_contact": has_contact}


def _compute_readability_score(reply: str) -> tuple[float, list[str]]:
    """Compute readability score with optional textstat."""
    sentences = [s.strip() for s in reply.split(".") if s.strip()]
    word_count = len(reply.split())
    sentence_count = max(len(sentences), 1)
    avg_words_per_sentence = word_count / sentence_count

    score = 85.0
    reasons = []

    if word_count < 15:
        score -= 15
        reasons.append("Too short (<15 words)")
    elif word_count > 150:
        score -= 10
        reasons.append("Too verbose (>150 words)")
    else:
        reasons.append("Length appropriate")

    try:
        import textstat
        grade = textstat.flesch_kincaid_grade(reply)
        if grade < 6:
            score += 5
            reasons.append("Very easy to read")
        elif grade > 12:
            score -= 5
            reasons.append("Complex language")
        else:
            reasons.append("Readable level")
    except ImportError:
        if avg_words_per_sentence > 25:
            score -= 5
            reasons.append("Long sentences")
        elif avg_words_per_sentence < 8:
            score -= 3
            reasons.append("Very short sentences")

    return max(0, min(100, score)), reasons


# ── Public Evaluation Functions ────────────────────────────────────

def evaluate_intent_coverage_llm(
    email: str,
    reply: str,
    expected_actions: list[str],
    provider,
) -> MetricResult:
    """Use LLM-as-judge to evaluate intent coverage."""
    user_prompt = INTENT_JUDGE_PROMPT.format(
        email=email, reply=reply,
        actions=", ".join(expected_actions) if expected_actions else "General support"
    )
    result = _call_llm_judge(provider, "You are an evaluation expert.", user_prompt)

    score = min(100, max(0, result.get("score", 50)))
    missing = result.get("missing", [])
    reason = result.get("reason", "LLM judge evaluation completed.")

    if missing:
        reason += f" Missing: {'; '.join(missing[:3])}"

    return MetricResult(
        name="Intent Coverage",
        score=score,
        reason=reason,
        method="LLM-as-Judge",
        details={"missing_items": missing},
    )


def evaluate_tone_llm(
    email: str,
    reply: str,
    analysis: dict,
    provider,
) -> MetricResult:
    """Use LLM-as-judge to evaluate tone appropriateness."""
    emotion = analysis.get("customer_emotion", "neutral")
    user_prompt = TONE_JUDGE_PROMPT.format(
        email=email, reply=reply, emotion=emotion
    )
    result = _call_llm_judge(provider, "You are a tone analysis expert.", user_prompt)

    score = min(100, max(0, result.get("score", 50)))
    tone_used = result.get("tone_used", "professional")
    reason = result.get("reason", "Tone evaluation completed.")

    return MetricResult(
        name="Tone Alignment",
        score=score,
        reason=reason,
        method="LLM-as-Judge",
        details={"expected_emotion": emotion, "tone_used": tone_used},
    )


def evaluate_helpfulness_llm(
    email: str,
    reply: str,
    provider,
) -> MetricResult:
    """Use LLM-as-judge to evaluate helpfulness."""
    user_prompt = HELPFULNESS_JUDGE_PROMPT.format(email=email, reply=reply)
    result = _call_llm_judge(provider, "You are a helpfulness evaluator.", user_prompt)

    score = min(100, max(0, result.get("score", 50)))
    reason = result.get("reason", "Helpfulness evaluation completed.")
    next_steps = result.get("next_steps_clear", False)
    effort = result.get("customer_effort", "medium")

    return MetricResult(
        name="Helpfulness",
        score=score,
        reason=reason,
        method="LLM-as-Judge",
        details={"next_steps_clear": next_steps, "customer_effort": effort},
    )


def evaluate_semantic_similarity(
    reply: str,
    gold_reply: str | None,
) -> MetricResult:
    """Evaluate semantic similarity between generated and gold reply.

    Uses sentence transformers with cosine similarity. Only applicable
    when a gold/ideal reply exists. This measures meaning overlap
    independent of exact phrasing.
    """
    if not gold_reply:
        return MetricResult(
            name="Semantic Similarity",
            score=0.0,
            reason="No gold reply available for comparison.",
            method="N/A",
            details={"available": False},
        )

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb1 = model.encode(reply, normalize_embeddings=True)
        emb2 = model.encode(gold_reply, normalize_embeddings=True)
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        similarity = float(cosine_similarity([emb1], [emb2])[0][0])
        score = round(similarity * 100, 1)
    except Exception as e:
        logger.warning("Semantic similarity failed: %s", e)
        return MetricResult(
            name="Semantic Similarity",
            score=0.0,
            reason=f"Embedding model unavailable: {e}",
            method="sentence-transformers",
            details={"error": str(e)},
        )

    if score >= 80:
        reason = "Strong semantic alignment with ideal reply."
    elif score >= 60:
        reason = "Moderate semantic overlap with ideal reply."
    else:
        reason = "Low semantic similarity to ideal reply — meaning differs."

    return MetricResult(
        name="Semantic Similarity",
        score=score,
        reason=reason,
        method="sentence-transformers",
        details={"available": True, "model": "all-MiniLM-L6-v2"},
    )


def evaluate_action_completeness_deterministic(reply: str) -> MetricResult:
    """Deterministic evaluation of action completeness."""
    reply_lower = reply.lower()
    signals = _check_action_phrases(reply_lower)

    score = 55.0
    reasons = []

    if signals["action_signals"] >= 3:
        score += 20
        reasons.append("Strong action language")
    elif signals["action_signals"] >= 1:
        score += 10
        reasons.append("Some action language")
    else:
        reasons.append("No action language detected")

    if signals["has_timeline"]:
        score += 15
        reasons.append("Timeline provided")

    if signals["has_contact"]:
        score += 10
        reasons.append("Follow-up invited")

    score = min(100, score)
    if score < 60:
        reasons.append("Unclear next steps")

    return MetricResult(
        name="Action Completeness",
        score=round(score, 1),
        reason=". ".join(reasons),
        method="deterministic",
        details=signals,
    )


def evaluate_readability_deterministic(reply: str, analysis: dict) -> MetricResult:
    """Deterministic readability evaluation."""
    score, reasons = _compute_readability_score(reply)
    complexity = analysis.get("complexity", "medium")
    sentences = len([s for s in reply.split(".") if s.strip()])

    if complexity == "simple" and sentences > 5:
        score -= 5
        reasons.append("Overly verbose for simple issue")

    return MetricResult(
        name="Readability",
        score=round(score, 1),
        reason=". ".join(reasons),
        method="deterministic",
        details={"word_count": len(reply.split()), "sentence_count": sentences},
    )


def evaluate_policy_compliance_deterministic(reply: str) -> MetricResult:
    """Check policy compliance with deterministic rules."""
    reply_lower = reply.lower()

    unprofessional = [
        "i don't know", "not my problem", "can't help",
        "you're wrong", "no idea",
    ]
    unsupported = [
        "guaranteed", "100%", "definitely fix",
        "absolutely solve", "promise you",
    ]

    found = []
    score = 100.0

    for phrase in unprofessional:
        if phrase in reply_lower:
            score -= 25
            found.append(f"Unprofessional: '{phrase}'")

    for phrase in unsupported:
        if phrase in reply_lower:
            score -= 15
            found.append(f"Unsupported: '{phrase}'")

    return MetricResult(
        name="Policy Compliance",
        score=max(0, round(score, 1)),
        reason=". ".join(found) if found else "No policy violations detected.",
        method="deterministic",
        details={"issues": found},
    )


def evaluate_safety_deterministic(reply: str) -> MetricResult:
    """Check safety with deterministic rules."""
    reply_lower = reply.lower()

    unsafe = ["ignore", "don't worry about", "it's fine", "skip"]
    misleading = ["this always works", "never", "always", "guaranteed to"]

    found = []
    score = 100.0

    for pattern in unsafe:
        if pattern in reply_lower:
            score -= 20
            found.append(f"Potentially unsafe: '{pattern}'")

    for pattern in misleading:
        if pattern in reply_lower:
            score -= 15
            found.append(f"Potentially misleading: '{pattern}'")

    return MetricResult(
        name="Safety",
        score=max(0, round(score, 1)),
        reason=". ".join(found) if found else "No safety issues detected.",
        method="deterministic",
        details={"issues": found},
    )
