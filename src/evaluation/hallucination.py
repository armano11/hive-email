"""Hallucination detection using NLI-based claim verification.

Extracts atomic claims from the generated reply and verifies each
against the original email using Natural Language Inference via an LLM.

This mirrors production systems (Microsoft, Vectara, Anthropic) that
decompose generation into claims and check support.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClaimResult:
    claim: str
    verdict: str
    confidence: float
    explanation: str


@dataclass
class HallucinationResult:
    risk: str
    score: float
    claim_results: list[ClaimResult]
    summary: str


HALLUCINATION_EVALUATOR_SYSTEM_PROMPT = """You are a factual consistency evaluator for customer support replies.
Your job is to detect contradictions — claims in the reply that DIRECTLY CONTRADICT the original customer email.

For each claim in the reply, classify as:
- SUPPORTED: The claim is directly supported by the email
- CONTRADICTED: The claim contradicts something in the email (this is the ONLY bad outcome)
- NEUTRAL: The claim is new information not mentioned in the email (this is ACCEPTABLE for support replies)
- UNVERIFIABLE: The claim cannot be verified from the email alone

Rules:
- NEUTRAL is OK and expected — support replies often provide new information
- Only CONTRADICTED is problematic (saying something opposite to the customer's email)
- Generic pleasantries ("thank you", "let me know", "we're here to help") are always SUPPORTED
- If the reply promises something the customer didn't request, mark as NEUTRAL (this is fine)
- If the reply states a fact that contradicts the email, mark as CONTRADICTED
- Be generous: when in doubt, prefer SUPPORTED or NEUTRAL over CONTRADICTED

Return JSON:
{
  "claims": [
    {
      "claim": "exact claim text",
      "verdict": "SUPPORTED|CONTRADICTED|NEUTRAL|UNVERIFIABLE",
      "confidence": 0.0-1.0,
      "explanation": "why this verdict was chosen"
    }
  ],
  "overall_assessment": "brief summary"
}"""


def _extract_claims(reply: str) -> list[str]:
    """Split reply into atomic claims for verification."""
    sentences = re.split(r'(?<=[.!?])\s+', reply.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 3:
            continue
        claims.append(s)
    return claims


def detect_hallucinations(
    reply: str,
    email_text: str,
    analysis: dict,
    expected_actions: list[str],
    provider=None,
) -> HallucinationResult:
    """Detect hallucinations using LLM-based NLI claim verification.

    Args:
        reply: The generated reply to evaluate
        email_text: The original customer email
        analysis: Metadata from email analysis
        expected_actions: Actions the customer requested
        provider: LLM provider for NLI evaluation. If None, uses heuristic fallback.

    Returns:
        HallucinationResult with per-claim breakdown and overall risk.
    """
    if provider is not None:
        try:
            return _nli_evaluate(reply, email_text, provider, analysis, expected_actions)
        except Exception as e:
            logger.warning("NLI evaluation failed, falling back to heuristic: %s", e)

    return _heuristic_fallback(reply, email_text, expected_actions, analysis)


def _nli_evaluate(reply: str, email_text: str, provider, analysis: dict | None = None, expected_actions: list[str] | None = None) -> HallucinationResult:
    """Use LLM to perform NLI on each claim."""
    claims = _extract_claims(reply)

    if not claims:
        return HallucinationResult(
            risk="low", score=100.0, claim_results=[],
            summary="No substantive claims to verify."
        )

    prompt = (
        f"Original Customer Email:\n{email_text}\n\n"
        f"Generated Reply:\n{reply}\n\n"
        f"Extract each claim from the reply and classify it as SUPPORTED, "
        f"CONTRADICTED, NEUTRAL, or UNVERIFIABLE against the original email."
    )

    response = provider.generate(
        HALLUCINATION_EVALUATOR_SYSTEM_PROMPT, prompt, max_tokens=1024
    )

    import json
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse NLI JSON, using heuristics")
        return _heuristic_fallback(reply, email_text, expected_actions or [], analysis)

    claim_results = []
    total = 0
    supported = 0

    for c in data.get("claims", []):
        verdict = c.get("verdict", "NEUTRAL").upper()
        result = ClaimResult(
            claim=c.get("claim", ""),
            verdict=verdict,
            confidence=c.get("confidence", 0.5),
            explanation=c.get("explanation", ""),
        )
        claim_results.append(result)
        total += 1
        if verdict == "SUPPORTED":
            supported += 1

    contradicted = sum(
        1 for c in claim_results
        if c.verdict in ("CONTRADICTED", "UNVERIFIABLE")
    )

    if total == 0:
        score = 100.0
    else:
        score = ((total - contradicted) / total) * 100 + (supported / total) * 10
        score = min(100, score)

    if score >= 90:
        risk = "low"
    elif score >= 70:
        risk = "medium"
    else:
        risk = "high"

    summary = data.get("overall_assessment", data.get("summary", ""))
    if not summary:
        summary = f"{supported}/{total} claims supported. {contradicted} potentially contradicted."

    return HallucinationResult(
        risk=risk,
        score=round(score, 1),
        claim_results=claim_results,
        summary=summary,
    )


def _heuristic_fallback(
    reply: str,
    email_text: str,
    expected_actions: list[str],
    analysis: dict | None = None,
) -> HallucinationResult:
    """Fallback heuristic when LLM is unavailable."""
    reply_lower = reply.lower()
    email_lower = email_text.lower()
    details = []
    score = 100.0

    hallucination_patterns = [
        "i have processed", "i have issued", "i have initiated",
        "your refund has been", "your request has been",
        "we have credited", "a credit of", "you will receive",
        "i have refunded", "i have cancelled",
    ]

    for pattern in hallucination_patterns:
        if pattern in reply_lower:
            words = pattern.split()
            present = sum(1 for w in words if w in email_lower)
            if present < 2:
                score -= 12
                details.append(f"Unsolicited action: '{pattern}'")

    import re
    dollar_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
    for m in re.findall(dollar_pattern, reply, re.IGNORECASE):
        if m.lower() not in email_lower and \
           m not in str(analysis.get("entities", {}).values()):
            score -= 3
            details.append(f"Specific unsupported claim: '{m}'")

    time_patterns = [
        r'\d+[-–]\d+\s*(?:business\s*)?days?',
        r'within\s+\d+',
        r'in the next\s+\d+',
        r'by\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
    ]
    for pattern in time_patterns:
        matches = re.findall(pattern, reply, re.IGNORECASE)
        for m in matches:
            if m.lower() not in email_lower and \
               m not in str(analysis.get("entities", {}).values()):
                score -= 8
                details.append(f"Specific unsupported claim: '{m}'")

    for action in expected_actions:
        action_keywords = action.lower().split()
        if not any(kw in reply_lower for kw in action_keywords):
            score -= 10
            details.append(f"Expected action '{action}' not addressed in reply")

    score = max(0, score)

    if score >= 90:
        risk = "low"
    elif score >= 70:
        risk = "medium"
    else:
        risk = "high"

    claim_results = [ClaimResult(
        claim=d, verdict="NEUTRAL",
        confidence=0.5, explanation="Heuristic detection"
    ) for d in details[:5]]

    return HallucinationResult(
        risk=risk,
        score=round(score, 1),
        claim_results=claim_results,
        summary=f"Heuristic hallucination check: {len(details)} potential issue(s)" if details
                else "No hallucination signals detected by heuristics.",
    )
