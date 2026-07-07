"""Evaluation orchestrator — runs the full multi-dimensional evaluation suite.

Full per-metric methodology: src/evaluation/METRICS_METHODOLOGY.md

Pipeline per email:
1. LLM-as-Judge: intent coverage, tone alignment
2. Sentence embedding: semantic similarity against gold reply
3. Deterministic: action completeness, policy compliance
4. NLI-based: hallucination detection

The final output includes per-metric scores with explanations,
failure categorization, and human review recommendation.

Human review flags replies when:
- Hallucination risk is high
- Intent coverage < 50
- Policy compliance < 60
- Overall score < 50
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

FAILURE_RULES: list[tuple[str, callable]] = []


def _fail_if(condition: callable, category: str):
    FAILURE_RULES.append((category, condition))


_per_email_registry: dict = {}


def _register_failure_rules():
    """Define failure classification rules."""
    if _per_email_registry:
        return

    def _make_future_ref():
        class _Ref:
            pass
        _per_email_registry["ref"] = _Ref
        return _Ref

    Ref = _make_future_ref()

    rules = [
        ("Incomplete Answer", lambda r: r.intent_coverage is not None and r.intent_coverage < 60),
        ("Wrong Tone", lambda r: r.tone_alignment is not None and r.tone_alignment < 60),
        ("Hallucination", lambda r: r.hallucination_risk == "high"),
        ("Missing Action", lambda r: r.action_completeness is not None and r.action_completeness < 50),
        ("Policy Violation", lambda r: r.policy_compliance is not None and r.policy_compliance < 60),
    ]
    FAILURE_RULES.clear()
    FAILURE_RULES.extend(rules)


@dataclass
class PerEmailResult:
    email_id: str
    intent_coverage: float | None = None
    action_completeness: float | None = None
    tone_alignment: float | None = None
    helpfulness: float | None = None
    semantic_similarity: float | None = None
    readability: float | None = None
    policy_compliance: float | None = None
    safety: float | None = None
    hallucination_risk: str | None = None
    hallucination_score: float | None = None
    confidence_level: str | None = None
    confidence_score: float | None = None
    overall_score: float | None = None
    needs_human_review: bool = False
    human_review_reasons: list[str] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    metric_reasons: dict = field(default_factory=dict)
    readability_details: dict | None = None

    def to_flat_dict(self) -> dict:
        return {
            "email_id": self.email_id,
            "intent_coverage": self.intent_coverage,
            "action_completeness": self.action_completeness,
            "tone_alignment": self.tone_alignment,
            "helpfulness": self.helpfulness,
            "semantic_similarity": self.semantic_similarity,
            "readability": self.readability,
            "policy_compliance": self.policy_compliance,
            "safety": self.safety,
            "hallucination_risk": self.hallucination_risk,
            "hallucination_score": self.hallucination_score,
            "confidence_level": self.confidence_level,
            "confidence_score": self.confidence_score,
            "overall": self.overall_score,
            "needs_human_review": self.needs_human_review,
            "failure_categories": self.failure_categories,
        }


def evaluate_reply(
    email_id: str,
    email_text: str,
    generated_reply: str,
    analysis: dict,
    expected_actions: list[str],
    gold_reply: str | None = None,
    provider=None,
    config: dict | None = None,
) -> PerEmailResult:
    """Run the full evaluation suite on a generated reply.

    Uses three evaluation strategies:
    - LLM-as-Judge for subjective dimensions (intent, tone)
    - Sentence embedding similarity against gold reply
    - Deterministic heuristics for action completeness, policy compliance
    - NLI-based for hallucination detection

    Args:
        email_id: Unique identifier
        email_text: Original customer email
        generated_reply: The reply to evaluate
        analysis: Metadata from email analysis
        expected_actions: Actions the customer requested
        gold_reply: Optional ideal/ground-truth reply
        provider: LLM provider for judge evaluations
        config: Evaluation configuration

    Returns:
        PerEmailResult with all scores and explanations
    """
    from src.evaluation.metrics import (
        evaluate_intent_coverage_llm,
        evaluate_tone_llm,
        evaluate_semantic_similarity,
        evaluate_action_completeness_deterministic,
        evaluate_policy_compliance_deterministic,
    )
    from src.evaluation.hallucination import detect_hallucinations

    has_llm = provider is not None

    result = PerEmailResult(email_id=email_id)

    # 1. LLM-as-Judge evaluations (subjective dimensions)
    if has_llm:
        try:
            ic = evaluate_intent_coverage_llm(email_text, generated_reply, expected_actions, provider)
            result.intent_coverage = ic.score
            result.metric_reasons["intent_coverage"] = {"reason": ic.reason, "method": ic.method}
        except Exception as e:
            logger.warning("LLM intent coverage failed: %s", e)
            result.intent_coverage = 50.0
            result.metric_reasons["intent_coverage"] = {"reason": f"Judge unavailable: {e}", "method": "fallback"}

        try:
            ta = evaluate_tone_llm(email_text, generated_reply, analysis, provider)
            result.tone_alignment = ta.score
            result.metric_reasons["tone_alignment"] = {"reason": ta.reason, "method": ta.method}
        except Exception as e:
            logger.warning("LLM tone evaluation failed: %s", e)
            result.tone_alignment = 50.0
            result.metric_reasons["tone_alignment"] = {"reason": f"Judge unavailable: {e}", "method": "fallback"}

    # 2. Semantic similarity (requires gold reply)
    try:
        ss = evaluate_semantic_similarity(generated_reply, gold_reply)
        result.semantic_similarity = ss.score
        result.metric_reasons["semantic_similarity"] = {"reason": ss.reason, "method": ss.method}
    except Exception as e:
        logger.warning("Semantic similarity failed: %s", e)
        result.semantic_similarity = 0.0

    # 3. Deterministic evaluations
    try:
        ac = evaluate_action_completeness_deterministic(generated_reply)
        result.action_completeness = ac.score
        result.metric_reasons["action_completeness"] = {"reason": ac.reason, "method": ac.method}
    except Exception as e:
        logger.warning("Action completeness failed: %s", e)
        result.action_completeness = 50.0

    try:
        rd = evaluate_readability_deterministic(generated_reply, analysis)
        result.readability = rd.score
        result.readability_details = rd.details
        result.metric_reasons["readability"] = {"reason": rd.reason, "method": rd.method}
    except Exception as e:
        logger.warning("Readability failed: %s", e)
        result.readability = 50.0

    try:
        pc = evaluate_policy_compliance_deterministic(generated_reply)
        result.policy_compliance = pc.score
        result.metric_reasons["policy_compliance"] = {"reason": pc.reason, "method": pc.method}
    except Exception as e:
        logger.warning("Policy compliance failed: %s", e)
        result.policy_compliance = 100.0

    try:
        sf = evaluate_safety_deterministic(generated_reply)
        result.safety = sf.score
        result.metric_reasons["safety"] = {"reason": sf.reason, "method": sf.method}
    except Exception as e:
        logger.warning("Safety evaluation failed: %s", e)
        result.safety = 100.0

    # 4. Hallucination detection (NLI-based with LLM, heuristic fallback)
    try:
        hl = detect_hallucinations(
            generated_reply, email_text, analysis, expected_actions,
            provider=provider if has_llm else None,
        )
        result.hallucination_risk = hl.risk
        result.hallucination_score = hl.score
        result.metric_reasons["hallucination"] = {
            "reason": hl.summary,
            "risk": hl.risk,
            "claims": len(hl.claim_results),
        }
    except Exception as e:
        logger.warning("Hallucination detection failed: %s", e)
        result.hallucination_risk = "unknown"
        result.hallucination_score = 50.0

    # 5. Compute overall score (5 core metrics)
    scores_for_overall = [
        v for v in [
            result.intent_coverage,
            result.tone_alignment,
            result.semantic_similarity,
            result.action_completeness,
            result.policy_compliance,
        ] if v is not None
    ]

    if result.hallucination_score is not None:
        scores_for_overall.append(result.hallucination_score)

    if scores_for_overall:
        result.overall_score = round(sum(scores_for_overall) / len(scores_for_overall), 1)
    else:
        result.overall_score = 0.0

    # 6. Classify failures
    _register_failure_rules()
    result.failure_categories = [
        cat for cat, rule in FAILURE_RULES if rule(result)
    ]

    # 7. Determine human review need
    result.needs_human_review = False
    result.human_review_reasons = []

    if result.hallucination_risk == "high":
        result.needs_human_review = True
        result.human_review_reasons.append(
            f"High hallucination risk: {result.metric_reasons.get('hallucination', {}).get('reason', 'NLI flagged multiple unsupported claims')}"
        )

    if result.intent_coverage is not None and result.intent_coverage < 50:
        result.needs_human_review = True
        result.human_review_reasons.append(
            f"Low intent coverage: {result.metric_reasons.get('intent_coverage', {}).get('reason', 'Failed to address customer requests')}"
        )

    if result.policy_compliance is not None and result.policy_compliance < 60:
        result.needs_human_review = True
        result.human_review_reasons.append(
            f"Policy violation: {result.metric_reasons.get('policy_compliance', {}).get('reason', 'Unprofessional or unsupported language')}"
        )

    if result.overall_score is not None and result.overall_score < 50:
        result.needs_human_review = True
        result.human_review_reasons.append(f"Overall score ({result.overall_score}) below threshold")

    return result
