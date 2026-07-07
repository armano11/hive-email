"""Confidence estimation using bootstrap resampling.

Computes calibrated confidence intervals for evaluation scores
instead of reporting a single point estimate. This reveals whether
a score difference is meaningful or just noise.

The approach:
1. Collect per-example scores for each metric
2. Bootstrap resample (with replacement) N times
3. Compute percentile interval from the bootstrap distribution
4. Aggregate into overall confidence level with uncertainty

Inspired by: evalstats, bootstrap CI methodology from USENIX ATC '25
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapCI:
    lower: float
    upper: float
    mean: float
    std: float


@dataclass
class ConfidenceResult:
    level: str
    score: float
    confidence_interval: BootstrapCI
    per_metric_scores: dict
    calibrated: bool
    n_examples: int
    explanation: str


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a list of scores.

    Uses the percentile method: resample with replacement,
    compute mean each time, read percentiles.
    """
    scores_arr = np.array(scores)
    n = len(scores_arr)
    if n == 0:
        return BootstrapCI(lower=0.0, upper=0.0, mean=0.0, std=0.0)
    if n == 1:
        return BootstrapCI(
            lower=scores_arr[0], upper=scores_arr[0],
            mean=scores_arr[0], std=0.0,
        )

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)

    for i in range(n_resamples):
        sample = scores_arr[rng.integers(0, n, n)]
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean = float(np.mean(boot_means))
    std = float(np.std(boot_means))

    return BootstrapCI(lower=lower, upper=upper, mean=mean, std=std)


def estimate_confidence(
    per_example_results: list[dict],
    confidence_threshold: float = 0.7,
) -> ConfidenceResult:
    """Estimate overall confidence from per-example evaluation results.

    Uses bootstrap resampling on each metric and the overall score
    to produce calibrated confidence intervals.

    Args:
        per_example_results: List of dicts with per-example metric scores
        confidence_threshold: Threshold for high/medium/low classification

    Returns:
        ConfidenceResult with bootstrap intervals and calibrated level
    """
    if not per_example_results:
        return ConfidenceResult(
            level="N/A", score=0.0,
            confidence_interval=BootstrapCI(0, 0, 0, 0),
            per_metric_scores={}, calibrated=False,
            n_examples=0, explanation="No examples evaluated.",
        )

    metric_keys = [
        "intent_coverage", "tone_alignment",
        "semantic_similarity", "action_completeness",
        "policy_compliance",
    ]

    per_metric_scores = {}
    for key in metric_keys:
        scores = [r.get(key, 50) for r in per_example_results if r.get(key) is not None]
        if scores:
            ci = bootstrap_ci(scores)
            per_metric_scores[key] = {
                "mean": round(ci.mean, 1),
                "ci_95": [round(ci.lower, 1), round(ci.upper, 1)],
                "std": round(ci.std, 1),
                "n": len(scores),
            }

    overall_scores = [
        r.get("overall", 50)
        for r in per_example_results
        if r.get("overall") is not None
    ]

    if not overall_scores:
        return ConfidenceResult(
            level="N/A", score=0.0,
            confidence_interval=BootstrapCI(0, 0, 0, 0),
            per_metric_scores=per_metric_scores, calibrated=False,
            n_examples=len(per_example_results),
            explanation="No overall scores available.",
        )

    overall_ci = bootstrap_ci(overall_scores)
    mean_score = overall_ci.mean

    hallucination_scores = [
        r.get("hallucination_score", 100)
        for r in per_example_results
        if r.get("hallucination_score") is not None
    ]
    avg_hallucination = np.mean(hallucination_scores) if hallucination_scores else 100.0

    ci_width = overall_ci.upper - overall_ci.lower

    if mean_score >= 75 and avg_hallucination >= 85 and ci_width < 15:
        level = "High"
        explanation = (
            f"Strong average score ({mean_score:.1f}) with narrow confidence "
            f"interval (+-{ci_width / 2:.1f}) and low hallucination risk ({avg_hallucination:.1f}). "
            f"Bootstrap CI: [{overall_ci.lower:.1f}, {overall_ci.upper:.1f}] at 95% confidence."
        )
    elif mean_score >= 50 and avg_hallucination >= 70:
        level = "Medium"
        explanation = (
            f"Moderate score ({mean_score:.1f}) with CI width {ci_width:.1f}. "
            f"Hallucination risk score: {avg_hallucination:.1f}. "
            f"Bootstrap CI: [{overall_ci.lower:.1f}, {overall_ci.upper:.1f}]."
        )
    else:
        level = "Low"
        explanation = (
            f"Low score ({mean_score:.1f}) or high hallucination risk ({avg_hallucination:.1f}). "
            f"CI width: {ci_width:.1f}. "
            f"Bootstrap CI: [{overall_ci.lower:.1f}, {overall_ci.upper:.1f}]."
        )

    return ConfidenceResult(
        level=level,
        score=round(mean_score, 1),
        confidence_interval=overall_ci,
        per_metric_scores=per_metric_scores,
        calibrated=True,
        n_examples=len(per_example_results),
        explanation=explanation,
    )
