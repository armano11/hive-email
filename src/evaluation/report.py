"""Report generation — JSON, CSV, and Markdown outputs.

Produces structured reports with per-example scores, aggregate statistics,
bootstrap confidence intervals, and failure analysis.
"""

import csv
import json
from pathlib import Path

from src.evaluation.judge import PerEmailResult
from src.evaluation.error_analysis import ErrorAnalysis
from src.evaluation.confidence import ConfidenceResult


def generate_report(
    results: list[PerEmailResult],
    output_dir: str = "results",
    confidence: ConfidenceResult | None = None,
) -> dict:
    """Generate all report formats and return the aggregate data."""
    analysis = ErrorAnalysis(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_data = _build_report_data(analysis, results, confidence)
    _write_json(report_data, out / "report.json")
    _write_csv(results, out / "leaderboard.csv")
    _write_markdown(report_data, analysis, confidence, out / "summary.md")

    return report_data


def _build_report_data(
    analysis: ErrorAnalysis,
    results: list[PerEmailResult],
    confidence: ConfidenceResult | None,
) -> dict:
    return {
        "total_emails": len(results),
        "average_scores": analysis.metric_averages(),
        "confidence": {
            "level": confidence.level if confidence else "N/A",
            "score": confidence.score if confidence else 0,
            "bootstrap_ci_95": [
                confidence.confidence_interval.lower if confidence else 0,
                confidence.confidence_interval.upper if confidence else 0,
            ],
            "calibrated": confidence.calibrated if confidence else False,
            "per_metric": confidence.per_metric_scores if confidence else {},
            "explanation": confidence.explanation if confidence else "",
        } if confidence else None,
        "confidence_distribution": analysis.confidence_distribution(),
        "failure_distribution": analysis.failure_distribution(),
        "hallucination_breakdown": analysis.hallucination_breakdown(),
        "needs_human_review": analysis.needs_human_review_count(),
        "common_weaknesses": [
            {"metric": k, "average_score": v}
            for k, v in analysis.common_weaknesses()
        ],
        "worst_examples": [
            {
                "id": r.email_id,
                "overall_score": r.overall_score,
                "failure_categories": r.failure_categories,
                "needs_human_review": r.needs_human_review,
            }
            for r in analysis.worst_examples(5)
        ],
        "best_examples": [
            {
                "id": r.email_id,
                "overall_score": r.overall_score,
                "confidence_level": r.confidence_level,
            }
            for r in analysis.best_examples(5)
        ],
        "all_results": [r.to_flat_dict() for r in results],
    }


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Report: {path}")


def _write_csv(results: list[PerEmailResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "email_id", "overall_score",
            "intent_coverage", "tone_alignment", "semantic_similarity",
            "action_completeness", "policy_compliance",
            "hallucination_risk", "hallucination_score",
            "needs_human_review", "failure_categories",
        ])
        sorted_results = sorted(results, key=lambda x: x.overall_score or 0, reverse=True)
        for r in sorted_results:
            writer.writerow([
                r.email_id,
                r.overall_score,
                r.intent_coverage,
                r.tone_alignment,
                r.semantic_similarity,
                r.action_completeness,
                r.policy_compliance,
                r.hallucination_risk,
                r.hallucination_score,
                "Yes" if r.needs_human_review else "No",
                "; ".join(r.failure_categories),
            ])
    print(f"  Leaderboard: {path}")


def _write_markdown(
    data: dict,
    analysis: ErrorAnalysis,
    confidence: ConfidenceResult | None,
    path: Path,
) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"**Total Emails Evaluated:** {data['total_emails']}",
        f"**Needs Human Review:** {data['needs_human_review']}",
        "",
        "## Average Scores",
        "",
        "| Metric | Average Score |",
        "|--------|--------------|",
    ]
    for metric, score in sorted(analysis.metric_averages().items()):
        metric_name = metric.replace("_", " ").title()
        lines.append(f"| {metric_name} | {score} |")

    if confidence:
        ci = confidence.confidence_interval
        lines += [
            "",
            "## Confidence (Bootstrap)",
            "",
            f"**Level:** {confidence.level}",
            f"**Score:** {confidence.score}",
            f"**95% CI:** [{ci.lower:.1f}, {ci.upper:.1f}]",
            f"**Std Dev:** {ci.std:.2f}",
            "",
            f"**Explanation:** {confidence.explanation}",
        ]
        if confidence.per_metric_scores:
            lines += [
                "",
                "### Per-Metric Confidence",
                "",
                "| Metric | Mean | 95% CI | Std |",
                "|--------|------|--------|-----|",
            ]
            for metric, scores in confidence.per_metric_scores.items():
                ci_vals = scores.get("ci_95", [0, 0])
                lines.append(
                    f"| {metric.replace('_', ' ').title()} | "
                    f"{scores.get('mean', 0)} | "
                    f"[{ci_vals[0]}, {ci_vals[1]}] | "
                    f"{scores.get('std', 0)} |"
                )

    lines += [
        "",
        "## Confidence Distribution",
        "",
        "| Level | Count |",
        "|-------|-------|",
    ]
    for level, count in sorted(analysis.confidence_distribution().items()):
        lines.append(f"| {level} | {count} |")

    lines += [
        "",
        "## Failure Distribution",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for cat, count in analysis.failure_distribution().items():
        lines.append(f"| {cat} | {count} |")

    lines += [
        "",
        "## Hallucination Breakdown",
        "",
        "| Risk Level | Count |",
        "|------------|-------|",
    ]
    for risk, count in analysis.hallucination_breakdown().items():
        lines.append(f"| {risk.title()} | {count} |")

    if data["common_weaknesses"]:
        lines += ["", "## Common Weaknesses (ranked)", ""]
        for i, item in enumerate(data["common_weaknesses"], 1):
            metric = item["metric"].replace("_", " ").title()
            lines.append(f"{i}. **{metric}**: {item['average_score']}")

    lines += [
        "",
        "## Best Examples (Top 5)",
        "",
    ]
    for ex in data["best_examples"]:
        lines.append(
            f"- **{ex['id']}**: {ex['overall_score']} ({ex.get('confidence_level', 'N/A')})"
        )

    lines += [
        "",
        "## Worst Examples (Bottom 5)",
        "",
    ]
    for ex in data["worst_examples"]:
        failures = ", ".join(ex["failure_categories"]) if ex["failure_categories"] else "None"
        lines.append(
            f"- **{ex['id']}**: {ex['overall_score']} - Failures: {failures}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {path}")
