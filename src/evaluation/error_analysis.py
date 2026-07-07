"""Error analysis — classify and aggregate failures across evaluated emails.

Provides:
- Failure mode distribution
- Best/worst examples with explanations
- Average scores with confidence intervals
- Common weakness identification
"""

from collections import Counter

from src.evaluation.judge import PerEmailResult


class ErrorAnalysis:
    def __init__(self, results: list[PerEmailResult]):
        self.results = results

    def failure_distribution(self) -> dict[str, int]:
        counter: Counter = Counter()
        for r in self.results:
            for cat in r.failure_categories:
                counter[cat] += 1
        return dict(counter.most_common())

    def worst_examples(self, top_n: int = 5) -> list[PerEmailResult]:
        return sorted(self.results, key=lambda r: r.overall_score)[:top_n]

    def best_examples(self, top_n: int = 5) -> list[PerEmailResult]:
        return sorted(self.results, key=lambda r: r.overall_score, reverse=True)[:top_n]

    def metric_averages(self) -> dict:
        metrics = [
            "intent_coverage", "tone_alignment",
            "semantic_similarity", "action_completeness",
            "policy_compliance",
        ]
        averages = {}
        for name in metrics:
            scores = [getattr(r, name, None) for r in self.results]
            scores = [s for s in scores if s is not None and s > 0]
            averages[name] = round(sum(scores) / len(scores), 1) if scores else 0.0

        hall_scores = [
            r.hallucination_score for r in self.results
            if r.hallucination_score is not None
        ]
        averages["hallucination"] = round(
            sum(hall_scores) / len(hall_scores), 1
        ) if hall_scores else 0.0

        overall = [
            r.overall_score for r in self.results
            if r.overall_score is not None
        ]
        averages["overall"] = round(
            sum(overall) / len(overall), 1
        ) if overall else 0.0

        return averages

    def confidence_distribution(self) -> dict[str, int]:
        dist: Counter = Counter()
        for r in self.results:
            level = r.confidence_level or "N/A"
            dist[level] += 1
        return dict(dist)

    def common_weaknesses(self) -> list[tuple[str, float]]:
        avgs = self.metric_averages()
        return sorted(
            [(k, v) for k, v in avgs.items() if k not in ("overall",)],
            key=lambda x: x[1],
        )

    def needs_human_review_count(self) -> int:
        return sum(1 for r in self.results if r.needs_human_review)

    def hallucination_breakdown(self) -> dict:
        breakdown: Counter = Counter()
        for r in self.results:
            if r.hallucination_risk:
                breakdown[r.hallucination_risk] += 1
        return dict(breakdown)
