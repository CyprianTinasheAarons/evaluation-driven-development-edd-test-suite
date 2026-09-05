"""Aggregate scores and CI-friendly baseline regression comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import EvaluationRun


@dataclass(frozen=True, slots=True)
class MetricSummary:
    name: str
    mean: float | None
    count: int
    applicable_count: int

    @property
    def coverage(self) -> float:
        return self.applicable_count / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "count": self.count,
            "applicable_count": self.applicable_count,
            "coverage": round(self.coverage, 6),
        }


@dataclass(frozen=True, slots=True)
class MetricComparison:
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    tolerance: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    baseline_run_id: str
    candidate_run_id: str
    tolerance: float
    comparisons: tuple[MetricComparison, ...]

    @property
    def passed(self) -> bool:
        # A missing metric is a configuration failure, not a successful gate.
        return bool(self.comparisons) and all(item.status in {"improved", "unchanged"} for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


def summarize_run(run: EvaluationRun) -> dict[str, MetricSummary]:
    values: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for result in run.results:
        for metric in result.metrics:
            counts[metric.metric] = counts.get(metric.metric, 0) + 1
            if metric.applicable and metric.score is not None:
                values.setdefault(metric.metric, []).append(metric.score)
    names = list(dict.fromkeys([*counts.keys(), *values.keys()]))
    return {
        name: MetricSummary(
            name=name,
            mean=None if not values.get(name) else sum(values[name]) / len(values[name]),
            count=counts.get(name, 0),
            applicable_count=len(values.get(name, [])),
        )
        for name in names
    }


def compare_runs(baseline: EvaluationRun, candidate: EvaluationRun, *, tolerance: float = 0.0) -> ComparisonReport:
    """Compare metric means; fail only when a candidate drops beyond tolerance."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if {result.example.id for result in baseline.results} != {result.example.id for result in candidate.results}:
        raise ValueError("baseline and candidate must evaluate the same example ids")
    baseline_summary = summarize_run(baseline)
    candidate_summary = summarize_run(candidate)
    names = list(dict.fromkeys([*baseline_summary.keys(), *candidate_summary.keys()]))
    comparisons: list[MetricComparison] = []
    for name in names:
        old = baseline_summary.get(name, MetricSummary(name, None, 0, 0)).mean
        new = candidate_summary.get(name, MetricSummary(name, None, 0, 0)).mean
        if old is None or new is None:
            status = "not_comparable"
            delta = None
        else:
            delta = new - old
            if delta < -tolerance:
                status = "regressed"
            elif delta > tolerance:
                status = "improved"
            else:
                status = "unchanged"
        comparisons.append(MetricComparison(name, old, new, delta, tolerance, status))
    return ComparisonReport(baseline.run_id, candidate.run_id, tolerance, tuple(comparisons))
