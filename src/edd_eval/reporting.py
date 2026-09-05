"""Stable JSON and human-readable Markdown output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .comparison import ComparisonReport, summarize_run
from .models import EvaluationRun


def write_json(value: EvaluationRun | ComparisonReport | dict[str, Any], path: str | Path) -> None:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_run(path: str | Path) -> EvaluationRun:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("run JSON must be an object")
        return EvaluationRun.from_dict(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not read evaluation run {path}: {exc}") from exc


def run_to_markdown(run: EvaluationRun) -> str:
    summary = summarize_run(run)
    lines = [
        f"# Evaluation run: {run.system_name}",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Dataset: `{run.dataset_name}`",
        f"- Examples: {len(run.results)}",
        f"- Duration: {run.duration_ms:.1f} ms",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Mean | Coverage |",
        "| --- | ---: | ---: |",
    ]
    for item in summary.values():
        mean = "n/a" if item.mean is None else f"{item.mean:.3f}"
        lines.append(f"| `{item.name}` | {mean} | {item.coverage:.0%} |")
    lines.extend(["", "## Per-example results", "", "| Example | Metrics |", "| --- | --- |"])
    for result in run.results:
        metric_text = ", ".join(
            f"`{metric.metric}`={metric.score:.3f}" if metric.score is not None else f"`{metric.metric}`=n/a"
            for metric in result.metrics
        )
        lines.append(f"| `{result.example.id}` | {metric_text} |")
    return "\n".join(lines) + "\n"


def comparison_to_markdown(report: ComparisonReport) -> str:
    lines = [
        "# Evaluation comparison",
        "",
        f"- Baseline: `{report.baseline_run_id}`",
        f"- Candidate: `{report.candidate_run_id}`",
        f"- Result: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "| Metric | Baseline | Candidate | Delta | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in report.comparisons:
        old = "n/a" if item.baseline is None else f"{item.baseline:.3f}"
        new = "n/a" if item.candidate is None else f"{item.candidate:.3f}"
        delta = "n/a" if item.delta is None else f"{item.delta:+.3f}"
        lines.append(f"| `{item.name}` | {old} | {new} | {delta} | {item.status} |")
    return "\n".join(lines) + "\n"


def write_markdown(value: EvaluationRun | ComparisonReport, path: str | Path) -> None:
    content = run_to_markdown(value) if isinstance(value, EvaluationRun) else comparison_to_markdown(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
