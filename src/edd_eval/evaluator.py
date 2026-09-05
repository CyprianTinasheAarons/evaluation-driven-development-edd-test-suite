"""The evaluation loop: run a system, score every case, and build a run."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Mapping

from .metrics import Metric, default_metrics, metric_names
from .models import EvaluationRun, Example, ExampleResult, Prediction
from .systems import SystemUnderTest


class EvaluationError(RuntimeError):
    """Raised when a system cannot produce a valid prediction."""


class Evaluator:
    def __init__(self, metrics: Iterable[Metric] | None = None):
        self.metrics = tuple(metrics or default_metrics())
        if not self.metrics:
            raise ValueError("at least one metric is required")
        names = metric_names(self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")

    def evaluate(
        self,
        examples: Iterable[Example],
        system: SystemUnderTest,
        *,
        dataset_name: str = "dataset",
        run_id: str | None = None,
        config: Mapping[str, object] | None = None,
    ) -> EvaluationRun:
        cases = list(examples)
        if not cases:
            raise ValueError("cannot evaluate an empty dataset")
        started = time.perf_counter()
        results: list[ExampleResult] = []
        for example in cases:
            prediction_started = time.perf_counter()
            try:
                raw_output = system.run(example.question, example.metadata)
                prediction = Prediction.from_output(
                    example.id,
                    raw_output,
                    (time.perf_counter() - prediction_started) * 1000,
                )
            except Exception as exc:  # Add the case id and preserve the original cause.
                raise EvaluationError(f"system {system.name!r} failed for example {example.id!r}: {exc}") from exc
            metric_results = tuple(metric.evaluate(example, prediction) for metric in self.metrics)
            results.append(ExampleResult(example=example, prediction=prediction, metrics=metric_results))

        return EvaluationRun.new(
            run_id=run_id or uuid.uuid4().hex,
            system_name=system.name,
            dataset_name=dataset_name,
            duration_ms=(time.perf_counter() - started) * 1000,
            results=results,
            config={"metrics": metric_names(self.metrics), **dict(config or {})},
        )
