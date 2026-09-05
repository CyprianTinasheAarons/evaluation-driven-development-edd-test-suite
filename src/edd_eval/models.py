"""Small, serializable data models shared by the evaluator and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def _as_strings(value: Any, field_name: str) -> tuple[str, ...]:
    """Convert a JSON list of strings to an immutable tuple with a useful error."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Example:
    """One golden evaluation case.

    ``reference_answer`` and ``reference_contexts`` are optional because some
    metrics (for example groundedness) only need the generated answer. Metrics
    that require a reference mark themselves as not applicable when it is absent.
    """

    id: str
    question: str
    reference_answer: str | None = None
    reference_contexts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, line_number: int | None = None) -> "Example":
        """Build and validate an example from a JSON-compatible mapping."""

        location = f" on line {line_number}" if line_number else ""
        example_id = value.get("id")
        question = value.get("question")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"id must be a non-empty string{location}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question must be a non-empty string{location}")

        reference_answer = value.get("reference_answer", value.get("answer"))
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise ValueError(f"reference_answer must be a string or null{location}")

        reference_contexts = value.get("reference_contexts", value.get("contexts", []))
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be an object{location}")

        try:
            contexts = _as_strings(reference_contexts, "reference_contexts")
        except ValueError as exc:
            raise ValueError(f"{exc}{location}") from exc

        return cls(
            id=example_id,
            question=question,
            reference_answer=reference_answer,
            reference_contexts=contexts,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "reference_contexts": list(self.reference_contexts),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SystemOutput:
    """The provider-independent output returned by a system under test."""

    answer: str
    contexts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        if not all(isinstance(item, str) for item in self.contexts):
            raise TypeError("contexts must contain only strings")

    @classmethod
    def from_value(cls, value: Any) -> "SystemOutput":
        """Coerce common adapter return values into ``SystemOutput``."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(answer=value)
        if isinstance(value, Mapping):
            answer = value.get("answer", value.get("response"))
            if not isinstance(answer, str):
                raise TypeError("system output mapping must contain a string 'answer'")
            contexts = _as_strings(value.get("contexts", value.get("retrieved_contexts", [])), "contexts")
            metadata = value.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise TypeError("system output metadata must be an object")
            return cls(answer=answer, contexts=contexts, metadata=dict(metadata))
        raise TypeError("system output must be a string, mapping, or SystemOutput")

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "contexts": list(self.contexts), "metadata": dict(self.metadata)}


@dataclass(frozen=True, slots=True)
class Prediction:
    """A system output associated with a golden example."""

    example_id: str
    answer: str
    contexts: tuple[str, ...] = ()
    latency_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_output(cls, example_id: str, output: Any, latency_ms: float) -> "Prediction":
        result = SystemOutput.from_value(output)
        return cls(
            example_id=example_id,
            answer=result.answer,
            contexts=result.contexts,
            latency_ms=round(max(0.0, latency_ms), 3),
            metadata=dict(result.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "answer": self.answer,
            "contexts": list(self.contexts),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One metric result. A non-applicable metric is excluded from aggregates."""

    metric: str
    score: float | None
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    applicable: bool = True

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("metric score must be between 0 and 1")
        if not self.applicable and self.score is not None:
            raise ValueError("non-applicable metric results must have score=None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "score": self.score,
            "reason": self.reason,
            "details": dict(self.details),
            "applicable": self.applicable,
        }


@dataclass(frozen=True, slots=True)
class ExampleResult:
    example: Example
    prediction: Prediction
    metrics: tuple[MetricResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "prediction": self.prediction.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Complete, JSON-serializable result of evaluating one system version."""

    run_id: str
    system_name: str
    dataset_name: str
    created_at: str
    duration_ms: float
    results: tuple[ExampleResult, ...]
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        system_name: str,
        dataset_name: str,
        duration_ms: float,
        results: list[ExampleResult],
        config: Mapping[str, Any] | None = None,
    ) -> "EvaluationRun":
        return cls(
            run_id=run_id,
            system_name=system_name,
            dataset_name=dataset_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(max(0.0, duration_ms), 3),
            results=tuple(results),
            config=dict(config or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "system_name": self.system_name,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
            "config": dict(self.config),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRun":
        try:
            results: list[ExampleResult] = []
            for item in value["results"]:
                example = Example.from_mapping(item["example"])
                prediction_value = item["prediction"]
                prediction = Prediction(
                    example_id=str(prediction_value["example_id"]),
                    answer=str(prediction_value["answer"]),
                    contexts=_as_strings(prediction_value.get("contexts", []), "contexts"),
                    latency_ms=float(prediction_value.get("latency_ms", 0.0)),
                    metadata=dict(prediction_value.get("metadata", {})),
                )
                metrics = tuple(
                    MetricResult(
                        metric=str(metric["metric"]),
                        score=None if metric.get("score") is None else float(metric["score"]),
                        reason=str(metric.get("reason", "")),
                        details=dict(metric.get("details", {})),
                        applicable=bool(metric.get("applicable", True)),
                    )
                    for metric in item["metrics"]
                )
                results.append(ExampleResult(example=example, prediction=prediction, metrics=metrics))
            return cls(
                run_id=str(value["run_id"]),
                system_name=str(value["system_name"]),
                dataset_name=str(value["dataset_name"]),
                created_at=str(value["created_at"]),
                duration_ms=float(value.get("duration_ms", 0.0)),
                results=tuple(results),
                config=dict(value.get("config", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid evaluation run: {exc}") from exc
