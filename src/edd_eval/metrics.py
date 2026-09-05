"""Deterministic baseline metrics and a small extension point for external judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from .models import Example, MetricResult, Prediction
from .text import normalize, token_f1, token_recall, token_set


class Metric(ABC):
    """Metric interface implemented by local and provider-backed metrics."""

    name: str

    @abstractmethod
    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        raise NotImplementedError


class ExactMatch(Metric):
    name = "exact_match"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        if example.reference_answer is None:
            return MetricResult(self.name, None, "No reference answer supplied", applicable=False)
        score = float(normalize(prediction.answer) == normalize(example.reference_answer))
        return MetricResult(self.name, score, "Normalized answer equality")


class AnswerCorrectness(Metric):
    """A deterministic lexical proxy for answer correctness.

    This is intentionally transparent rather than pretending to be a semantic
    judge. Teams can add a model-based judge through ``FunctionMetric``.
    """

    name = "answer_correctness"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        if example.reference_answer is None:
            return MetricResult(self.name, None, "No reference answer supplied", applicable=False)
        score = token_f1(prediction.answer, example.reference_answer)
        return MetricResult(self.name, score, "Token-set F1 against the reference answer")


class AnswerRelevance(Metric):
    name = "answer_relevance"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        question_tokens = token_set(example.question)
        answer_tokens = token_set(prediction.answer)
        if not question_tokens:
            return MetricResult(self.name, None, "Question contains no comparable tokens", applicable=False)
        if not answer_tokens:
            return MetricResult(self.name, 0.0, "Answer is empty")
        score = len(question_tokens & answer_tokens) / len(question_tokens)
        return MetricResult(self.name, score, "Question-token coverage in the answer")


class ContextRecall(Metric):
    name = "context_recall"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        if not example.reference_contexts:
            return MetricResult(self.name, None, "No reference contexts supplied", applicable=False)
        reference = " ".join(example.reference_contexts)
        retrieved = " ".join(prediction.contexts)
        score = token_recall(retrieved, reference)
        return MetricResult(self.name, score, "Reference-context token coverage")


class ContextPrecision(Metric):
    name = "context_precision"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        if not example.reference_contexts:
            return MetricResult(self.name, None, "No reference contexts supplied", applicable=False)
        if not prediction.contexts:
            return MetricResult(self.name, 0.0, "No contexts were retrieved")
        reference_tokens = token_set(" ".join(example.reference_contexts))
        relevant = sum(bool(token_set(context) & reference_tokens) for context in prediction.contexts)
        score = relevant / len(prediction.contexts)
        return MetricResult(
            self.name,
            score,
            "Fraction of retrieved contexts with lexical reference overlap",
            details={"relevant_contexts": relevant, "retrieved_contexts": len(prediction.contexts)},
        )


class Groundedness(Metric):
    name = "groundedness"

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        del example
        answer_words = token_set(prediction.answer)
        context_words = token_set(" ".join(prediction.contexts))
        if not answer_words:
            return MetricResult(self.name, 0.0, "Answer is empty")
        if not context_words:
            return MetricResult(self.name, 0.0, "No retrieved contexts were provided")
        score = len(answer_words & context_words) / len(answer_words)
        return MetricResult(self.name, score, "Answer-token coverage in retrieved contexts")


class FunctionMetric(Metric):
    """Wrap a project-specific judge without coupling the core to an SDK.

    The callback may return a float or a complete ``MetricResult``. This is the
    bridge used for Ragas, ARES, LangSmith, or an in-house evaluator.
    """

    def __init__(self, name: str, callback: Callable[[Example, Prediction], float | MetricResult], description: str = ""):
        if not name.strip():
            raise ValueError("metric name must be non-empty")
        self.name = name
        self.callback = callback
        self.description = description

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        result = self.callback(example, prediction)
        if isinstance(result, MetricResult):
            if result.metric != self.name:
                raise ValueError(f"callback returned metric {result.metric!r}, expected {self.name!r}")
            return result
        score = float(result)
        return MetricResult(self.name, score, self.description)


def default_metrics() -> tuple[Metric, ...]:
    """Return an explicit, deterministic starter metric set."""

    return (AnswerCorrectness(), ContextRecall(), ContextPrecision(), Groundedness(), AnswerRelevance())


def metric_names(metrics: Iterable[Metric]) -> list[str]:
    return [metric.name for metric in metrics]
