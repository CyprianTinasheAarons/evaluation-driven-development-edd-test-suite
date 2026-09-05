"""Lazy DeepEval metric integration.

DeepEval stays optional: importing this module never imports the provider. The
adapter only requires it when a metric is actually evaluated, so deterministic
tests and CI can run without an LLM provider or API key.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .integrations import OptionalProviderError, require_provider
from .metrics import Metric
from .models import Example, MetricResult, Prediction


def _default_test_case(example: Example, prediction: Prediction) -> Any:
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise OptionalProviderError(
            "DeepEval support requires the optional dependency; install with `pip install 'edd-eval[deepeval]'`"
        ) from exc

    values: dict[str, Any] = {
        "input": example.question,
        "actual_output": prediction.answer,
    }
    if example.reference_answer is not None:
        values["expected_output"] = example.reference_answer
    if prediction.contexts:
        values["retrieval_context"] = list(prediction.contexts)
    if example.reference_contexts:
        values["context"] = list(example.reference_contexts)
    return LLMTestCase(**values)


class DeepEvalMetricAdapter(Metric):
    """Use a configured DeepEval metric inside the EDD evaluator."""

    def __init__(
        self,
        metric: Any,
        *,
        name: str | None = None,
        test_case_factory: Callable[[Example, Prediction], Any] | None = None,
    ):
        if metric is None or not hasattr(metric, "measure"):
            raise TypeError("metric must provide DeepEval's measure(test_case) method")
        self.provider_metric = metric
        provider_name = getattr(metric, "name", None) or metric.__class__.__name__
        self.name = name or str(provider_name).lower()
        self._uses_default_factory = test_case_factory is None
        self.test_case_factory = test_case_factory or _default_test_case

    def evaluate(self, example: Example, prediction: Prediction) -> MetricResult:
        # Custom factories are useful for offline tests and provider wrappers.
        if self._uses_default_factory:
            require_provider("deepeval", "pip install 'edd-eval[deepeval]'")
        test_case = self.test_case_factory(example, prediction)
        measured = self.provider_metric.measure(test_case)
        if inspect.isawaitable(measured):
            raise RuntimeError("DeepEval metric.measure returned an awaitable; use a synchronous metric adapter")
        score = getattr(self.provider_metric, "score", measured)
        if score is None:
            raise ValueError(f"DeepEval metric {self.name!r} did not expose a score")
        reason = str(getattr(self.provider_metric, "reason", ""))
        details = {"provider": "deepeval", "metric_class": self.provider_metric.__class__.__name__}
        threshold = getattr(self.provider_metric, "threshold", None)
        if threshold is not None:
            details["threshold"] = threshold
        return MetricResult(self.name, float(score), reason, details=details)
