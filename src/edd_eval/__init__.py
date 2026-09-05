"""Evaluation-driven development primitives for AI systems."""

from .comparison import ComparisonReport, compare_runs, summarize_run
from .dataset import DatasetError, load_dataset, save_dataset
from .evaluator import EvaluationError, Evaluator
from .deepeval import DeepEvalMetricAdapter
from .langchain import LangChainSystem
from .metrics import (
    AnswerCorrectness,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    ExactMatch,
    Groundedness,
    default_metrics,
)
from .models import EvaluationRun, Example, MetricResult, Prediction, SystemOutput
from .systems import CallableSystem, StaticSystem, SystemUnderTest

__all__ = [
    "AnswerCorrectness",
    "AnswerRelevance",
    "CallableSystem",
    "ComparisonReport",
    "ContextPrecision",
    "ContextRecall",
    "DatasetError",
    "DeepEvalMetricAdapter",
    "EvaluationError",
    "EvaluationRun",
    "Evaluator",
    "ExactMatch",
    "Example",
    "Groundedness",
    "LangChainSystem",
    "MetricResult",
    "Prediction",
    "StaticSystem",
    "SystemOutput",
    "SystemUnderTest",
    "compare_runs",
    "default_metrics",
    "load_dataset",
    "save_dataset",
    "summarize_run",
]
