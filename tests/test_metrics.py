import unittest

from edd_eval.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    ExactMatch,
    FunctionMetric,
    Groundedness,
)
from edd_eval.models import Example, MetricResult, Prediction


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.example = Example(
            id="1",
            question="What is a cat?",
            reference_answer="A cat is a small animal.",
            reference_contexts=("A cat is a small animal.",),
        )
        self.prediction = Prediction(
            example_id="1",
            answer="A cat is a small animal.",
            contexts=("A cat is a small animal.",),
        )

    def test_exact_match_normalizes_case_and_punctuation(self):
        result = ExactMatch().evaluate(self.example, Prediction("1", "a CAT is a small animal!"))
        self.assertEqual(result.score, 1.0)

    def test_reference_metrics_score_perfect_prediction(self):
        for metric in (AnswerCorrectness(), ContextRecall(), ContextPrecision(), Groundedness()):
            self.assertEqual(metric.evaluate(self.example, self.prediction).score, 1.0, metric.name)

    def test_missing_reference_is_not_applicable(self):
        example = Example(id="1", question="q")
        result = AnswerCorrectness().evaluate(example, Prediction("1", "a"))
        self.assertFalse(result.applicable)
        self.assertIsNone(result.score)

    def test_function_metric_supports_external_judges(self):
        metric = FunctionMetric("judge", lambda example, prediction: MetricResult("judge", 0.75, "fixture"))
        self.assertEqual(metric.evaluate(self.example, self.prediction).score, 0.75)
