import unittest

from edd_eval.comparison import compare_runs, summarize_run
from edd_eval.evaluator import EvaluationError, Evaluator
from edd_eval.models import Example, SystemOutput
from edd_eval.systems import CallableSystem, StaticSystem


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            Example(id="1", question="one", reference_answer="yes", metadata={"answer": "yes"}),
            Example(id="2", question="two", reference_answer="yes", metadata={"answer": "yes"}),
        ]

    def test_evaluator_runs_system_and_aggregates(self):
        system = CallableSystem(lambda question, metadata: SystemOutput(str(metadata["answer"])), name="fixture")
        run = Evaluator().evaluate(self.examples, system, run_id="run-1")
        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(len(run.results), 2)
        self.assertEqual(summarize_run(run)["answer_correctness"].mean, 1.0)

    def test_system_failures_include_example_id(self):
        system = CallableSystem(lambda question: 42, name="bad")
        with self.assertRaisesRegex(EvaluationError, "example '1'"):
            Evaluator().evaluate(self.examples, system)

    def test_regression_gate_detects_drop_beyond_tolerance(self):
        baseline = Evaluator().evaluate(self.examples, StaticSystem({"one": "yes", "two": "yes"}, "base"), run_id="base")
        candidate = Evaluator().evaluate(self.examples, StaticSystem({"one": "no", "two": "yes"}, "candidate"), run_id="candidate")
        report = compare_runs(baseline, candidate, tolerance=0.1)
        self.assertFalse(report.passed)
        self.assertEqual(report.comparisons[0].status, "regressed")
