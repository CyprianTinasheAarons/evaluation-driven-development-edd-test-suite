import tempfile
import unittest
from pathlib import Path

from edd_eval.evaluator import Evaluator
from edd_eval.models import Example
from edd_eval.reporting import read_run, run_to_markdown, write_json
from edd_eval.systems import StaticSystem


class ReportingTests(unittest.TestCase):
    def test_run_json_round_trip_and_markdown(self):
        examples = [Example(id="1", question="q", reference_answer="a")]
        run = Evaluator().evaluate(examples, StaticSystem({"q": "a"}, "fixture"), run_id="run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_json(run, path)
            loaded = read_run(path)
            self.assertEqual(loaded, run)
        self.assertIn("Aggregate scores", run_to_markdown(run))
