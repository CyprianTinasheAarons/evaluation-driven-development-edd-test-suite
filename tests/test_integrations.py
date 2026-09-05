import json
import tempfile
import unittest
from pathlib import Path

from edd_eval.evaluator import Evaluator
from edd_eval.integrations import export_langsmith_jsonl
from edd_eval.models import Example
from edd_eval.systems import StaticSystem


class IntegrationTests(unittest.TestCase):
    def test_langsmith_export_preserves_inputs_outputs_references_and_scores(self):
        example = Example(id="1", question="q", reference_answer="a", metadata={"tag": "smoke"})
        run = Evaluator().evaluate([example], StaticSystem({"q": {"answer": "a"}}, "fixture"), run_id="run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.jsonl"
            export_langsmith_jsonl(run, path)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["inputs"]["question"], "q")
        self.assertEqual(row["outputs"]["answer"], "a")
        self.assertEqual(row["reference_outputs"]["answer"], "a")
        self.assertEqual(row["metadata"]["tag"], "smoke")
        self.assertEqual(row["metrics"]["answer_correctness"], 1.0)
