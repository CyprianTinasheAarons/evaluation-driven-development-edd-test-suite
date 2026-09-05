import json
import tempfile
import unittest
from pathlib import Path

from edd_eval.cli import main


class CliTests(unittest.TestCase):
    def test_validate_and_evaluate_commands_write_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "question": "What is one?",
                        "reference_answer": "one",
                        "metadata": {"demo_answer": "one", "demo_contexts": ["one"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_path = root / "nested" / "run.json"
            markdown_path = root / "nested" / "run.md"
            self.assertEqual(main(["validate", str(dataset)]), 0)
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--dataset",
                        str(dataset),
                        "--system",
                        "edd_eval.demo:system",
                        "--output",
                        str(run_path),
                        "--markdown",
                        str(markdown_path),
                        "--run-id",
                        "cli-run",
                    ]
                ),
                0,
            )
            self.assertTrue(run_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertEqual(json.loads(run_path.read_text(encoding="utf-8"))["run_id"], "cli-run")

