import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from edd_eval.cli import main


def failing_system(question, metadata):
    raise RuntimeError("model backend unavailable")


def _write_dataset(path):
    path.write_text(
        json.dumps({"id": "one", "question": "What is one?", "reference_answer": "one"}) + "\n",
        encoding="utf-8",
    )


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

    def test_evaluate_returns_exit_code_2_when_system_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_dataset(dataset)
            run_path = root / "run.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "evaluate",
                        "--dataset",
                        str(dataset),
                        "--system",
                        f"{__name__}:failing_system",
                        "--output",
                        str(run_path),
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("failed for example 'one': model backend unavailable", stderr.getvalue())
            self.assertFalse(run_path.exists())

    def test_validate_returns_exit_code_2_for_missing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["validate", str(Path(directory) / "missing.jsonl")])
            self.assertEqual(exit_code, 2)
            self.assertIn("file not found", stderr.getvalue())
