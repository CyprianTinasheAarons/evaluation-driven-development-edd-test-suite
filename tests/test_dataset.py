import json
import tempfile
import unittest
from pathlib import Path

from edd_eval.dataset import DatasetError, load_dataset, sample_dataset, save_dataset


class DatasetTests(unittest.TestCase):
    def test_loads_jsonl_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text(json.dumps({"question": "q", "reference_answer": "a"}) + "\n", encoding="utf-8")
            examples = load_dataset(source)
            self.assertEqual(examples[0].id, "example-1")
            destination = Path(directory) / "copy.jsonl"
            save_dataset(examples, destination)
            self.assertEqual(load_dataset(destination), examples)

    def test_rejects_bad_records_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text(
                '{"id":"same","question":"q"}\n{"id":"same","question":"q2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "duplicate"):
                load_dataset(source)

    def test_sampling_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.json"
            source.write_text(json.dumps({"examples": [{"id": str(i), "question": str(i)} for i in range(10)]}), encoding="utf-8")
            examples = load_dataset(source)
            self.assertEqual(sample_dataset(examples, limit=4, seed=7), sample_dataset(examples, limit=4, seed=7))
