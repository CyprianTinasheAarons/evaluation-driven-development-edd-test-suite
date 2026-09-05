"""Golden-dataset loading, validation, and deterministic sampling."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Example


class DatasetError(ValueError):
    """Raised when a dataset is malformed or cannot be decoded."""


def _parse_rows(rows: Iterable[Any], source: Path) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise DatasetError(f"{source}: record {line_number} must be a JSON object")
        if "id" not in row:
            row = {**row, "id": f"example-{line_number}"}
        try:
            example = Example.from_mapping(row, line_number=line_number)
        except ValueError as exc:
            raise DatasetError(f"{source}: {exc}") from exc
        if example.id in seen:
            raise DatasetError(f"{source}: duplicate example id {example.id!r}")
        seen.add(example.id)
        examples.append(example)
    if not examples:
        raise DatasetError(f"{source}: dataset is empty")
    return examples


def load_dataset(path: str | Path) -> list[Example]:
    """Load JSONL or JSON-array datasets and validate every record."""

    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            rows = []
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise DatasetError(f"{source}: invalid JSON on line {line_number}: {exc.msg}") from exc
        else:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload = payload.get("examples")
            rows = payload
            if not isinstance(rows, list):
                raise DatasetError(f"{source}: expected a JSON array or an object with an 'examples' array")
    except FileNotFoundError as exc:
        raise DatasetError(f"{source}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{source}: invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise DatasetError(f"{source}: {exc}") from exc
    return _parse_rows(rows, source)


def save_dataset(examples: Iterable[Example], path: str | Path) -> None:
    """Write a dataset as UTF-8 JSONL."""

    destination = Path(path)
    rows = list(examples)
    if not rows:
        raise DatasetError("cannot write an empty dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in rows:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def sample_dataset(examples: Iterable[Example], *, limit: int | None = None, seed: int = 0) -> list[Example]:
    """Take a reproducible sample without changing the input order otherwise."""

    result = list(examples)
    if limit is None:
        return result
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit >= len(result):
        return result
    generator = random.Random(seed)
    indexes = sorted(generator.sample(range(len(result)), limit))
    return [result[index] for index in indexes]
