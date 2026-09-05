"""Dependency-free export and integration seams for external evaluation tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import EvaluationRun


def export_langsmith_jsonl(run: EvaluationRun, path: str | Path) -> None:
    """Export run cases in a LangSmith-compatible dataset-style JSONL shape.

    This deliberately writes an interchange file instead of importing the
    LangSmith SDK. A deployment can upload the file with its preferred SDK or
    API client, keeping local tests deterministic and credentials optional.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for result in run.results:
            row: dict[str, Any] = {
                "id": result.example.id,
                "inputs": {"question": result.example.question},
                "outputs": {"answer": result.prediction.answer, "contexts": list(result.prediction.contexts)},
                "reference_outputs": {
                    "answer": result.example.reference_answer,
                    "contexts": list(result.example.reference_contexts),
                },
                "metadata": {"run_id": run.run_id, **dict(result.example.metadata)},
                "metrics": {metric.metric: metric.score for metric in result.metrics if metric.applicable},
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class OptionalProviderError(RuntimeError):
    """Explain how to install/configure an optional provider dependency."""


def require_provider(module_name: str, install_hint: str | None = None) -> Any:
    """Import an optional provider with a focused error message."""

    try:
        return __import__(module_name)
    except ImportError as exc:
        hint = install_hint or f"pip install {module_name}"
        raise OptionalProviderError(f"Optional provider {module_name!r} is not installed; use {hint}") from exc
