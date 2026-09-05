"""Offline demo system used by the README and smoke tests."""

from __future__ import annotations

from collections.abc import Mapping

from .models import SystemOutput


def system(question: str, metadata: Mapping[str, object]) -> SystemOutput:
    """Answer from metadata-provided demo facts; no model or network is needed."""

    answer = metadata.get("demo_answer")
    contexts = metadata.get("demo_contexts", [])
    if isinstance(answer, str) and isinstance(contexts, list) and all(isinstance(item, str) for item in contexts):
        return SystemOutput(answer=answer, contexts=tuple(contexts), metadata={"question_length": len(question)})
    return SystemOutput(answer="I do not know.", contexts=())


system.name = "offline_demo"  # type: ignore[attr-defined]
