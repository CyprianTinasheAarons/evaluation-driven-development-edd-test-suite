"""Adapters for systems under test."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from .models import Example, SystemOutput


@runtime_checkable
class SystemUnderTest(Protocol):
    """A minimal provider-neutral contract for an AI application."""

    name: str

    def run(self, question: str, metadata: Mapping[str, object]) -> SystemOutput | str | Mapping[str, object]:
        ...


class CallableSystem:
    """Adapt a function accepting ``question`` or ``question, metadata``."""

    def __init__(self, function: Callable[..., object], name: str | None = None):
        self.function = function
        self.name = name or getattr(function, "name", getattr(function, "__name__", "callable_system"))
        try:
            self._accepts_metadata = len(inspect.signature(function).parameters) >= 2
        except (TypeError, ValueError):
            self._accepts_metadata = True

    def run(self, question: str, metadata: Mapping[str, object]) -> object:
        if self._accepts_metadata:
            return self.function(question, metadata)
        return self.function(question)


class StaticSystem:
    """Useful for fixtures and for replaying provider outputs in CI."""

    def __init__(self, outputs: Mapping[str, object], name: str = "static_system"):
        self.outputs = dict(outputs)
        self.name = name

    def for_examples(self, examples: list[Example]) -> "StaticSystem":
        """Return a static system keyed by question for a quick fixture setup."""

        return StaticSystem({example.question: self.outputs[example.id] for example in examples}, name=self.name)

    def run(self, question: str, metadata: Mapping[str, object]) -> object:
        del metadata
        if question not in self.outputs:
            raise KeyError(f"no static output configured for question {question!r}")
        return self.outputs[question]
