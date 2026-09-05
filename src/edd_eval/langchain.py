"""Optional LangChain runnable support without importing LangChain at import time.

LangChain's runnable protocol is intentionally structural here. This keeps the
core package usable without LangChain while allowing a chain or agent with an
``invoke(input, config=...)`` method to be evaluated directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .models import SystemOutput


def _text(value: Any) -> str:
    """Extract text from strings, message-like objects, or content blocks."""

    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, Mapping) and isinstance(value.get("text"), str):
        return value["text"]
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _text(content)
    return str(value)


class LangChainSystem:
    """Adapt a LangChain chain, runnable, or agent to ``SystemUnderTest``.

    ``input_builder`` is useful for applications whose chain expects a shape
    other than ``{"question": question}``. Callback handlers are passed through
    LangChain's standard ``config`` argument, enabling tracing integrations.
    """

    def __init__(
        self,
        runnable: Any,
        *,
        name: str = "langchain_system",
        input_key: str = "question",
        output_key: str | None = None,
        contexts_key: str | None = None,
        callbacks: Sequence[Any] | None = None,
        input_builder: Callable[[str, Mapping[str, object]], Any] | None = None,
    ):
        if not hasattr(runnable, "invoke") and not callable(runnable):
            raise TypeError("runnable must provide invoke() or be callable")
        self.runnable = runnable
        self.name = name
        self.input_key = input_key
        self.output_key = output_key
        self.contexts_key = contexts_key
        self.callbacks = list(callbacks or [])
        self.input_builder = input_builder

    def run(self, question: str, metadata: Mapping[str, object]) -> SystemOutput:
        input_value = (
            self.input_builder(question, metadata)
            if self.input_builder
            else {self.input_key: question}
        )
        config: dict[str, Any] = {"metadata": dict(metadata)}
        if self.callbacks:
            config["callbacks"] = self.callbacks
        if hasattr(self.runnable, "invoke"):
            raw = self.runnable.invoke(input_value, config=config)
        else:
            raw = self.runnable(input_value)
        return self._to_output(raw)

    def _to_output(self, raw: Any) -> SystemOutput:
        if isinstance(raw, Mapping):
            answer_key = self.output_key
            if answer_key is None:
                answer_key = next(
                    (key for key in ("answer", "output", "text", "content") if key in raw),
                    None,
                )
            if answer_key is None:
                raise TypeError("LangChain output mapping must contain answer, output, text, or content")
            answer = _text(raw[answer_key])
            contexts_value = raw.get(self.contexts_key) if self.contexts_key else None
            if contexts_value is None:
                contexts_value = raw.get("contexts", raw.get("retrieved_contexts", []))
            if isinstance(contexts_value, str):
                contexts = (contexts_value,)
            elif isinstance(contexts_value, Sequence):
                contexts = tuple(_text(item) for item in contexts_value)
            else:
                raise TypeError("LangChain contexts must be a string or sequence")
            output_metadata = raw.get("metadata", {})
            if not isinstance(output_metadata, Mapping):
                output_metadata = {}
            return SystemOutput(answer=answer, contexts=contexts, metadata=dict(output_metadata))
        return SystemOutput(answer=_text(raw))
