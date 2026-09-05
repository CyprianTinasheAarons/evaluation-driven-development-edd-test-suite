# Evaluation-Driven Development Test Suite

A small, reproducible evaluation harness for AI applications. It turns a golden
dataset and a system under test into a versioned run with per-example scores,
human-readable reports, and a CI-friendly regression gate.

The core has no runtime dependencies, network calls, or API-key requirements.
Ragas, ARES, LangSmith, DeepEval, or an in-house judge can be connected through
the optional adapters and `FunctionMetric` when a project needs model-based
scoring.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

edd-eval validate examples/golden.jsonl
edd-eval evaluate \
  --dataset examples/golden.jsonl \
  --system edd_eval.demo:system \
  --output reports/demo.json \
  --markdown reports/demo.md \
  --run-id demo-v1
```

The demo is deliberately offline. It reads facts from each example's
`metadata.demo_answer` and `metadata.demo_contexts`, making it a safe smoke test
for the complete pipeline.

## Golden dataset format

JSONL is the recommended format. Each record needs a unique `id` and a
non-empty `question`. References are optional overall, but reference-based
metrics are marked not applicable when their reference is absent.

```json
{
  "id": "python-functions",
  "question": "What keyword defines a function in Python?",
  "reference_answer": "The def keyword defines a function in Python.",
  "reference_contexts": [
    "In Python, the def keyword starts a function definition."
  ],
  "metadata": {"tenant": "docs"}
}
```

The loader also accepts a JSON array or `{ "examples": [...] }`. `answer` and
`contexts` are accepted as aliases for the reference fields.

## Built-in metrics

- `answer_correctness`: transparent token-set F1 against the reference answer.
- `context_recall`: reference-context token coverage.
- `context_precision`: fraction of retrieved contexts with reference overlap.
- `groundedness`: answer-token coverage in retrieved contexts.
- `answer_relevance`: question-token coverage in the answer.
- `exact_match`: normalized answer equality, available as a library metric.

These are deterministic baseline metrics, useful for fast feedback and
regression detection. They are lexical proxies—not substitutes for a semantic
judge. Add a provider-backed judge with `FunctionMetric`:

```python
from edd_eval.metrics import FunctionMetric

judge = FunctionMetric(
    "faithfulness_judge",
    lambda example, prediction: your_ragas_or_ares_score(example, prediction),
)
run = Evaluator(metrics=[*default_metrics(), judge]).evaluate(examples, system)
```

## LangChain and DeepEval support

The optional integrations follow the current provider patterns:

```bash
python -m pip install -e '.[langchain]'
python -m pip install -e '.[deepeval]'
```

LangChain chains, agents, and runnables can be evaluated directly. The adapter
passes the question as `{ "question": ... }`, forwards metadata and callback
handlers through LangChain's `config`, and accepts string, message-like, or
mapping outputs:

```python
from edd_eval import Evaluator, LangChainSystem, load_dataset

chain = LangChainSystem(my_chain, callbacks=[langsmith_callback])
run = Evaluator().evaluate(load_dataset("examples/golden.jsonl"), chain)
```

Use `input_builder` for a chain-specific input shape, and `output_key` or
`contexts_key` when the chain uses non-standard output keys. This aligns with
[LangSmith's evaluation model](https://docs.langchain.com/langsmith/evaluation-quickstart):
dataset, target function, evaluators, and traceable experiment results.

DeepEval metrics can be placed in the same run alongside local metrics:

```python
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from edd_eval import DeepEvalMetricAdapter, Evaluator

metrics = [
    DeepEvalMetricAdapter(AnswerRelevancyMetric(threshold=0.7)),
    DeepEvalMetricAdapter(FaithfulnessMetric(threshold=0.7)),
]
run = Evaluator(metrics=metrics).evaluate(examples, chain)
```

The adapter creates a DeepEval `LLMTestCase` from each EDD example and
prediction, calls the metric's `measure()` method, and records its score,
reason, metric class, and threshold. See the [DeepEval introduction](https://deepeval.com/docs/introduction)
and [metric documentation](https://deepeval.com/docs/metrics-introduction) for
provider-specific metric requirements. The SDK remains lazy and optional, so
the offline suite does not need DeepEval installed.

## Compare runs in CI

```bash
edd-eval compare \
  --baseline reports/baseline.json \
  --candidate reports/demo.json \
  --tolerance 0.02 \
  --output reports/comparison.json \
  --markdown reports/comparison.md
```

The command exits `0` when no metric drops beyond the tolerance, `1` when a
regression is detected, and `2` for malformed input or an execution error.

## Python API

```python
from edd_eval import Evaluator, default_metrics, load_dataset
from edd_eval.systems import CallableSystem

examples = load_dataset("examples/golden.jsonl")
system = CallableSystem(lambda question: my_app.answer(question), name="my-app")
run = Evaluator(metrics=default_metrics()).evaluate(examples, system)
```

`EvaluationRun.to_dict()` is stable JSON-ready output. Use
`edd_eval.reporting.write_json`, `write_markdown`, and
`edd_eval.integrations.export_langsmith_jsonl` for storage/export.

## What problems does this help solve?

The suite helps teams address these 15 recurring problems in AI application
development:

1. No repeatable AI evaluation process.
2. Missing or inconsistent golden datasets.
3. Invalid dataset records and duplicate test cases.
4. Non-reproducible evaluation samples.
5. Answer-quality regressions between releases.
6. Retrieval recall failures.
7. Retrieval precision and irrelevant-context problems.
8. Unsupported or hallucinated answers.
9. Answers that do not address the question.
10. Inconsistent output formats from AI systems.
11. Difficulty evaluating LangChain chains and agents.
12. Difficulty using DeepEval metrics in one evaluation pipeline.
13. Lack of per-example scores and explanations.
14. No machine-readable or human-readable evaluation reports.
15. No automated CI quality gate for model changes.

The core framework is functional for deterministic evaluation and provider
adapter workflows. Semantic model judging, native Ragas/ARES execution, async
evaluation, agent trajectory scoring, and direct LangSmith experiment uploads
remain extension areas rather than built-in behavior.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

The suite tests dataset validation and round-tripping, deterministic sampling,
metric behavior, system failures, aggregation, regression detection, and report
serialization.

## Design notes from the supplied resources

The implementation follows the recurring engineering themes in the supplied
AI engineering, RAG, LLMOps, and observability books: define evaluation data
before optimization; keep retrieval and answer quality visible separately;
preserve traceable per-example outputs; make runs reproducible; and turn
quality changes into an explicit deployment signal. Those books are treated as
reference material only—this repository contains the executable specification.

Open-source (MIT). See [LICENSE](LICENSE).
