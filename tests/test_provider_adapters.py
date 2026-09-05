import unittest

from edd_eval.deepeval import DeepEvalMetricAdapter
from edd_eval.langchain import LangChainSystem
from edd_eval.models import Example, Prediction


class FakeRunnable:
    def __init__(self):
        self.calls = []

    def invoke(self, value, config=None):
        self.calls.append((value, config))
        return {"answer": "The answer", "contexts": ["The supporting context"]}


class FakeDeepEvalMetric:
    name = "faithfulness"
    threshold = 0.7

    def __init__(self):
        self.score = None
        self.reason = None

    def measure(self, test_case):
        self.seen_test_case = test_case
        self.score = 0.9
        self.reason = "Supported by context"
        return None


class ProviderAdapterTests(unittest.TestCase):
    def test_langchain_adapter_forwards_input_metadata_callbacks_and_outputs(self):
        runnable = FakeRunnable()
        callback = object()
        system = LangChainSystem(runnable, callbacks=[callback])
        output = system.run("What?", {"trace_id": "abc"})
        self.assertEqual(output.answer, "The answer")
        self.assertEqual(output.contexts, ("The supporting context",))
        self.assertEqual(runnable.calls[0][0], {"question": "What?"})
        self.assertEqual(runnable.calls[0][1]["metadata"], {"trace_id": "abc"})
        self.assertEqual(runnable.calls[0][1]["callbacks"], [callback])

    def test_deepeval_adapter_records_score_reason_and_test_case(self):
        example = Example(id="1", question="What?", reference_answer="The answer", reference_contexts=("Context",))
        prediction = Prediction("1", "The answer", ("Context",))
        metric = FakeDeepEvalMetric()
        adapter = DeepEvalMetricAdapter(metric, test_case_factory=lambda e, p: {"input": e.question, "output": p.answer})
        result = adapter.evaluate(example, prediction)
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.reason, "Supported by context")
        self.assertEqual(result.details["provider"], "deepeval")
