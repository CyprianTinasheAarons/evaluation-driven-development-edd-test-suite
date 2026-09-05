"""Command-line entry point for local evaluation and CI regression gates."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .comparison import compare_runs, summarize_run
from .dataset import DatasetError, load_dataset, sample_dataset
from .evaluator import Evaluator
from .reporting import read_run, write_json, write_markdown
from .systems import CallableSystem


def _load_target(spec: str) -> object:
    if ":" not in spec:
        raise ValueError("system target must use the form module:attribute")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("system target must use the form module:attribute")
    target = getattr(importlib.import_module(module_name), attribute)
    if isinstance(target, type):
        target = target()
    if hasattr(target, "run") and hasattr(target, "name"):
        return target
    if callable(target):
        return CallableSystem(target)
    raise ValueError(f"{spec!r} is not a system object or callable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edd-eval", description="Run reproducible evaluations for AI systems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a JSONL or JSON dataset")
    validate.add_argument("dataset", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a system against a golden dataset")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--system", required=True, help="module:attribute system target")
    evaluate.add_argument("--output", required=True, type=Path, help="run JSON output")
    evaluate.add_argument("--markdown", type=Path, help="optional Markdown report")
    evaluate.add_argument("--run-id", help="stable run id, useful in CI")
    evaluate.add_argument("--limit", type=int, help="evaluate a deterministic sample of this size")
    evaluate.add_argument("--seed", type=int, default=0)

    compare = subparsers.add_parser("compare", help="compare a candidate run with a baseline")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--tolerance", type=float, default=0.0)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--markdown", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            examples = load_dataset(args.dataset)
            print(f"valid: {len(examples)} examples")
            return 0
        if args.command == "evaluate":
            examples = sample_dataset(load_dataset(args.dataset), limit=args.limit, seed=args.seed)
            system = _load_target(args.system)
            run = Evaluator().evaluate(
                examples,
                system,
                dataset_name=args.dataset.name,
                run_id=args.run_id,
                config={"sample_limit": args.limit, "sample_seed": args.seed},
            )
            write_json(run, args.output)
            if args.markdown:
                write_markdown(run, args.markdown)
            summary = summarize_run(run)
            scores = ", ".join(
                f"{name}={item.mean:.3f}" for name, item in summary.items() if item.mean is not None
            )
            print(f"evaluated {len(run.results)} examples: {scores}")
            return 0
        if args.command == "compare":
            report = compare_runs(read_run(args.baseline), read_run(args.candidate), tolerance=args.tolerance)
            write_json(report, args.output)
            if args.markdown:
                write_markdown(report, args.markdown)
            for item in report.comparisons:
                delta = "n/a" if item.delta is None else f"{item.delta:+.3f}"
                print(f"{item.name}: {item.status} ({delta})")
            return 0 if report.passed else 1
    except (DatasetError, EvaluationError, ImportError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
