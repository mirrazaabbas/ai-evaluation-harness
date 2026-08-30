"""Installed CLI for AI Evaluation Harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bridge import _read_json, evaluate_record
from suite_runner import _read_cases, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(prog="ai-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite = subparsers.add_parser("suite", help="run a heterogeneous benchmark suite")
    suite.add_argument("path", type=Path)
    suite.add_argument("--threshold", type=float, default=0.75)

    portfolio = subparsers.add_parser("portfolio", help="evaluate a portfolio evidence record")
    portfolio.add_argument("record", type=Path)
    portfolio.add_argument("expected", type=Path)

    args = parser.parse_args()
    if args.command == "suite":
        result = run_suite(_read_cases(args.path), pass_threshold=args.threshold)
    else:
        result = evaluate_record(_read_json(args.record), _read_json(args.expected))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
