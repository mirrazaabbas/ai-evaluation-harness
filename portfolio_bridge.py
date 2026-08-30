"""Evaluate versioned evidence emitted by the RAG and Agent portfolio projects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from advanced_metrics import (
    citation_precision_recall,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
    tool_call_accuracy,
)

SCHEMA_VERSION = "portfolio-evidence/v1"


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if not isinstance(record.get("producer"), str) or not record["producer"].strip():
        errors.append("producer must be a non-empty string")
    for field in ("query", "output"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{field} must be a non-empty string")
    for field in ("retrieved_ids", "citations", "context", "tool_calls"):
        if not isinstance(record.get(field), list):
            errors.append(f"{field} must be a list")
    latency = record.get("latency_ms")
    if latency is not None and (not isinstance(latency, (int, float)) or latency < 0):
        errors.append("latency_ms must be a non-negative number when provided")
    return errors


def evaluate_record(record: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    errors = validate_record(record)
    if errors:
        raise ValueError("Invalid evidence record: " + "; ".join(errors))

    metrics: dict[str, float] = {}
    reference = expected.get("reference")
    if isinstance(reference, str):
        metrics["answer_token_f1"] = token_f1(reference, record["output"])

    relevant_ids = expected.get("relevant_ids")
    if isinstance(relevant_ids, list):
        retrieved = [str(item) for item in record["retrieved_ids"]]
        k = int(expected.get("k", max(1, len(retrieved))))
        metrics["retrieval_recall_at_k"] = recall_at_k(
            [str(item) for item in relevant_ids], retrieved, k
        )
        metrics["retrieval_mrr"] = reciprocal_rank([str(item) for item in relevant_ids], retrieved)
        metrics["retrieval_ndcg_at_k"] = ndcg_at_k(
            [str(item) for item in relevant_ids], retrieved, k
        )

    expected_citations = expected.get("expected_citations")
    if isinstance(expected_citations, list):
        citation_scores = citation_precision_recall(
            [str(item) for item in expected_citations],
            [str(item) for item in record["citations"]],
        )
        metrics["citation_precision"] = citation_scores["precision"]
        metrics["citation_recall"] = citation_scores["recall"]

    expected_tools = expected.get("expected_tool_calls")
    if isinstance(expected_tools, list):
        metrics["tool_call_accuracy"] = tool_call_accuracy(expected_tools, record["tool_calls"])

    overall = sum(metrics.values()) / len(metrics) if metrics else 0.0
    threshold = float(expected.get("pass_threshold", 0.75))
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": record["producer"],
        "query": record["query"],
        "metrics": metrics,
        "overall_score": round(overall, 6),
        "pass_threshold": threshold,
        "passed": overall >= threshold,
        "latency_ms": record.get("latency_ms"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a portfolio-evidence/v1 record.")
    parser.add_argument("record", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_record(_read_json(args.record), _read_json(args.expected))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
