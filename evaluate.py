"""Deterministic evaluation harness for AI-system quality regression checks."""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "sample_cases.json"
PASS_THRESHOLD = 0.65


@dataclass(frozen=True)
class Score:
    case_id: str
    keyword_recall: float
    groundedness: float
    concision: float
    citation_coverage: float | None
    overall: float


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def keyword_recall(output: str, required: list[str]) -> float:
    if not required:
        return 1.0
    text = output.lower()
    return sum(term.lower() in text for term in required) / len(required)


def groundedness(output: str, context: str) -> float:
    out = words(output)
    ctx = words(context)
    return len(out & ctx) / len(out) if out else 0.0


def concision(output: str, max_words: int) -> float:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")
    count = len(output.split())
    return 1.0 if count <= max_words else max_words / count


def citation_coverage(output: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    return sum(citation in output for citation in expected) / len(expected)


def validate_case(case: dict[str, Any], index: int) -> None:
    for field in ("id", "output"):
        if field not in case or not isinstance(case[field], str) or not case[field].strip():
            raise ValueError(f"Case {index}: '{field}' must be a non-empty string.")
    if "required_terms" in case and not isinstance(case["required_terms"], list):
        raise ValueError(f"Case {index}: 'required_terms' must be a list.")
    if "expected_citations" in case and not isinstance(case["expected_citations"], list):
        raise ValueError(f"Case {index}: 'expected_citations' must be a list.")
    if "max_words" in case and (
        not isinstance(case["max_words"], int) or case["max_words"] <= 0
    ):
        raise ValueError(f"Case {index}: 'max_words' must be a positive integer.")
    for field in ("latency_ms", "cost_usd"):
        if field in case and (
            not isinstance(case[field], (int, float)) or case[field] < 0
        ):
            raise ValueError(f"Case {index}: '{field}' must be a non-negative number.")


def evaluate(case: dict[str, Any]) -> Score:
    recall = keyword_recall(case["output"], case.get("required_terms", []))
    ground = groundedness(case["output"], case.get("context", ""))
    concise = concision(case["output"], case.get("max_words", 120))
    expected_citations = case.get("expected_citations")
    citations = (
        citation_coverage(case["output"], expected_citations)
        if expected_citations is not None
        else None
    )
    overall = 0.45 * recall + 0.40 * ground + 0.15 * concise
    return Score(
        case["id"],
        round(recall, 3),
        round(ground, 3),
        round(concise, 3),
        round(citations, 3) if citations is not None else None,
        round(overall, 3),
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset contains invalid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset must be a non-empty JSON array.")
    for index, case in enumerate(raw, 1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        validate_case(case, index)
    return raw


def _optional_average(cases: list[dict[str, Any]], field: str) -> float | None:
    values = [float(case[field]) for case in cases if field in case]
    return round(sum(values) / len(values), 4) if values else None


def build_report(cases: list[dict[str, Any]], threshold: float = PASS_THRESHOLD) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    scores = [evaluate(case) for case in cases]
    citation_scores = [score.citation_coverage for score in scores if score.citation_coverage is not None]
    return {
        "cases": [asdict(score) for score in scores],
        "threshold": threshold,
        "average_overall": round(sum(score.overall for score in scores) / len(scores), 3),
        "pass_rate": round(sum(score.overall >= threshold for score in scores) / len(scores), 3),
        "average_citation_coverage": (
            round(sum(citation_scores) / len(citation_scores), 3) if citation_scores else None
        ),
        "average_latency_ms": _optional_average(cases, "latency_ms"),
        "average_cost_usd": _optional_average(cases, "cost_usd"),
    }


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "average_overall_delta": round(
            candidate["average_overall"] - baseline["average_overall"], 3
        ),
        "pass_rate_delta": round(candidate["pass_rate"] - baseline["pass_rate"], 3),
    }


def render_html_report(report: dict[str, Any], comparison: dict[str, float] | None = None) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(case['case_id']))}</td>"
        f"<td>{case['keyword_recall']:.3f}</td>"
        f"<td>{case['groundedness']:.3f}</td>"
        f"<td>{case['concision']:.3f}</td>"
        f"<td>{case['overall']:.3f}</td>"
        "</tr>"
        for case in report["cases"]
    )
    comparison_html = ""
    if comparison is not None:
        comparison_html = (
            "<h2>Baseline comparison</h2>"
            f"<p>Overall delta: {comparison['average_overall_delta']:+.3f}<br>"
            f"Pass-rate delta: {comparison['pass_rate_delta']:+.3f}</p>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>AI Evaluation Report</title>"
        "<style>body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px;}"
        "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;}"
        "th{text-align:left;}</style></head><body>"
        "<h1>AI Evaluation Report</h1>"
        f"<p>Average overall: <strong>{report['average_overall']:.3f}</strong><br>"
        f"Pass rate: <strong>{report['pass_rate']:.3f}</strong></p>"
        f"{comparison_html}"
        "<table><thead><tr><th>Case</th><th>Recall</th><th>Groundedness</th>"
        "<th>Concision</th><th>Overall</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI outputs against repeatable quality metrics.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, help="Optional baseline dataset for regression comparison")
    parser.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    parser.add_argument("--html", type=Path, dest="html_path", help="Write an HTML report to this path")
    args = parser.parse_args()
    try:
        report = build_report(load_cases(args.dataset), args.threshold)
        comparison = None
        if args.baseline:
            baseline = build_report(load_cases(args.baseline), args.threshold)
            comparison = compare_reports(baseline, report)
        if args.html_path:
            args.html_path.write_text(render_html_report(report, comparison), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    payload = {"report": report, "comparison": comparison} if comparison else report
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
